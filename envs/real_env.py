"""Level 2 Physical 6-DoF SO-ARM101 Whiteboard Environment with 4-Corner Plane Calibration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from envs.base import (
    CAMERA_NAMES,
    HOME_PROPRIO_6D,
    STS3215_CENTER_TICK,
    WORKSPACE_SPEC,
    SO101WhiteboardEnv,
    is_xy_in_bounds,
    norm_gripper_to_raw,
    radians_to_ticks,
    raw_gripper_to_norm,
    ticks_to_radians,
    xy_to_canvas_px,
)
from envs.ik_solver import SO101IKSolver
from envs.trajectory import interpolate_waypoints_3d


class RealWhiteboardEnv(SO101WhiteboardEnv):
    """Physical SO-ARM101 environment matching SimWhiteboardEnv's exact API and coordinate space."""

    INK_COLOR_RGB: tuple[int, int, int] = (18, 42, 138)
    BG_COLOR_RGB: tuple[int, int, int] = (250, 250, 252)

    def __init__(
        self,
        serial_port: str = "/dev/tty.usbmodem5B790143431",
        baudrate: int = 1_000_000,
        calibration_path: str | Path | None = None,
        camera_indices: dict[str, int] | None = None,
        control_hz: int = 20,
        camera_resolution: tuple[int, int] = (640, 480),
        zero_offsets: np.ndarray | None = None,
        mock_hardware: bool = False,
    ) -> None:
        """Initialize the physical SO-ARM101 follower arm, plane calibration, and USB webcam."""
        self.serial_port = serial_port
        self.baudrate = int(baudrate)
        self.control_hz = int(control_hz)
        self.step_dt = 1.0 / float(self.control_hz)
        self.camera_resolution = camera_resolution
        self.mock_hardware = bool(mock_hardware)
        self.zero_offsets = (
            np.asarray(zero_offsets, dtype=np.float32)
            if zero_offsets is not None
            else np.full(6, float(STS3215_CENTER_TICK), dtype=np.float32)
        )
        self.camera_indices = camera_indices or {"overhead_cam": 0}

        if calibration_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            calibration_path = repo_root / "config" / "calibration.json"
        self.calibration_path = Path(calibration_path)
        self.plane_coeffs: tuple[float, float, float] = self._load_calibration()

        self.ik_solver = SO101IKSolver(site_name="pen_tip")

        self.canvas_h = int(WORKSPACE_SPEC["canvas_resolution_px"]["height"])
        self.canvas_w = int(WORKSPACE_SPEC["canvas_resolution_px"]["width"])
        self._canvas_pil = Image.new("RGB", (self.canvas_w, self.canvas_h), self.BG_COLOR_RGB)
        self._canvas_draw = ImageDraw.Draw(self._canvas_pil)
        self._prev_contact_px: tuple[int, int] | None = None

        self._mock_proprio = HOME_PROPRIO_6D.copy()
        self._port_handler: Any = None
        self._packet_handler: Any = None
        self._captures: dict[str, Any] = {}

        self.qpos_log: list[np.ndarray] = []
        self.pen_tip_xyz_log: list[np.ndarray] = []
        self.ink_contact_log: list[bool] = []
        self.ink_pixels_drawn: int = 0
        self.timelapse_frames: list[np.ndarray] = []

        if not self.mock_hardware:
            self._connect_hardware()

    def _load_calibration(self) -> tuple[float, float, float]:
        """Load 4-corner plane fit Z_board(X, Y) = a*X + b*Y + c from config/calibration.json."""
        if self.calibration_path.exists():
            with open(self.calibration_path, encoding="utf-8") as f:
                data = json.load(f)
            coeffs = data.get("plane_coeffs_abc", [0.0, 0.0, float(WORKSPACE_SPEC["z_draw_m"])])
            return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        return (0.0, 0.0, float(WORKSPACE_SPEC["z_draw_m"]))

    def apply_plane_calibration(self, xyz: np.ndarray) -> np.ndarray:
        """Adjust nominal Z coordinate by the calibrated physical desk tilt at (X, Y)."""
        a, b, c = self.plane_coeffs
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        z_nominal_draw = float(WORKSPACE_SPEC["z_draw_m"])
        z_calibrated_board = a * x + b * y + c
        dz = z_calibrated_board - z_nominal_draw
        return np.array([x, y, z + dz], dtype=np.float64)

    def _connect_hardware(self) -> None:
        """Open Feetech STS3215 USB serial bus and USB camera streams."""
        try:
            import scservo_sdk as scs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "feetech-servo-sdk (scservo_sdk) is required for physical SO-ARM101 deployment. "
                "Activate so101-setup/.venv or pass mock_hardware=True for offline tests."
            ) from exc

        self._port_handler = scs.PortHandler(self.serial_port)
        self._packet_handler = scs.sms_sts(self._port_handler)
        if not self._port_handler.openPort() or not self._port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(
                f"Failed to open SO-ARM101 serial port {self.serial_port} at {self.baudrate} baud."
            )

        try:
            import cv2  # type: ignore[import-not-found]

            for cam_name, dev_idx in self.camera_indices.items():
                cap = cv2.VideoCapture(dev_idx)
                if cap.isOpened():
                    self._captures[cam_name] = cap
        except ImportError:
            pass

    def _read_proprio_6d(self) -> np.ndarray:
        """Read 6 Feetech STS3215 encoders and convert to [5x radians, 1x normalized gripper]."""
        if self.mock_hardware:
            return self._mock_proprio.copy()

        ticks = np.zeros(6, dtype=np.int32)
        for servo_id in range(1, 7):
            pos, _, _ = self._packet_handler.ReadPosSpeed(servo_id)
            ticks[servo_id - 1] = int(pos)

        rads = ticks_to_radians(ticks, self.zero_offsets)
        proprio = np.zeros(6, dtype=np.float32)
        proprio[:5] = rads[:5]
        proprio[5] = raw_gripper_to_norm(float(rads[5]))
        return proprio

    def _write_action_6d(self, action_6d: np.ndarray) -> None:
        """Send 6D joint target [5x radians, 1x normalized gripper] to Feetech servos."""
        if self.mock_hardware:
            self._mock_proprio = np.asarray(action_6d, dtype=np.float32).copy()
            return

        raw_rads = np.asarray(action_6d, dtype=np.float32).copy()
        raw_rads[5] = norm_gripper_to_raw(float(action_6d[5]))
        target_ticks = radians_to_ticks(raw_rads, self.zero_offsets)
        for servo_id in range(1, 7):
            self._packet_handler.WritePosEx(
                servo_id, int(target_ticks[servo_id - 1]), 2200, 45
            )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Move physical arm to HOME_PROPRIO_6D and initialize tracking canvas."""
        del seed
        self._canvas_pil = Image.new("RGB", (self.canvas_w, self.canvas_h), self.BG_COLOR_RGB)
        self._canvas_draw = ImageDraw.Draw(self._canvas_pil)
        self._prev_contact_px = None
        self.qpos_log.clear()
        self.pen_tip_xyz_log.clear()
        self.ink_contact_log.clear()
        self.ink_pixels_drawn = 0
        self.timelapse_frames.clear()

        self._write_action_6d(HOME_PROPRIO_6D)
        if not self.mock_hardware:
            time.sleep(0.8)
        obs = self.get_obs()
        self.timelapse_frames.append(obs["rgb_overhead_cam"].copy())
        return obs

    def step(self, action_6d: np.ndarray) -> dict[str, Any]:
        """Command 6D joint target at control_hz and update forward-kinematics ink mirror."""
        t0 = time.perf_counter()
        act = np.asarray(action_6d, dtype=np.float32).reshape(6)
        self._write_action_6d(act)

        if not self.mock_hardware:
            elapsed = time.perf_counter() - t0
            if elapsed < self.step_dt:
                time.sleep(self.step_dt - elapsed)

        proprio = self._read_proprio_6d()
        tip_xyz = self.ik_solver.forward_kinematics(proprio[:5])
        z_contact_thresh = float(WORKSPACE_SPEC["z_contact_threshold_m"])
        in_contact = bool(
            tip_xyz[2] <= z_contact_thresh and is_xy_in_bounds(tip_xyz[0], tip_xyz[1], margin_m=0.003)
        )

        if in_contact:
            row, col = xy_to_canvas_px(tip_xyz[0], tip_xyz[1], self.canvas_w, self.canvas_h)
            if self._prev_contact_px is not None:
                r0, c0 = self._prev_contact_px
                self._canvas_draw.line([(c0, r0), (col, row)], fill=self.INK_COLOR_RGB, width=4)
                self.ink_pixels_drawn += max(1, int(np.hypot(row - r0, col - c0)))
            else:
                self._canvas_draw.ellipse(
                    [(col - 2, row - 2), (col + 2, row + 2)],
                    fill=self.INK_COLOR_RGB,
                )
                self.ink_pixels_drawn += 1
            self._prev_contact_px = (row, col)
        else:
            self._prev_contact_px = None

        self.qpos_log.append(proprio)
        self.pen_tip_xyz_log.append(tip_xyz.astype(np.float32))
        self.ink_contact_log.append(in_contact)

        return {
            "proprio": proprio,
            "pen_tip_xyz": tip_xyz.astype(np.float32),
            "ink_pixels_drawn": self.ink_pixels_drawn,
        }

    def move_to_home(self) -> None:
        """Lift marker above board first, then return smoothly to HOME_PROPRIO_6D."""
        proprio = self._read_proprio_6d()
        cur_tip = self.ik_solver.forward_kinematics(proprio[:5])
        z_up = float(WORKSPACE_SPEC["z_up_m"])
        if cur_tip[2] < z_up:
            lift_pt = self.apply_plane_calibration(np.array([cur_tip[0], cur_tip[1], z_up]))
            res = self.ik_solver.solve(lift_pt, q_init=proprio[:5])
            self.step(res.action_6d)

        q_start = self._read_proprio_6d()
        for step_idx in range(1, 7):
            alpha = float(step_idx) / 6.0
            self.step((1.0 - alpha) * q_start + alpha * HOME_PROPRIO_6D)

    def execute_cartesian_path(
        self,
        waypoints_xyz: list[list[float]] | np.ndarray,
    ) -> dict[str, Any]:
        """Track metric [X, Y, Z] waypoints on physical SO-ARM101 with 4-corner plane offset."""
        raw_pts = np.asarray(waypoints_xyz, dtype=np.float64)
        if raw_pts.ndim != 2 or raw_pts.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) waypoints array, got shape {raw_pts.shape}")

        x_min, x_max = WORKSPACE_SPEC["x_bounds_m"]
        y_min, y_max = WORKSPACE_SPEC["y_bounds_m"]
        z_floor = float(WORKSPACE_SPEC["z_collision_floor_m"])

        spatial_hallucinations = 0
        warnings: list[str] = []
        oob_xy_count = 0
        table_crash_count = 0
        clamped_pts = raw_pts.copy()

        for idx, pt in enumerate(raw_pts):
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            if not is_xy_in_bounds(x, y):
                oob_xy_count += 1
                clamped_pts[idx, 0] = float(np.clip(x, x_min, x_max))
                clamped_pts[idx, 1] = float(np.clip(y, y_min, y_max))
            if z < z_floor:
                table_crash_count += 1
                clamped_pts[idx, 2] = 0.002

        if oob_xy_count > 0:
            spatial_hallucinations += 1
            warnings.append(f"{oob_xy_count}/{len(raw_pts)} points exceeded XY bounds and were clamped.")
        if table_crash_count > 0:
            spatial_hallucinations += 1
            warnings.append(f"{table_crash_count}/{len(raw_pts)} points commanded Z < 0.0m and were clamped.")

        dense_pts = interpolate_waypoints_3d(clamped_pts, step_size_m=0.004)
        ink_before = self.ink_pixels_drawn
        ik_failures = 0
        max_ik_err_m = 0.0
        q_prev = self._read_proprio_6d()[:5]

        for pt in dense_pts:
            cal_pt = self.apply_plane_calibration(pt)
            res = self.ik_solver.solve(cal_pt, q_init=q_prev)
            q_prev = res.q_5d
            max_ik_err_m = max(max_ik_err_m, res.pos_error_m)
            if not res.success:
                ik_failures += 1
            self.step(res.action_6d)

        ink_added = self.ink_pixels_drawn - ink_before
        if ik_failures > 0:
            spatial_hallucinations += 1
            warnings.append(f"{ik_failures} steps exceeded IK tolerance.")
        if len(raw_pts) > 0 and ink_added == 0:
            spatial_hallucinations += 1
            warnings.append("Air-drawing hallucination: 0 ink pixels recorded.")

        final_tip = self.ik_solver.forward_kinematics(q_prev)
        obs = self.get_obs()
        self.timelapse_frames.append(obs["rgb_overhead_cam"].copy())

        return {
            "waypoints_commanded": len(raw_pts),
            "interpolated_steps": len(dense_pts),
            "ink_pixels_added": int(ink_added),
            "total_ink_pixels": int(self.ink_pixels_drawn),
            "max_ik_error_mm": round(max_ik_err_m * 1000.0, 3),
            "final_pen_tip_xyz": [round(float(v), 4) for v in final_tip],
            "spatial_hallucinations": int(spatial_hallucinations),
            "warnings": warnings,
        }

    def render_canvas_with_grid(self) -> np.ndarray:
        """Return annotated 2D canvas image with metric border."""
        pad_top, pad_bottom, pad_left, pad_right = 36, 36, 68, 28
        out_w = self.canvas_w + pad_left + pad_right
        out_h = self.canvas_h + pad_top + pad_bottom
        annotated = Image.new("RGB", (out_w, out_h), (235, 238, 244))
        annotated.paste(self._canvas_pil, (pad_left, pad_top))
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(
            [(pad_left - 1, pad_top - 1), (pad_left + self.canvas_w, pad_top + self.canvas_h)],
            outline=(70, 75, 85),
            width=2,
        )
        return np.asarray(annotated, dtype=np.uint8)

    def get_obs(self) -> dict[str, Any]:
        """Return observation dict with proprio, canvas_rgb, and USB camera frames."""
        proprio = self._read_proprio_6d()
        tip_xyz = self.ik_solver.forward_kinematics(proprio[:5])
        canvas_np = np.asarray(self._canvas_pil, dtype=np.uint8).copy()
        annotated_np = self.render_canvas_with_grid()

        obs: dict[str, Any] = {
            "proprio": proprio,
            "pen_tip_xyz": tip_xyz.astype(np.float32),
            "ink_pixels_drawn": self.ink_pixels_drawn,
            "canvas_rgb": canvas_np,
            "canvas_annotated_rgb": annotated_np,
        }

        w, h = self.camera_resolution
        for cam_name in CAMERA_NAMES:
            if not self.mock_hardware and cam_name in self._captures:
                import cv2  # type: ignore[import-not-found]

                ok, bgr = self._captures[cam_name].read()
                if ok and bgr is not None:
                    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                    obs[f"rgb_{cam_name}"] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    continue
            obs[f"rgb_{cam_name}"] = np.asarray(
                self._canvas_pil.resize((w, h)), dtype=np.uint8
            )
        return obs

    def get_trajectory_arrays(self) -> dict[str, np.ndarray]:
        """Return logged joint and pen tip trajectory arrays."""
        if not self.qpos_log:
            return {
                "qpos": np.zeros((0, 6), dtype=np.float32),
                "qvel": np.zeros((0, 6), dtype=np.float32),
                "pen_tip_xyz": np.zeros((0, 3), dtype=np.float32),
                "ink_contact": np.zeros((0,), dtype=bool),
            }
        qpos = np.asarray(self.qpos_log, dtype=np.float32)
        return {
            "qpos": qpos,
            "qvel": np.zeros_like(qpos),
            "pen_tip_xyz": np.asarray(self.pen_tip_xyz_log, dtype=np.float32),
            "ink_contact": np.asarray(self.ink_contact_log, dtype=bool),
        }

    def close(self) -> None:
        """Release camera captures and serial port."""
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
