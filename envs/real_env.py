"""Level 2 Physical 6-DoF SO-ARM101 Whiteboard Environment with 4-Corner Plane Calibration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from envs.base import (
    AVAILABLE_MARKER_COLORS,
    CAMERA_NAMES,
    GRIPPER_NORM_GRASP,
    GRIPPER_NORM_OPEN,
    HOME_PROPRIO_6D,
    MARKER_COLORS_RGB,
    MARKER_DOCK_HOVER_Z,
    MARKER_DOCK_SLOTS_XYZ,
    STS3215_CENTER_TICK,
    WORKSPACE_SPEC,
    SO101WhiteboardEnv,
    is_xy_in_bounds,
    norm_gripper_to_raw,
    radians_to_ticks,
    ticks_to_radians,
    xy_to_canvas_px,
)
from envs.ik_solver import SO101IKSolver
from envs.trajectory import interpolate_waypoints_3d


class RealWhiteboardEnv(SO101WhiteboardEnv):
    """Physical SO-ARM101 environment matching SimWhiteboardEnv's exact API and 3-slot marker dock."""

    BG_COLOR_RGB: tuple[int, int, int] = (250, 250, 252)

    MOTOR_NAMES: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    def __init__(
        self,
        serial_port: str = "/dev/tty.usbmodem5B790143431",
        baudrate: int = 1_000_000,
        calibration_path: str | Path | None = None,
        lerobot_calib_path: str | Path | None = None,
        camera_indices: dict[str, int] | None = None,
        control_hz: int = 20,
        camera_resolution: tuple[int, int] = (640, 480),
        zero_offsets: np.ndarray | None = None,
        mock_hardware: bool = False,
    ) -> None:
        """Initialize the physical SO-ARM101 follower arm, plane calibration, and USB webcams."""
        self.serial_port = serial_port
        self.baudrate = int(baudrate)
        self.control_hz = int(control_hz)
        self.step_dt = 1.0 / float(self.control_hz)
        self.camera_resolution = camera_resolution
        self.mock_hardware = bool(mock_hardware)
        self.camera_indices = camera_indices or {"overhead_cam": 0, "third_person_cam": 1}

        self.range_min = np.array([660, 850, 998, 935, 0, 2028], dtype=np.float32)
        self.range_max = np.array([3395, 2852, 3072, 2925, 4095, 3212], dtype=np.float32)
        self._load_lerobot_motor_calibration(lerobot_calib_path)

        if zero_offsets is not None:
            self.zero_offsets = np.asarray(zero_offsets, dtype=np.float32)
        else:
            self.zero_offsets = 0.5 * (self.range_min + self.range_max)
            self.zero_offsets[4] = 1997.0
            self.zero_offsets[5] = float(STS3215_CENTER_TICK)

        if calibration_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            calibration_path = repo_root / "config" / "calibration.json"
        self.calibration_path = Path(calibration_path)
        self.marker_slots_xyz: dict[str, list[float]] = {
            k: list(v) for k, v in MARKER_DOCK_SLOTS_XYZ.items()
        }
        self.single_marker_mode: bool = False
        self.interactive_holder_swap: bool = False
        self.pickup_on_reset: bool = False
        self.pickup_ticks: list[int] = [2495, 1698, 2859, 1909, 973]
        self.target_pitch: float = 1.35
        self.plane_coeffs: tuple[float, float, float] = self._load_calibration()
        self.home_proprio_6d: np.ndarray = (
            HOME_PROPRIO_6D.copy()
            if self.mock_hardware
            else np.array([0.70, -1.05, 1.40, 0.03, -np.pi / 2.0, GRIPPER_NORM_GRASP], dtype=np.float32)
        )

        site_name = "pinch_site" if self.single_marker_mode else "pen_tip"
        self.ik_solver = SO101IKSolver(site_name=site_name, target_pitch=self.target_pitch)

        self.canvas_h = int(WORKSPACE_SPEC["canvas_resolution_px"]["height"])
        self.canvas_w = int(WORKSPACE_SPEC["canvas_resolution_px"]["width"])
        self._canvas_pil = Image.new("RGB", (self.canvas_w, self.canvas_h), self.BG_COLOR_RGB)
        self._canvas_draw = ImageDraw.Draw(self._canvas_pil)
        self._prev_contact_px: tuple[int, int] | None = None

        self.current_marker_color: str = str(WORKSPACE_SPEC["default_marker_color"])
        self._holding_marker: bool = True
        self.colors_used: set[str] = set()
        self.marker_swaps: int = 0

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

    def _load_lerobot_motor_calibration(self, lerobot_calib_path: str | Path | None) -> None:
        """Load per-motor range_min and range_max from LeRobot's so_101_follower.json if present."""
        path = (
            Path(lerobot_calib_path)
            if lerobot_calib_path is not None
            else Path.home()
            / ".cache"
            / "huggingface"
            / "lerobot"
            / "calibration"
            / "robots"
            / "so_follower"
            / "so_101_follower.json"
        )
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for idx, name in enumerate(self.MOTOR_NAMES):
                if name in data:
                    self.range_min[idx] = float(data[name].get("range_min", self.range_min[idx]))
                    self.range_max[idx] = float(data[name].get("range_max", self.range_max[idx]))
        except (OSError, ValueError, TypeError):
            pass

    def _load_calibration(self) -> tuple[float, float, float]:
        """Load 4-corner plane fit and optional marker pickup/dock settings from calibration.json."""
        if self.calibration_path.exists():
            with open(self.calibration_path, encoding="utf-8") as f:
                data = json.load(f)
            if "marker_slots_xyz" in data and isinstance(data["marker_slots_xyz"], dict):
                for color_key, pt in data["marker_slots_xyz"].items():
                    if color_key in self.marker_slots_xyz and len(pt) == 3:
                        self.marker_slots_xyz[color_key] = [float(pt[0]), float(pt[1]), float(pt[2])]
            self.single_marker_mode = bool(data.get("single_marker_mode", False))
            self.interactive_holder_swap = bool(data.get("interactive_holder_swap", False))
            self.pickup_on_reset = bool(data.get("pickup_on_reset", False))
            if "pickup_ticks" in data and isinstance(data["pickup_ticks"], list) and len(data["pickup_ticks"]) == 5:
                self.pickup_ticks = [int(v) for v in data["pickup_ticks"]]
            if (
                "zero_offsets_ticks" in data
                and isinstance(data["zero_offsets_ticks"], list)
                and len(data["zero_offsets_ticks"]) == 6
            ):
                self.zero_offsets = np.array(data["zero_offsets_ticks"], dtype=np.float32)
            if "target_pitch" in data:
                self.target_pitch = float(data["target_pitch"])
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
        z_phys = z + dz
        if self.single_marker_mode:
            # Guard against pressing deeper than 2 mm below the calibrated paper contact plane
            z_phys = max(z_phys, z_calibrated_board - 0.002)
        return np.array([x, y, z_phys], dtype=np.float64)

    def _connect_hardware(self) -> None:
        """Open Feetech STS3215 USB serial bus and USB camera streams."""
        try:
            import scservo_sdk as scs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "feetech-servo-sdk (scservo_sdk) is required for physical SO-ARM101 deployment."
            ) from exc

        self._port_handler = scs.PortHandler(self.serial_port)
        self._packet_handler = scs.PacketHandler(0)
        if not self._port_handler.openPort() or not self._port_handler.setBaudRate(self.baudrate):
            raise RuntimeError(
                f"Failed to open SO-ARM101 serial port {self.serial_port} at {self.baudrate} baud."
            )

        # Configure Motor 6 (gripper) for continuous holding without tripping overload latch
        self._configure_gripper_safe_holding()

        try:
            import cv2  # type: ignore[import-not-found]

            w, h = self.camera_resolution
            for cam_name, dev_idx in self.camera_indices.items():
                cap = cv2.VideoCapture(dev_idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
                    cap.set(cv2.CAP_PROP_FPS, 15.0)
                    for _ in range(3):
                        cap.read()
                    self._captures[cam_name] = cap
        except ImportError:
            pass

    def _configure_gripper_safe_holding(self) -> None:
        """Configure Motor 6 overload thresholds and 45% continuous holding torque limit."""
        if self.mock_hardware or self._port_handler is None or self._packet_handler is None:
            return
        self._packet_handler.write1ByteTxRx(self._port_handler, 6, 55, 0)
        self._packet_handler.write1ByteTxRx(self._port_handler, 6, 34, 75)
        self._packet_handler.write1ByteTxRx(self._port_handler, 6, 36, 95)
        self._packet_handler.write1ByteTxRx(self._port_handler, 6, 55, 1)
        self._packet_handler.write2ByteTxRx(self._port_handler, 6, 48, 450)

    def _compute_safe_gripper_hold_tick(self, target_g: int) -> int:
        """Cap closed-loop stall error on Motor 6 so continuous holding never trips Reg 60 -> 0."""
        if self.mock_hardware or self._port_handler is None or self._packet_handler is None:
            return int(target_g)
        self._packet_handler.write2ByteTxRx(self._port_handler, 6, 48, 450)
        cur_g_raw, _, _ = self._packet_handler.read2ByteTxRx(self._port_handler, 6, 56)
        cur_g = int(cur_g_raw)
        if 0 < cur_g < 2600 and int(target_g) <= 1850:
            hold_goal = max(int(target_g), cur_g - 110)
            cur_load_raw, _, _ = self._packet_handler.read2ByteTxRx(self._port_handler, 6, 60)
            if int(cur_load_raw) == 0:
                self._packet_handler.write1ByteTxRx(self._port_handler, 6, 40, 0)
                self._packet_handler.write2ByteTxRx(self._port_handler, 6, 42, cur_g)
                time.sleep(0.03)
                self._packet_handler.write1ByteTxRx(self._port_handler, 6, 40, 1)
                time.sleep(0.02)
            return hold_goal
        return int(target_g)

    def set_torque(self, enabled: bool, keep_gripper: bool = False) -> None:
        """Enable or disable servo torque on the Feetech STS3215 motors."""
        if self.mock_hardware or self._port_handler is None or self._packet_handler is None:
            return
        val = 1 if enabled else 0
        end_id = 6 if (not enabled and keep_gripper) else 7
        for servo_id in range(1, end_id):
            self._packet_handler.write1ByteTxRx(self._port_handler, servo_id, 40, val)

    def _read_proprio_6d(self) -> np.ndarray:
        """Read 6 Feetech STS3215 encoders and convert to [5x radians, 1x normalized gripper]."""
        if self.mock_hardware:
            return self._mock_proprio.copy()

        ticks = np.zeros(6, dtype=np.int32)
        for servo_id in range(1, 7):
            pos, _, _ = self._packet_handler.read2ByteTxRx(self._port_handler, servo_id, 56)
            ticks[servo_id - 1] = int(pos)

        rads = ticks_to_radians(ticks, self.zero_offsets)
        proprio = np.zeros(6, dtype=np.float32)
        proprio[:5] = rads[:5]
        g_span = max(1.0, float(self.range_max[5] - self.range_min[5]))
        proprio[5] = float(np.clip((ticks[5] - self.range_min[5]) / g_span, 0.0, 1.0))
        return proprio

    def _write_action_6d(self, action_6d: np.ndarray) -> None:
        """Send 6D joint target [5x radians, 1x normalized gripper] to Feetech servos."""
        if self.mock_hardware:
            self._mock_proprio = np.asarray(action_6d, dtype=np.float32).copy()
            return

        raw_rads = np.asarray(action_6d, dtype=np.float32).copy()
        raw_rads[5] = norm_gripper_to_raw(float(action_6d[5]))
        target_ticks = radians_to_ticks(raw_rads, self.zero_offsets).astype(np.float32)
        g_norm = float(np.clip(action_6d[5], 0.0, 1.0))
        target_ticks[5] = self.range_min[5] + g_norm * (self.range_max[5] - self.range_min[5])
        clamped_ticks = np.clip(np.round(target_ticks), self.range_min, self.range_max).astype(np.int32)
        if g_norm <= GRIPPER_NORM_GRASP + 0.05:
            clamped_ticks[5] = self._compute_safe_gripper_hold_tick(1700)

        for servo_id in range(1, 7):
            self._packet_handler.write2ByteTxRx(
                self._port_handler, servo_id, 42, int(clamped_ticks[servo_id - 1])
            )

    def pickup_marker_from_desk(self) -> dict[str, Any]:
        """Clamp the marker resting in the figurine holder at safe continuous holding torque and lift."""
        if self.mock_hardware:
            self._holding_marker = True
            return {"picked_up": True, "pickup_ticks": self.pickup_ticks}

        self.set_torque(True)
        self._configure_gripper_safe_holding()

        p0, p1, p2, p3, p4 = self.pickup_ticks
        open_tick = int(self.range_max[5] - 65)
        grasp_pose = np.array([p0, p1, p2, p3, p4, open_tick], dtype=float)
        lift_up = np.array(
            [p0, max(1100, p1 - 450), max(2500, p2 - 100), min(2300, p3 + 210), p4, 1700],
            dtype=float,
        )

        # 1. Ensure arm is at the figurine holder pose with open jaws if not already there
        cur = np.array(
            [self._packet_handler.read2ByteTxRx(self._port_handler, i + 1, 56)[0] for i in range(6)],
            dtype=float,
        )
        if float(np.max(np.abs(cur[:5] - grasp_pose[:5]))) > 35.0:
            for step_idx in range(1, 21):
                alpha = float(step_idx) / 20.0
                cmd = np.round((1.0 - alpha) * cur + alpha * grasp_pose).astype(int)
                for i in range(6):
                    self._packet_handler.write2ByteTxRx(self._port_handler, i + 1, 42, int(cmd[i]))
                time.sleep(0.04)
            time.sleep(0.2)

        # 2. Clamp Motor 6 smoothly onto barrel, then cap continuous holding goal
        for g_cmd in (2500, 2250, 2000):
            self._packet_handler.write2ByteTxRx(self._port_handler, 6, 42, g_cmd)
            time.sleep(0.28)
        hold_tick = self._compute_safe_gripper_hold_tick(1700)
        self._packet_handler.write2ByteTxRx(self._port_handler, 6, 42, hold_tick)
        time.sleep(0.15)

        # 3. Lift marker smoothly straight up and out of the figurine holder
        grasp_closed = grasp_pose.copy()
        grasp_closed[5] = float(hold_tick)
        lift_up[5] = float(hold_tick)
        for step_idx in range(1, 21):
            alpha = float(step_idx) / 20.0
            cmd = np.round((1.0 - alpha) * grasp_closed + alpha * lift_up).astype(int)
            cmd[5] = hold_tick
            for i in range(6):
                self._packet_handler.write2ByteTxRx(self._port_handler, i + 1, 42, int(cmd[i]))
            time.sleep(0.04)
        time.sleep(0.3)
        self._holding_marker = True
        grip_pos, _, _ = self._packet_handler.read2ByteTxRx(self._port_handler, 6, 56)
        return {"picked_up": True, "gripper_tick": int(grip_pos)}

    def release_marker_at_holder(self) -> dict[str, Any]:
        """Move the arm to the figurine holder pose and open the jaws for marker handoff/swap."""
        if self.mock_hardware:
            self._holding_marker = False
            return {"released": True}

        hold_tick = self._compute_safe_gripper_hold_tick(1700)
        p0, p1, p2, p3, p4 = self.pickup_ticks
        open_tick = int(self.range_max[5] - 65)
        holder_pose = np.array([p0, p1, p2, p3, p4, hold_tick], dtype=float)

        cur = np.array(
            [self._packet_handler.read2ByteTxRx(self._port_handler, i + 1, 56)[0] for i in range(6)],
            dtype=float,
        )
        cur[5] = float(hold_tick)
        for step_idx in range(1, 21):
            alpha = float(step_idx) / 20.0
            cmd = np.round((1.0 - alpha) * cur + alpha * holder_pose).astype(int)
            cmd[5] = hold_tick
            for i in range(6):
                self._packet_handler.write2ByteTxRx(self._port_handler, i + 1, 42, int(cmd[i]))
            time.sleep(0.04)
        time.sleep(0.3)

        # Open gripper jaws and stay parked directly above the figurine hands for the next marker
        self._packet_handler.write2ByteTxRx(self._port_handler, 6, 42, open_tick)
        time.sleep(0.4)
        self._holding_marker = False
        return {"released": True}

    def _make_timelapse_frame(self, obs: dict[str, Any]) -> np.ndarray:
        """Create a side-by-side (third_person_cam | overhead_cam) frame for GIF recording."""
        tp = Image.fromarray(obs["rgb_third_person_cam"]).resize((400, 300))
        ov = Image.fromarray(obs["rgb_overhead_cam"]).resize((400, 300))
        combined = Image.new("RGB", (800, 300))
        combined.paste(tp, (0, 0))
        combined.paste(ov, (400, 0))
        return np.asarray(combined, dtype=np.uint8)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Move physical arm smoothly to unobstructed home pose and initialize tracking canvas."""
        del seed
        self._canvas_pil = Image.new("RGB", (self.canvas_w, self.canvas_h), self.BG_COLOR_RGB)
        self._canvas_draw = ImageDraw.Draw(self._canvas_pil)
        self._prev_contact_px = None
        self.current_marker_color = str(WORKSPACE_SPEC["default_marker_color"])
        self._holding_marker = True
        self.colors_used.clear()
        self.marker_swaps = 0

        self.qpos_log.clear()
        self.pen_tip_xyz_log.clear()
        self.ink_contact_log.clear()
        self.ink_pixels_drawn = 0
        self.timelapse_frames.clear()

        if not self.mock_hardware:
            self.set_torque(True)
            if self.pickup_on_reset:
                self.pickup_marker_from_desk()
            self.move_to_home()
        else:
            self._write_action_6d(HOME_PROPRIO_6D)

        obs = self.get_obs()
        self.timelapse_frames.append(self._make_timelapse_frame(obs))
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
        a, b, c = self.plane_coeffs
        z_board_xy = a * float(tip_xyz[0]) + b * float(tip_xyz[1]) + c
        contact_margin = float(WORKSPACE_SPEC["z_contact_threshold_m"]) - float(
            WORKSPACE_SPEC["z_draw_m"]
        )
        z_contact_thresh = z_board_xy + contact_margin
        active_ink_rgb = MARKER_COLORS_RGB[self.current_marker_color]
        in_contact = bool(
            self._holding_marker
            and tip_xyz[2] <= z_contact_thresh
            and is_xy_in_bounds(tip_xyz[0], tip_xyz[1], margin_m=0.003)
        )

        if in_contact:
            self.colors_used.add(self.current_marker_color)
            row, col = xy_to_canvas_px(tip_xyz[0], tip_xyz[1], self.canvas_w, self.canvas_h)
            if self._prev_contact_px is not None:
                r0, c0 = self._prev_contact_px
                self._canvas_draw.line([(c0, r0), (col, row)], fill=active_ink_rgb, width=4)
                self.ink_pixels_drawn += max(1, int(np.hypot(row - r0, col - c0)))
            else:
                self._canvas_draw.ellipse(
                    [(col - 2, row - 2), (col + 2, row + 2)],
                    fill=active_ink_rgb,
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
            "active_marker_color": self.current_marker_color,
            "colors_used": sorted(self.colors_used),
            "marker_swaps": self.marker_swaps,
        }

    def _move_tip_to_xyz(
        self,
        target_xyz: list[float] | np.ndarray,
        gripper_norm: float,
        step_size_m: float = 0.006,
    ) -> None:
        """Interpolate and track a 3D target point with a specified gripper state on hardware."""
        proprio = self._read_proprio_6d()
        cur_tip = self.ik_solver.forward_kinematics(proprio[:5])
        pts = interpolate_waypoints_3d([cur_tip.tolist(), list(target_xyz)], step_size_m=step_size_m)
        q_prev = proprio[:5]
        for pt in pts:
            cal_pt = self.apply_plane_calibration(pt)
            res = self.ik_solver.solve(cal_pt, q_init=q_prev, gripper_norm=gripper_norm)
            q_prev = res.q_5d
            self.step(res.action_6d)

    def select_marker(self, color: str) -> dict[str, Any]:
        """Move physical arm to the 3-slot marker dock, drop current marker, and pick up requested color."""
        target_color = str(color).strip().lower()
        if target_color not in AVAILABLE_MARKER_COLORS:
            raise ValueError(
                f"Invalid marker color '{color}'. Available colors: {list(AVAILABLE_MARKER_COLORS)}"
            )

        if target_color == self.current_marker_color:
            return {
                "swapped": False,
                "previous_color": self.current_marker_color,
                "active_color": self.current_marker_color,
                "marker_swaps": self.marker_swaps,
                "colors_used": sorted(self.colors_used),
            }

        prev_color = self.current_marker_color
        old_slot = self.marker_slots_xyz[prev_color]
        new_slot = self.marker_slots_xyz[target_color]
        z_hover = float(MARKER_DOCK_HOVER_Z)

        if self.single_marker_mode:
            if not self.mock_hardware and self.interactive_holder_swap:
                if self._holding_marker:
                    self.release_marker_at_holder()
                print(
                    f"\n[MARKER SWAP] Please place the '{target_color.upper()}' marker "
                    "(uncapped, tip down) into the blue figurine holder and press Enter..."
                )
                try:
                    input()
                except EOFError:
                    pass
                self.pickup_marker_from_desk()
                self.move_to_home()
            self.current_marker_color = target_color
            self.marker_swaps += 1
            return {
                "swapped": True,
                "previous_color": prev_color,
                "active_color": self.current_marker_color,
                "dock_slot_xyz_m": new_slot,
                "marker_swaps": self.marker_swaps,
                "colors_used": sorted(self.colors_used),
            }

        # 1. Lift vertically from current position to safe dock hover height
        proprio = self._read_proprio_6d()
        cur_tip = self.ik_solver.forward_kinematics(proprio[:5])
        self._move_tip_to_xyz([float(cur_tip[0]), float(cur_tip[1]), z_hover], GRIPPER_NORM_GRASP)

        # 2. Move above old marker slot, lower into slot, and open gripper to release
        self._move_tip_to_xyz([old_slot[0], old_slot[1], z_hover], GRIPPER_NORM_GRASP)
        self._move_tip_to_xyz(old_slot, GRIPPER_NORM_GRASP, step_size_m=0.004)
        q_cur = self._read_proprio_6d()
        q_cur[5] = GRIPPER_NORM_OPEN
        self._holding_marker = False
        for _ in range(3):
            self.step(q_cur)

        # 3. Lift above old slot, traverse to new marker slot, and lower onto new marker
        self._move_tip_to_xyz([old_slot[0], old_slot[1], z_hover], GRIPPER_NORM_OPEN)
        self._move_tip_to_xyz([new_slot[0], new_slot[1], z_hover], GRIPPER_NORM_OPEN)
        self._move_tip_to_xyz(new_slot, GRIPPER_NORM_OPEN, step_size_m=0.004)

        # 4. Close gripper to grasp new marker and lift to hover height
        q_new = self._read_proprio_6d()
        q_new[5] = GRIPPER_NORM_GRASP
        self.current_marker_color = target_color
        self._holding_marker = True
        for _ in range(3):
            self.step(q_new)

        self._move_tip_to_xyz([new_slot[0], new_slot[1], z_hover], GRIPPER_NORM_GRASP)
        self.marker_swaps += 1

        return {
            "swapped": True,
            "previous_color": prev_color,
            "active_color": self.current_marker_color,
            "dock_slot_xyz_m": new_slot,
            "marker_swaps": self.marker_swaps,
            "colors_used": sorted(self.colors_used),
        }

    def move_to_home(self) -> None:
        """Lift marker above board first, then return smoothly to unobstructed home pose."""
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
            self.step((1.0 - alpha) * q_start + alpha * self.home_proprio_6d)

        obs = self.get_obs()
        self.timelapse_frames.append(self._make_timelapse_frame(obs))

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
        self.timelapse_frames.append(self._make_timelapse_frame(obs))

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
            "active_marker_color": self.current_marker_color,
            "colors_used": sorted(self.colors_used),
            "marker_swaps": self.marker_swaps,
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

    def reload_calibration(self) -> tuple[float, float, float]:
        """Reload plane_coeffs_abc and target_pitch from calibration.json after live calibration."""
        self.plane_coeffs = self._load_calibration()
        self.ik_solver.target_pitch = self.target_pitch
        return self.plane_coeffs

    def suspend_connections(self) -> None:
        """Release USB serial port and cameras without disabling torque so so101_cli can run."""
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
            self._packet_handler = None

    def resume_connections(self) -> None:
        """Re-open USB serial port and cameras after an external so101_cli call completes."""
        if not self.mock_hardware and self._port_handler is None:
            self._connect_hardware()

    def close(self) -> None:
        """Disable servo torque and release camera captures and serial port."""
        self.set_torque(False)
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
