"""MuJoCo 500 Hz Digital Twin Whiteboard Environment for the 6-DoF SO-ARM101."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from envs.base import (
    ALL_JOINT_NAMES,
    CAMERA_NAMES,
    HOME_PROPRIO_6D,
    WORKSPACE_SPEC,
    SO101WhiteboardEnv,
    is_xy_in_bounds,
    norm_gripper_to_raw,
    raw_gripper_to_norm,
    xy_to_canvas_px,
)
from envs.ik_solver import SO101IKSolver
from envs.trajectory import interpolate_waypoints_3d


class SimWhiteboardEnv(SO101WhiteboardEnv):
    """MuJoCo 500 Hz simulation environment with live contact ink rendering on a whiteboard."""

    INK_COLOR_RGB: tuple[int, int, int] = (18, 42, 138)
    BG_COLOR_RGB: tuple[int, int, int] = (250, 250, 252)

    def __init__(
        self,
        xml_path: str | Path | None = None,
        control_hz: int = 20,
        physics_dt: float = 0.002,
        camera_resolution: tuple[int, int] = (640, 480),
        record_timelapse: bool = True,
    ) -> None:
        """Initialize the SO-ARM101 MuJoCo whiteboard environment."""
        if xml_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            xml_path = repo_root / "assets" / "so101" / "whiteboard_scene.xml"

        self.xml_path = str(xml_path)
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Whiteboard scene XML not found at: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.control_hz = int(control_hz)
        self.physics_dt = float(physics_dt)
        self.model.opt.timestep = self.physics_dt
        self.substeps = max(1, round((1.0 / self.control_hz) / self.physics_dt))
        self.camera_resolution = camera_resolution
        self.record_timelapse = bool(record_timelapse)

        # Joint and actuator indices
        self._joint_ids = [self.model.joint(name).id for name in ALL_JOINT_NAMES]
        self._qpos_adrs = [int(self.model.jnt_qposadr[jid]) for jid in self._joint_ids]
        self._dof_adrs = [int(self.model.jnt_dofadr[jid]) for jid in self._joint_ids]
        self._actuator_ids = [self.model.actuator(name).id for name in ALL_JOINT_NAMES]
        self._pen_tip_site_id = self.model.site("pen_tip").id
        self._whiteboard_tex_id = self.model.texture("whiteboard_tex").id

        # Dedicated IK solver sharing the loaded model
        self.ik_solver = SO101IKSolver(model=self.model, site_name="pen_tip")

        # 2D canvas buffer (height=500, width=1000, 3)
        self.canvas_h = int(WORKSPACE_SPEC["canvas_resolution_px"]["height"])
        self.canvas_w = int(WORKSPACE_SPEC["canvas_resolution_px"]["width"])
        self._canvas_pil = Image.new("RGB", (self.canvas_w, self.canvas_h), self.BG_COLOR_RGB)
        self._canvas_draw = ImageDraw.Draw(self._canvas_pil)
        self._texture_dirty = True
        self._prev_contact_px: tuple[int, int] | None = None

        # Offscreen renderer (lazy-initialized)
        self._renderer: mujoco.Renderer | None = None

        # Episode telemetry and 500 Hz logs
        self.qpos_500hz: list[np.ndarray] = []
        self.qvel_500hz: list[np.ndarray] = []
        self.pen_tip_xyz_500hz: list[np.ndarray] = []
        self.ink_contact_500hz: list[bool] = []
        self.ink_pixels_drawn: int = 0
        self.timelapse_frames: list[np.ndarray] = []
        self._substep_counter: int = 0

        self.reset()

    def _ensure_renderer(self) -> mujoco.Renderer:
        """Create or return the MuJoCo offscreen renderer."""
        if self._renderer is None:
            w, h = self.camera_resolution
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._texture_dirty = True
        return self._renderer

    def _sync_whiteboard_texture(self) -> None:
        """Copy the 2D PIL canvas into MuJoCo's whiteboard_tex buffer and upload to OpenGL."""
        canvas_np = np.asarray(self._canvas_pil, dtype=np.uint8)
        tid = self._whiteboard_tex_id
        adr = int(self.model.tex_adr[tid])
        h_tex = int(self.model.tex_height[tid])
        w_tex = int(self.model.tex_width[tid])
        tex_view = self.model.tex_data[adr : adr + h_tex * w_tex * 3].reshape(h_tex, w_tex, 3)
        tex_view[:] = np.transpose(canvas_np[::-1, :, :], (1, 0, 2))
        if self._renderer is not None:
            mujoco.mjr_uploadTexture(self.model, self._renderer._mjr_context, tid)
        self._texture_dirty = False

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset simulation state to HOME_PROPRIO_6D and clear whiteboard canvas for a new episode."""
        del seed
        mujoco.mj_resetData(self.model, self.data)

        for idx in range(5):
            q_val = float(HOME_PROPRIO_6D[idx])
            self.data.qpos[self._qpos_adrs[idx]] = q_val
            self.data.ctrl[self._actuator_ids[idx]] = q_val

        g_raw = norm_gripper_to_raw(float(HOME_PROPRIO_6D[5]))
        self.data.qpos[self._qpos_adrs[5]] = g_raw
        self.data.ctrl[self._actuator_ids[5]] = g_raw
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Reset permanent additive canvas for the new episode
        self._canvas_pil = Image.new("RGB", (self.canvas_w, self.canvas_h), self.BG_COLOR_RGB)
        self._canvas_draw = ImageDraw.Draw(self._canvas_pil)
        self._prev_contact_px = None
        self._texture_dirty = True
        self._sync_whiteboard_texture()

        self.qpos_500hz.clear()
        self.qvel_500hz.clear()
        self.pen_tip_xyz_500hz.clear()
        self.ink_contact_500hz.clear()
        self.ink_pixels_drawn = 0
        self.timelapse_frames.clear()
        self._substep_counter = 0

        obs = self.get_obs()
        if self.record_timelapse:
            self.timelapse_frames.append(self._make_timelapse_frame(obs))
        return obs

    def _make_timelapse_frame(self, obs: dict[str, Any]) -> np.ndarray:
        """Create a side-by-side (third_person_cam | overhead_cam) frame for GIF recording."""
        tp = Image.fromarray(obs["rgb_third_person_cam"]).resize((400, 300))
        ov = Image.fromarray(obs["rgb_overhead_cam"]).resize((400, 300))
        combined = Image.new("RGB", (800, 300))
        combined.paste(tp, (0, 0))
        combined.paste(ov, (400, 0))
        return np.asarray(combined, dtype=np.uint8)

    def step(self, action_6d: np.ndarray) -> dict[str, Any]:
        """Step the simulation by one control interval (25 sub-steps at 500 Hz)."""
        act = np.asarray(action_6d, dtype=np.float32).reshape(6)
        for idx in range(5):
            self.data.ctrl[self._actuator_ids[idx]] = float(act[idx])
        self.data.ctrl[self._actuator_ids[5]] = norm_gripper_to_raw(float(act[5]))

        z_contact_thresh = float(WORKSPACE_SPEC["z_contact_threshold_m"])

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            self._substep_counter += 1

            tip_xyz = self.data.site_xpos[self._pen_tip_site_id].copy()
            in_contact = bool(
                tip_xyz[2] <= z_contact_thresh and is_xy_in_bounds(tip_xyz[0], tip_xyz[1], margin_m=0.002)
            )

            if in_contact:
                row, col = xy_to_canvas_px(tip_xyz[0], tip_xyz[1], self.canvas_w, self.canvas_h)
                if self._prev_contact_px is not None:
                    r0, c0 = self._prev_contact_px
                    if abs(row - r0) + abs(col - c0) <= 60:
                        self._canvas_draw.line([(c0, r0), (col, row)], fill=self.INK_COLOR_RGB, width=5)
                        self.ink_pixels_drawn += max(1, int(np.hypot(row - r0, col - c0)))
                else:
                    self._canvas_draw.ellipse(
                        [(col - 2, row - 2), (col + 2, row + 2)],
                        fill=self.INK_COLOR_RGB,
                    )
                    self.ink_pixels_drawn += 1
                self._prev_contact_px = (row, col)
                self._texture_dirty = True
            else:
                self._prev_contact_px = None

            qpos_6d = np.array(
                [self.data.qpos[adr] for adr in self._qpos_adrs],
                dtype=np.float32,
            )
            qvel_6d = np.array(
                [self.data.qvel[adr] for adr in self._dof_adrs],
                dtype=np.float32,
            )
            self.qpos_500hz.append(qpos_6d)
            self.qvel_500hz.append(qvel_6d)
            self.pen_tip_xyz_500hz.append(tip_xyz.astype(np.float32))
            self.ink_contact_500hz.append(in_contact)

        return self._get_fast_state()

    def move_to_home(self) -> None:
        """Lift marker vertically to safe height first, then retract smoothly to HOME_PROPRIO_6D."""
        cur_tip = self.data.site_xpos[self._pen_tip_site_id].copy()
        z_up = float(WORKSPACE_SPEC["z_up_m"])
        if cur_tip[2] < z_up:
            lift_target = np.array([cur_tip[0], cur_tip[1], z_up], dtype=np.float64)
            q_cur = np.array([self.data.qpos[adr] for adr in self._qpos_adrs[:5]], dtype=np.float32)
            ik_res = self.ik_solver.solve(lift_target, q_init=q_cur)
            for _ in range(4):
                self.step(ik_res.action_6d)

        q_start = self._read_proprio_6d()
        for step_idx in range(1, 9):
            alpha = float(step_idx) / 8.0
            interp_act = (1.0 - alpha) * q_start + alpha * HOME_PROPRIO_6D
            self.step(interp_act)

        if self.record_timelapse:
            obs = self.get_obs()
            self.timelapse_frames.append(self._make_timelapse_frame(obs))

    def execute_cartesian_path(
        self,
        waypoints_xyz: list[list[float]] | np.ndarray,
    ) -> dict[str, Any]:
        """Track a sequence of metric [X, Y, Z] waypoints using DLS IK at 500 Hz."""
        raw_pts = np.asarray(waypoints_xyz, dtype=np.float64)
        if raw_pts.ndim != 2 or raw_pts.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) waypoints array, got shape {raw_pts.shape}")

        x_min, x_max = WORKSPACE_SPEC["x_bounds_m"]
        y_min, y_max = WORKSPACE_SPEC["y_bounds_m"]
        z_floor = float(WORKSPACE_SPEC["z_collision_floor_m"])

        spatial_hallucinations = 0
        warnings: list[str] = []

        # Audit raw waypoints for out-of-bounds or table collision before clamping
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
            warnings.append(
                f"{oob_xy_count}/{len(raw_pts)} waypoints exceeded whiteboard XY bounds "
                f"X in [{x_min}, {x_max}], Y in [{y_min}, {y_max}] and were clamped."
            )
        if table_crash_count > 0:
            spatial_hallucinations += 1
            warnings.append(
                f"{table_crash_count}/{len(raw_pts)} waypoints commanded Z < {z_floor}m "
                "(table collision) and were clamped to Z = 0.002m."
            )

        dense_pts = interpolate_waypoints_3d(clamped_pts, step_size_m=0.0025)
        ink_before = self.ink_pixels_drawn
        ik_failures = 0
        max_ik_err_m = 0.0

        q_prev = np.array([self.data.qpos[adr] for adr in self._qpos_adrs[:5]], dtype=np.float32)

        # Transition smoothly from current pen_tip position to first waypoint at safe Z
        cur_tip = self.data.site_xpos[self._pen_tip_site_id].copy()
        if len(dense_pts) > 0 and float(np.linalg.norm(dense_pts[0] - cur_tip)) > 0.006:
            z_safe = max(float(WORKSPACE_SPEC["z_up_m"]), float(cur_tip[2]), float(dense_pts[0][2]))
            bridge = interpolate_waypoints_3d(
                [
                    cur_tip.tolist(),
                    [float(cur_tip[0]), float(cur_tip[1]), z_safe],
                    [float(dense_pts[0][0]), float(dense_pts[0][1]), z_safe],
                    dense_pts[0].tolist(),
                ],
                step_size_m=0.005,
            )
            for b_pt in bridge:
                res = self.ik_solver.solve(b_pt, q_init=q_prev)
                q_prev = res.q_5d
                self.step(res.action_6d)

        frame_stride = max(1, len(dense_pts) // 12)
        for idx, pt in enumerate(dense_pts):
            res = self.ik_solver.solve(pt, q_init=q_prev)
            q_prev = res.q_5d
            max_ik_err_m = max(max_ik_err_m, res.pos_error_m)
            if not res.success:
                ik_failures += 1
            self.step(res.action_6d)

            if self.record_timelapse and (idx % frame_stride == 0 or idx == len(dense_pts) - 1):
                obs = self.get_obs()
                self.timelapse_frames.append(self._make_timelapse_frame(obs))

        ink_added = self.ink_pixels_drawn - ink_before
        if ik_failures > 0:
            spatial_hallucinations += 1
            warnings.append(
                f"{ik_failures}/{len(dense_pts)} interpolated steps exceeded IK tolerance "
                f"(max error {max_ik_err_m * 1000.0:.2f} mm)."
            )

        if len(raw_pts) > 0 and ink_added == 0:
            spatial_hallucinations += 1
            warnings.append(
                "Air-drawing hallucination: trajectory executed without ever lowering pen_tip "
                f"to z_draw <= {WORKSPACE_SPEC['z_contact_threshold_m']}m (0 pixels of ink drawn)."
            )

        final_tip = self.data.site_xpos[self._pen_tip_site_id].copy()
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
        """Return an annotated 2D canvas image with metric X and Y coordinate axes along the border."""
        pad_top, pad_bottom, pad_left, pad_right = 36, 36, 68, 28
        out_w = self.canvas_w + pad_left + pad_right
        out_h = self.canvas_h + pad_top + pad_bottom

        annotated = Image.new("RGB", (out_w, out_h), (235, 238, 244))
        annotated.paste(self._canvas_pil, (pad_left, pad_top))
        draw = ImageDraw.Draw(annotated)

        # Draw subtle coordinate grid and border
        draw.rectangle(
            [(pad_left - 1, pad_top - 1), (pad_left + self.canvas_w, pad_top + self.canvas_h)],
            outline=(70, 75, 85),
            width=2,
        )

        # X axis ticks (vertical: 0.16m bottom to 0.30m top)
        for x_val in [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
            row, _ = xy_to_canvas_px(x_val, 0.0, self.canvas_w, self.canvas_h)
            py = pad_top + row
            draw.line([(pad_left - 6, py), (pad_left, py)], fill=(50, 55, 65), width=2)
            draw.text((8, py - 6), f"X={x_val:.2f}", fill=(30, 35, 45))

        # Y axis ticks (horizontal: +0.14m left to -0.14m right)
        for y_val in [0.14, 0.10, 0.05, 0.00, -0.05, -0.10, -0.14]:
            _, col = xy_to_canvas_px(0.23, y_val, self.canvas_w, self.canvas_h)
            px = pad_left + col
            draw.line(
                [(px, pad_top + self.canvas_h), (px, pad_top + self.canvas_h + 6)],
                fill=(50, 55, 65),
                width=2,
            )
            draw.text((px - 22, pad_top + self.canvas_h + 10), f"Y={y_val:+.2f}", fill=(30, 35, 45))

        draw.text(
            (pad_left + 12, 10),
            "SO-ARM101 Metric Whiteboard Canvas (+X is UP, +Y is LEFT, units: meters)",
            fill=(25, 30, 40),
        )
        return np.asarray(annotated, dtype=np.uint8)

    def _read_proprio_6d(self) -> np.ndarray:
        """Return current 6D joint state [5x radians, 1x normalized gripper]."""
        proprio = np.zeros(6, dtype=np.float32)
        for idx in range(5):
            proprio[idx] = float(self.data.qpos[self._qpos_adrs[idx]])
        proprio[5] = raw_gripper_to_norm(float(self.data.qpos[self._qpos_adrs[5]]))
        return proprio

    def _get_fast_state(self) -> dict[str, Any]:
        """Return lightweight non-rendered state for inner loop stepping."""
        return {
            "proprio": self._read_proprio_6d(),
            "pen_tip_xyz": self.data.site_xpos[self._pen_tip_site_id].copy().astype(np.float32),
            "ink_pixels_drawn": self.ink_pixels_drawn,
        }

    def get_obs(self) -> dict[str, Any]:
        """Render cameras and return complete observation dictionary."""
        renderer = self._ensure_renderer()
        if self._texture_dirty:
            self._sync_whiteboard_texture()

        obs = self._get_fast_state()
        obs["canvas_rgb"] = np.asarray(self._canvas_pil, dtype=np.uint8).copy()
        obs["canvas_annotated_rgb"] = self.render_canvas_with_grid()

        for cam_name in CAMERA_NAMES:
            renderer.update_scene(self.data, camera=cam_name)
            obs[f"rgb_{cam_name}"] = renderer.render().copy()

        return obs

    def get_trajectory_arrays(self) -> dict[str, np.ndarray]:
        """Return recorded 500 Hz numpy arrays for dataset archival."""
        if not self.qpos_500hz:
            return {
                "qpos": np.zeros((0, 6), dtype=np.float32),
                "qvel": np.zeros((0, 6), dtype=np.float32),
                "pen_tip_xyz": np.zeros((0, 3), dtype=np.float32),
                "ink_contact": np.zeros((0,), dtype=bool),
            }
        return {
            "qpos": np.asarray(self.qpos_500hz, dtype=np.float32),
            "qvel": np.asarray(self.qvel_500hz, dtype=np.float32),
            "pen_tip_xyz": np.asarray(self.pen_tip_xyz_500hz, dtype=np.float32),
            "ink_contact": np.asarray(self.ink_contact_500hz, dtype=bool),
        }

    def close(self) -> None:
        """Close the MuJoCo offscreen renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
