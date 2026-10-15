"""5-DoF Damped Least-Squares Inverse Kinematics Solver for the SO-ARM101 Marker Tip."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from envs.base import (
    ALL_JOINT_NAMES,
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    norm_gripper_to_raw,
)


@dataclass
class IKResult:
    """Result container for SO-ARM101 Inverse Kinematics solve."""

    q_5d: np.ndarray
    action_6d: np.ndarray
    success: bool
    pos_error_m: float
    pitch_error_rad: float
    achieved_xyz: np.ndarray
    iterations: int


class SO101IKSolver:
    """Damped Least-Squares (DLS) IK solver targeting the SO-ARM101 marker pen_tip site."""

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        site_name: str = "pen_tip",
        target_pitch: float = 1.35,
        gripper_norm: float = 0.15,
        pos_tol_m: float = 2.5e-3,
        max_iters: int = 90,
    ) -> None:
        """Initialize the kinematic solver using the SO-ARM101 MuJoCo model."""
        if model is None:
            repo_root = Path(__file__).resolve().parent.parent
            xml_path = repo_root / "assets" / "so101" / "whiteboard_scene.xml"
            model = mujoco.MjModel.from_xml_path(str(xml_path))

        self.model = model
        self.data = mujoco.MjData(model)
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in SO-ARM101 MuJoCo model.")

        self.target_pitch = float(target_pitch)
        self.gripper_norm = float(gripper_norm)
        self.pos_tol_m = float(pos_tol_m)
        self.max_iters = int(max_iters)

        self._joint_ids = [model.joint(name).id for name in ALL_JOINT_NAMES]
        self._qpos_adrs = [int(model.jnt_qposadr[jid]) for jid in self._joint_ids]
        self._dof_adrs = [int(model.jnt_dofadr[jid]) for jid in self._joint_ids]

        self._low_4d = JOINT_LIMITS_LOW[:4].astype(np.float64)
        self._high_4d = JOINT_LIMITS_HIGH[:4].astype(np.float64)
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._j_pitch = np.array([[0.0, 0.15, 0.15, 0.15]], dtype=np.float64)

    def forward_kinematics(self, q_5d: np.ndarray) -> np.ndarray:
        """Compute 3D world position [X, Y, Z] of pen_tip for given 5 arm joint angles."""
        for idx in range(5):
            self.data.qpos[self._qpos_adrs[idx]] = float(q_5d[idx])
        self.data.qpos[self._qpos_adrs[5]] = norm_gripper_to_raw(self.gripper_norm)
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def solve(
        self,
        target_xyz: np.ndarray | list[float],
        target_pitch: float | None = None,
        q_init: np.ndarray | None = None,
    ) -> IKResult:
        """Solve for the 5 arm joint angles that place pen_tip at target_xyz."""
        tgt = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        pitch_goal = self.target_pitch if target_pitch is None else float(target_pitch)

        mujoco.mj_resetData(self.model, self.data)

        if q_init is not None:
            for idx in range(4):
                self.data.qpos[self._qpos_adrs[idx]] = float(q_init[idx])
        else:
            pan_guess = float(-np.arctan2(tgt[1], max(0.05, tgt[0] - 0.0388)))
            init_guess = [pan_guess, -0.25, 0.60, 1.00]
            for idx in range(4):
                self.data.qpos[self._qpos_adrs[idx]] = float(init_guess[idx])

        self.data.qpos[self._qpos_adrs[4]] = -np.pi / 2.0
        self.data.qpos[self._qpos_adrs[5]] = norm_gripper_to_raw(self.gripper_norm)

        dof_idx = self._dof_adrs[:4]
        q_idx = self._qpos_adrs[:4]
        iters = 0

        for step in range(self.max_iters):
            iters = step + 1
            mujoco.mj_forward(self.model, self.data)
            cur_pos = self.data.site_xpos[self.site_id]
            pos_err = tgt - cur_pos
            q_cur = np.array([self.data.qpos[i] for i in q_idx], dtype=np.float64)
            pitch_err = float(pitch_goal - (q_cur[1] + q_cur[2] + q_cur[3]))

            if np.linalg.norm(pos_err) < 2e-4 and abs(pitch_err) < 0.04:
                break

            mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self.site_id)
            j_pos = self._jacp[:, dof_idx]
            j_full = np.vstack([j_pos, self._j_pitch])
            err = np.concatenate([pos_err, [0.15 * pitch_err]])
            dq = j_full.T @ np.linalg.solve(j_full @ j_full.T + 1e-4 * np.eye(4), err)
            dq = np.clip(dq, -0.25, 0.25)
            q_new = np.clip(q_cur + dq, self._low_4d, self._high_4d)
            for i, qi in enumerate(q_idx):
                self.data.qpos[qi] = float(q_new[i])

        mujoco.mj_forward(self.model, self.data)
        achieved = self.data.site_xpos[self.site_id].copy()
        final_pos_err = float(np.linalg.norm(tgt - achieved))
        q_final_4d = np.array([self.data.qpos[i] for i in q_idx], dtype=np.float64)
        final_pitch_err = float(abs(pitch_goal - (q_final_4d[1] + q_final_4d[2] + q_final_4d[3])))

        q_5d = np.array(
            [
                q_final_4d[0],
                q_final_4d[1],
                q_final_4d[2],
                q_final_4d[3],
                -np.pi / 2.0,
            ],
            dtype=np.float32,
        )
        action_6d = np.concatenate([q_5d, [np.float32(self.gripper_norm)]])

        return IKResult(
            q_5d=q_5d,
            action_6d=action_6d,
            success=(final_pos_err <= self.pos_tol_m),
            pos_error_m=final_pos_err,
            pitch_error_rad=final_pitch_err,
            achieved_xyz=achieved,
            iterations=iters,
        )
