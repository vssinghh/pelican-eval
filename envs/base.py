"""Shared base interface, workspace geometry constants, and unit conversions for PelicanBench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_JOINT_NAMES: tuple[str, ...] = (*ARM_JOINT_NAMES, "gripper")

CAMERA_NAMES: tuple[str, ...] = (
    "overhead_cam",
    "third_person_cam",
    "wrist_cam",
)

# Neutral home pose (5 arm joints in radians + 1 normalized gripper in [0.0, 1.0])
# Keeps the arm retracted and swung to the side so the entire whiteboard (X in [0.16, 0.30])
# is 100% unobstructed in overhead_cam when inspect_canvas() is called.
HOME_PROPRIO_6D = np.array(
    [-1.35, -1.45, 1.30, 0.65, -np.pi / 2.0, 0.15],
    dtype=np.float32,
)

# Physical SO-ARM101 gripper hinge range mapped to normalized [0.0 (closed), 1.0 (open)]
GRIPPER_RAW_CLOSED: float = -0.15
GRIPPER_RAW_OPEN: float = 0.85

# Physical 6-DoF SO-ARM101 joint limits (5 arm joints in radians + normalized gripper in [0.0, 1.0])
JOINT_LIMITS_LOW = np.array(
    [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, 0.0],
    dtype=np.float32,
)
JOINT_LIMITS_HIGH = np.array(
    [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.0],
    dtype=np.float32,
)

# Feetech STS3215 12-bit magnetic encoder resolution (4096 ticks per 2*pi radians)
STS3215_TICKS_PER_REV: int = 4096
STS3215_CENTER_TICK: int = 2048

# Locked Metric Whiteboard Workspace Specification (in meters relative to SO-ARM101 base [0, 0, 0])
WORKSPACE_SPEC: dict[str, Any] = {
    "robot": "6-DoF SO-ARM101 (5 arm joints + marker held in gripper)",
    "coordinate_frame": (
        "Right-handed 3D metric coordinates [X, Y, Z] in meters relative to robot base [0, 0, 0]. "
        "On the 2D canvas view: +X points UP (top of drawing, away from robot base), "
        "-X points DOWN (bottom of drawing, closer to robot base), "
        "+Y points LEFT, and -Y points RIGHT."
    ),
    "x_bounds_m": [0.160, 0.300],
    "y_bounds_m": [-0.140, 0.140],
    "center_xy_m": [0.230, 0.000],
    "z_draw_m": 0.005,
    "z_contact_threshold_m": 0.008,
    "z_up_m": 0.030,
    "z_collision_floor_m": 0.000,
    "canvas_resolution_px": {"width": 1000, "height": 500},
    "ink_policy": "Additive and permanent. Once ink is drawn on the board, it cannot be erased.",
}


def xy_to_canvas_px(
    x: float,
    y: float,
    width: int = 1000,
    height: int = 500,
) -> tuple[int, int]:
    """Map metric world (X, Y) on the whiteboard to 2D canvas pixel (row, col).

    Canvas orientation (matching overhead_cam view):
    - Top edge (row = 0) corresponds to X_max = 0.300 m.
    - Bottom edge (row = height - 1) corresponds to X_min = 0.160 m.
    - Left edge (col = 0) corresponds to Y_max = +0.140 m.
    - Right edge (col = width - 1) corresponds to Y_min = -0.140 m.
    """
    x_min, x_max = WORKSPACE_SPEC["x_bounds_m"]
    y_min, y_max = WORKSPACE_SPEC["y_bounds_m"]

    u_norm = (y_max - float(y)) / (y_max - y_min)
    v_norm = (x_max - float(x)) / (x_max - x_min)

    col = int(np.clip(round(u_norm * (width - 1)), 0, width - 1))
    row = int(np.clip(round(v_norm * (height - 1)), 0, height - 1))
    return row, col


def canvas_px_to_xy(
    row: int,
    col: int,
    width: int = 1000,
    height: int = 500,
) -> tuple[float, float]:
    """Map 2D canvas pixel (row, col) back to metric world (X, Y) in meters."""
    x_min, x_max = WORKSPACE_SPEC["x_bounds_m"]
    y_min, y_max = WORKSPACE_SPEC["y_bounds_m"]

    u_norm = float(col) / float(max(1, width - 1))
    v_norm = float(row) / float(max(1, height - 1))

    y = float(y_max - u_norm * (y_max - y_min))
    x = float(x_max - v_norm * (x_max - x_min))
    return x, y


def is_xy_in_bounds(x: float, y: float, margin_m: float = 1e-4) -> bool:
    """Return True if metric (x, y) lies within the whiteboard bounds."""
    x_min, x_max = WORKSPACE_SPEC["x_bounds_m"]
    y_min, y_max = WORKSPACE_SPEC["y_bounds_m"]
    return (x_min - margin_m <= float(x) <= x_max + margin_m) and (
        y_min - margin_m <= float(y) <= y_max + margin_m
    )


def raw_gripper_to_norm(raw_rad: float) -> float:
    """Convert raw MuJoCo/servo gripper angle (radians) to normalized [0.0, 1.0]."""
    norm = (float(raw_rad) - GRIPPER_RAW_CLOSED) / (GRIPPER_RAW_OPEN - GRIPPER_RAW_CLOSED)
    return float(np.clip(norm, 0.0, 1.0))


def norm_gripper_to_raw(norm_val: float) -> float:
    """Convert normalized gripper command in [0.0, 1.0] to raw angle (radians)."""
    clipped = float(np.clip(norm_val, 0.0, 1.0))
    return float(GRIPPER_RAW_CLOSED + clipped * (GRIPPER_RAW_OPEN - GRIPPER_RAW_CLOSED))


def ticks_to_radians(ticks: np.ndarray, zero_offsets: np.ndarray | None = None) -> np.ndarray:
    """Convert Feetech STS3215 encoder ticks (0..4095) to radians relative to home offsets."""
    arr = np.asarray(ticks, dtype=np.float32)
    offsets = (
        np.asarray(zero_offsets, dtype=np.float32)
        if zero_offsets is not None
        else np.full_like(arr, float(STS3215_CENTER_TICK))
    )
    return (arr - offsets) * (2.0 * np.pi / float(STS3215_TICKS_PER_REV))


def radians_to_ticks(radians: np.ndarray, zero_offsets: np.ndarray | None = None) -> np.ndarray:
    """Convert radians relative to home offsets into Feetech STS3215 encoder ticks (0..4095)."""
    arr = np.asarray(radians, dtype=np.float32)
    offsets = (
        np.asarray(zero_offsets, dtype=np.float32)
        if zero_offsets is not None
        else np.full_like(arr, float(STS3215_CENTER_TICK))
    )
    ticks = np.round(offsets + arr * (float(STS3215_TICKS_PER_REV) / (2.0 * np.pi)))
    return np.clip(ticks, 0, STS3215_TICKS_PER_REV - 1).astype(np.int32)


class SO101WhiteboardEnv(ABC):
    """Shared environment interface implemented by SimWhiteboardEnv and RealWhiteboardEnv."""

    control_hz: int = 20

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset arm to home pose and initialize a fresh blank whiteboard canvas."""

    @abstractmethod
    def step(self, action_6d: np.ndarray) -> dict[str, Any]:
        """Execute one control step (25 physics sub-steps at 500 Hz in Sim) with a 6D joint target."""

    @abstractmethod
    def move_to_home(self) -> None:
        """Retract arm smoothly to HOME_PROPRIO_6D so overhead camera has a clear view."""

    @abstractmethod
    def execute_cartesian_path(
        self,
        waypoints_xyz: list[list[float]] | np.ndarray,
    ) -> dict[str, Any]:
        """Track a sequence of metric [X, Y, Z] waypoints using IK and record telemetry."""

    @abstractmethod
    def render_canvas_with_grid(self) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB image of the 2D whiteboard with metric ruler ticks."""

    @abstractmethod
    def get_obs(self) -> dict[str, Any]:
        """Return current observation dictionary with proprio, pen_tip_xyz, canvas, and camera frames."""

    @abstractmethod
    def close(self) -> None:
        """Release simulation renderer or hardware handles."""
