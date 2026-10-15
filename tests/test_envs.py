"""Unit tests for SO-ARM101 kinematics, trajectory primitives, SimWhiteboardEnv, and RealWhiteboardEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from envs.base import (
    WORKSPACE_SPEC,
    canvas_px_to_xy,
    norm_gripper_to_raw,
    radians_to_ticks,
    raw_gripper_to_norm,
    ticks_to_radians,
    xy_to_canvas_px,
)
from envs.calibrate_board import fit_plane_from_points
from envs.ik_solver import SO101IKSolver
from envs.real_env import RealWhiteboardEnv
from envs.sim_env import SimWhiteboardEnv
from envs.trajectory import compile_primitives_to_waypoints


def test_coordinate_and_unit_roundtrips() -> None:
    """Verify metric (X, Y) to canvas pixel (row, col) and servo encoder unit conversions."""
    for x in [0.160, 0.230, 0.300]:
        for y in [-0.140, 0.000, 0.140]:
            row, col = xy_to_canvas_px(x, y, width=1000, height=500)
            x_rec, y_rec = canvas_px_to_xy(row, col, width=1000, height=500)
            assert abs(x - x_rec) < 1e-3
            assert abs(y - y_rec) < 1e-3

    for g in [0.0, 0.15, 0.5, 1.0]:
        raw = norm_gripper_to_raw(g)
        assert abs(raw_gripper_to_norm(raw) - g) < 1e-5

    rads = np.array([0.0, -0.5, 0.5, 1.0, -1.5, 0.2], dtype=np.float32)
    ticks = radians_to_ticks(rads)
    rads_rec = ticks_to_radians(ticks)
    assert np.max(np.abs(rads - rads_rec)) < 0.002


def test_ik_solver_across_whiteboard_workspace() -> None:
    """Verify 5-DoF DLS IK converges to sub-millimeter accuracy across the whiteboard envelope."""
    solver = SO101IKSolver()
    x_min, x_max = WORKSPACE_SPEC["x_bounds_m"]
    y_min, y_max = WORKSPACE_SPEC["y_bounds_m"]

    for x in np.linspace(x_min, x_max, 5):
        for y in np.linspace(y_min, y_max, 5):
            for z in [WORKSPACE_SPEC["z_draw_m"], WORKSPACE_SPEC["z_up_m"]]:
                res = solver.solve([float(x), float(y), float(z)])
                assert res.success, f"IK failed at ({x}, {y}, {z}) with err={res.pos_error_m}"
                assert res.pos_error_m < 1e-3


def test_primitive_compiler_and_bounds_checking() -> None:
    """Verify all 6 primitive types compile cleanly and detect out-of-bounds hallucinations."""
    valid_strokes = [
        {"type": "line", "from_xy": [0.18, 0.05], "to_xy": [0.22, -0.05]},
        {"type": "polyline", "points_xy": [[0.20, 0.02], [0.24, 0.00], [0.22, -0.02]], "closed": True},
        {"type": "circle", "center_xy": [0.20, 0.04], "radius": 0.02},
        {"type": "arc", "center_xy": [0.23, 0.0], "radius": 0.02, "start_angle": 0.0, "end_angle": 1.57},
        {"type": "ellipse", "center_xy": [0.24, 0.01], "rx": 0.012, "ry": 0.02, "rotation": 0.1},
        {"type": "bezier", "p0": [0.22, 0.02], "p1": [0.25, 0.01], "p2": [0.26, -0.02], "p3": [0.27, -0.04]},
    ]
    pts, warnings, hall = compile_primitives_to_waypoints(valid_strokes)
    assert len(pts) > 50
    assert hall == 0
    assert len(warnings) == 0

    oob_strokes = [
        {"type": "line", "from_xy": [0.10, 0.0], "to_xy": [0.45, 0.25], "label": "bad_line"},
    ]
    pts_oob, warnings_oob, hall_oob = compile_primitives_to_waypoints(oob_strokes)
    assert len(pts_oob) > 0
    assert hall_oob == 1
    assert len(warnings_oob) == 1


def test_sim_and_real_whiteboard_environments(tmp_path: Path) -> None:
    """Verify 500 Hz contact ink drawing in SimWhiteboardEnv and 4-corner calibration in RealWhiteboardEnv."""
    sim_env = SimWhiteboardEnv(record_timelapse=False)
    try:
        obs0 = sim_env.reset()
        assert obs0["ink_pixels_drawn"] == 0

        pts, _, _ = compile_primitives_to_waypoints(
            [{"type": "line", "from_xy": [0.20, 0.04], "to_xy": [0.25, -0.04]}]
        )
        stats = sim_env.execute_cartesian_path(pts)
        assert stats["ink_pixels_added"] > 100
        assert stats["spatial_hallucinations"] == 0
        traj = sim_env.get_trajectory_arrays()
        assert len(traj["qpos"]) > 0
        assert np.any(traj["ink_contact"])
    finally:
        sim_env.close()

    # Test 4-corner plane fit and RealWhiteboardEnv with mock hardware
    corners = np.array(
        [
            [0.16, 0.14, 0.004],
            [0.16, -0.14, 0.006],
            [0.30, 0.14, 0.005],
            [0.30, -0.14, 0.007],
        ],
        dtype=np.float64,
    )
    a, b, c = fit_plane_from_points(corners)
    cal_file = tmp_path / "calibration.json"
    cal_file.write_text(
        f'{{"plane_coeffs_abc": [{a}, {b}, {c}]}}',
        encoding="utf-8",
    )

    real_env = RealWhiteboardEnv(calibration_path=cal_file, mock_hardware=True)
    try:
        real_env.reset()
        stats_real = real_env.execute_cartesian_path(pts)
        assert stats_real["ink_pixels_added"] > 50
        assert stats_real["spatial_hallucinations"] == 0
    finally:
        real_env.close()
