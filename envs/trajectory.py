"""Geometric primitive discretization and 3D waypoint interpolation for PelicanBench."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from envs.base import WORKSPACE_SPEC, is_xy_in_bounds


def interpolate_segment_3d(
    p0: np.ndarray,
    p1: np.ndarray,
    step_size_m: float = 0.0025,
) -> list[np.ndarray]:
    """Linearly sample 3D points from p0 to p1 at step_size_m spacing."""
    p0_arr = np.asarray(p0, dtype=np.float64).reshape(3)
    p1_arr = np.asarray(p1, dtype=np.float64).reshape(3)
    dist = float(np.linalg.norm(p1_arr - p0_arr))
    if dist < 1e-6:
        return [p1_arr]
    n_steps = max(1, math.ceil(dist / step_size_m))
    return [
        p0_arr + (float(i) / float(n_steps)) * (p1_arr - p0_arr)
        for i in range(1, n_steps + 1)
    ]


def interpolate_waypoints_3d(
    waypoints_xyz: list[list[float]] | np.ndarray,
    step_size_m: float = 0.0025,
) -> np.ndarray:
    """Convert a sequence of 3D [X, Y, Z] waypoints into a dense path for smooth tracking."""
    pts = np.asarray(waypoints_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected waypoints of shape (N, 3), got {pts.shape}")
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    dense: list[np.ndarray] = [pts[0].copy()]
    for idx in range(len(pts) - 1):
        seg = interpolate_segment_3d(pts[idx], pts[idx + 1], step_size_m=step_size_m)
        dense.extend(seg)
    return np.asarray(dense, dtype=np.float64)


def _sample_single_primitive_xy(stroke: dict[str, Any], ds: float = 0.0025) -> list[tuple[float, float]]:
    """Discretize one 2D primitive dictionary into a list of metric (x, y) coordinates."""
    stype = str(stroke.get("type", "")).strip().lower()

    if stype == "line":
        p0 = stroke.get("from_xy") or stroke.get("start_xy") or stroke.get("p0")
        p1 = stroke.get("to_xy") or stroke.get("end_xy") or stroke.get("p1")
        if p0 is None or p1 is None or len(p0) != 2 or len(p1) != 2:
            raise ValueError("Line primitive requires 'from_xy': [x1, y1] and 'to_xy': [x2, y2].")
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, math.ceil(dist / ds))
        return [(x0 + (i / n) * (x1 - x0), y0 + (i / n) * (y1 - y0)) for i in range(n + 1)]

    if stype == "polyline":
        raw_pts = stroke.get("points_xy") or stroke.get("points")
        if not raw_pts or len(raw_pts) < 2:
            raise ValueError("Polyline primitive requires 'points_xy' with at least 2 [x, y] points.")
        pts = [(float(p[0]), float(p[1])) for p in raw_pts]
        if bool(stroke.get("closed", False)):
            pts.append(pts[0])
        sampled: list[tuple[float, float]] = [pts[0]]
        for idx in range(len(pts) - 1):
            x0, y0 = pts[idx]
            x1, y1 = pts[idx + 1]
            dist = math.hypot(x1 - x0, y1 - y0)
            n = max(1, math.ceil(dist / ds))
            for i in range(1, n + 1):
                sampled.append((x0 + (i / n) * (x1 - x0), y0 + (i / n) * (y1 - y0)))
        return sampled

    if stype == "circle":
        center = stroke.get("center_xy") or stroke.get("center")
        radius = stroke.get("radius")
        if center is None or radius is None or len(center) != 2:
            raise ValueError("Circle primitive requires 'center_xy': [cx, cy] and 'radius': r.")
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        if r <= 0.0:
            raise ValueError(f"Circle radius must be positive, got {r}.")
        circumference = 2.0 * math.pi * r
        n = max(16, math.ceil(circumference / ds))
        return [
            (
                cx + r * math.cos(2.0 * math.pi * i / n),
                cy + r * math.sin(2.0 * math.pi * i / n),
            )
            for i in range(n + 1)
        ]

    if stype == "arc":
        center = stroke.get("center_xy") or stroke.get("center")
        radius = stroke.get("radius")
        a0 = stroke.get("start_angle", 0.0)
        a1 = stroke.get("end_angle", math.pi)
        if center is None or radius is None or len(center) != 2:
            raise ValueError(
                "Arc primitive requires 'center_xy': [cx, cy], 'radius': r, 'start_angle', 'end_angle'."
            )
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        if r <= 0.0:
            raise ValueError(f"Arc radius must be positive, got {r}.")
        a0_f, a1_f = float(a0), float(a1)
        arc_len = abs(a1_f - a0_f) * r
        n = max(8, math.ceil(arc_len / ds))
        return [
            (
                cx + r * math.cos(a0_f + (i / n) * (a1_f - a0_f)),
                cy + r * math.sin(a0_f + (i / n) * (a1_f - a0_f)),
            )
            for i in range(n + 1)
        ]

    if stype in ("ellipse",):
        center = stroke.get("center_xy") or stroke.get("center")
        rx = stroke.get("rx") or stroke.get("radius_x")
        ry = stroke.get("ry") or stroke.get("radius_y")
        rot = float(stroke.get("rotation", 0.0))
        a0_f = float(stroke.get("start_angle", 0.0))
        a1_f = float(stroke.get("end_angle", 2.0 * math.pi))
        if center is None or rx is None or ry is None or len(center) != 2:
            raise ValueError("Ellipse primitive requires 'center_xy': [cx, cy], 'rx', and 'ry'.")
        cx, cy = float(center[0]), float(center[1])
        rx_f, ry_f = float(rx), float(ry)
        if rx_f <= 0.0 or ry_f <= 0.0:
            raise ValueError("Ellipse radii rx and ry must be positive.")
        approx_len = abs(a1_f - a0_f) * max(rx_f, ry_f)
        n = max(16, math.ceil(approx_len / ds))
        c_rot, s_rot = math.cos(rot), math.sin(rot)
        pts_ell: list[tuple[float, float]] = []
        for i in range(n + 1):
            theta = a0_f + (i / n) * (a1_f - a0_f)
            lx = rx_f * math.cos(theta)
            ly = ry_f * math.sin(theta)
            wx = cx + lx * c_rot - ly * s_rot
            wy = cy + lx * s_rot + ly * c_rot
            pts_ell.append((wx, wy))
        return pts_ell

    if stype in ("bezier", "cubic_bezier"):
        p0 = stroke.get("p0")
        p1 = stroke.get("p1")
        p2 = stroke.get("p2")
        p3 = stroke.get("p3")
        if any(p is None or len(p) != 2 for p in (p0, p1, p2, p3)):
            raise ValueError("Bezier primitive requires control points 'p0', 'p1', 'p2', 'p3' as [x, y].")
        pts_ctrl = [np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3)]
        chord_len = (
            np.linalg.norm(pts_ctrl[1] - pts_ctrl[0])
            + np.linalg.norm(pts_ctrl[2] - pts_ctrl[1])
            + np.linalg.norm(pts_ctrl[3] - pts_ctrl[2])
        )
        n = max(12, math.ceil(float(chord_len) / ds))
        curve: list[tuple[float, float]] = []
        for i in range(n + 1):
            t = float(i) / float(n)
            mt = 1.0 - t
            pt = (
                (mt**3) * pts_ctrl[0]
                + 3.0 * (mt**2) * t * pts_ctrl[1]
                + 3.0 * mt * (t**2) * pts_ctrl[2]
                + (t**3) * pts_ctrl[3]
            )
            curve.append((float(pt[0]), float(pt[1])))
        return curve

    raise ValueError(
        f"Unknown primitive type '{stype}'. Supported types: 'line', 'polyline', 'circle', 'arc', 'ellipse', 'bezier'."
    )


def compile_primitives_to_waypoints(
    strokes: list[dict[str, Any]],
    step_size_m: float = 0.0025,
) -> tuple[np.ndarray, list[str], int]:
    """Compile a list of 2D primitive dictionaries into a 3D [X, Y, Z] trajectory.

    Returns:
        Tuple of:
        - waypoints_xyz: (N, 3) float64 array of 3D target poses (including z_up travel moves).
        - warnings: list of human-readable spatial or schema warnings.
        - spatial_hallucination_count: number of strokes that contained out-of-bounds coordinates.
    """
    z_draw = float(WORKSPACE_SPEC["z_draw_m"])
    z_up = float(WORKSPACE_SPEC["z_up_m"])
    x_min, x_max = WORKSPACE_SPEC["x_bounds_m"]
    y_min, y_max = WORKSPACE_SPEC["y_bounds_m"]

    all_waypoints: list[list[float]] = []
    warnings: list[str] = []
    spatial_hallucinations = 0

    for idx, stroke in enumerate(strokes):
        label = stroke.get("label", f"stroke_{idx}")
        pts_xy = _sample_single_primitive_xy(stroke, ds=step_size_m)
        if not pts_xy:
            continue

        oob_pts = [p for p in pts_xy if not is_xy_in_bounds(p[0], p[1])]
        if oob_pts:
            spatial_hallucinations += 1
            bad_x, bad_y = oob_pts[0]
            warnings.append(
                f"Stroke '{label}' (#{idx}) has {len(oob_pts)}/{len(pts_xy)} points outside whiteboard "
                f"bounds X in [{x_min}, {x_max}], Y in [{y_min}, {y_max}] (e.g. [{bad_x:.3f}, {bad_y:.3f}]). "
                "Clamping to whiteboard border."
            )

        clamped_xy = [
            (
                float(np.clip(x, x_min, x_max)),
                float(np.clip(y, y_min, y_max)),
            )
            for x, y in pts_xy
        ]

        start_x, start_y = clamped_xy[0]
        end_x, end_y = clamped_xy[-1]

        # 1. Travel at safe height z_up to start of stroke
        all_waypoints.append([start_x, start_y, z_up])
        # 2. Lower marker onto whiteboard at z_draw
        all_waypoints.append([start_x, start_y, z_draw])
        # 3. Trace stroke along whiteboard
        for x, y in clamped_xy[1:]:
            all_waypoints.append([x, y, z_draw])
        # 4. Lift marker back to z_up after completing stroke
        all_waypoints.append([end_x, end_y, z_up])

    if not all_waypoints:
        return np.zeros((0, 3), dtype=np.float64), warnings, spatial_hallucinations

    dense_xyz = interpolate_waypoints_3d(all_waypoints, step_size_m=step_size_m)
    return dense_xyz, warnings, spatial_hallucinations
