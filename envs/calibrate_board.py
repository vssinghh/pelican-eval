"""4-Corner Physical Whiteboard Plane Calibration CLI for Level 2 SO-ARM101 Execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from envs.base import AVAILABLE_MARKER_COLORS, MARKER_DOCK_SLOTS_XYZ, WORKSPACE_SPEC
from envs.ik_solver import SO101IKSolver
from envs.real_env import RealWhiteboardEnv


def fit_plane_from_points(points_xyz: np.ndarray) -> tuple[float, float, float]:
    """Fit least-squares plane Z(X, Y) = a*X + b*Y + c to N >= 3 3D points."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError(f"Expected (N>=3, 3) array of 3D points, got shape {pts.shape}")
    design = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts), dtype=np.float64)])
    coeffs, _, _, _ = np.linalg.lstsq(design, pts[:, 2], rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def main() -> None:
    """Interactive 4-corner + 3-marker-slot calibration utility for the physical SO-ARM101."""
    parser = argparse.ArgumentParser(
        description="Calibrate physical whiteboard Z(X, Y) plane and 3-slot marker stand for SO-ARM101."
    )
    parser.add_argument(
        "-p",
        dest="port",
        default="/dev/tty.usbmodem5B790143431",
        help="Serial port for SO-ARM101 follower arm.",
    )
    parser.add_argument(
        "-m",
        dest="mock",
        action="store_true",
        help="Run with mock hardware to generate default flat calibration file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "config" / "calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corners = [
        ("Front-Left (X_min=0.16, Y_max=+0.14)", [0.160, 0.140, float(WORKSPACE_SPEC["z_draw_m"])]),
        ("Front-Right (X_min=0.16, Y_min=-0.14)", [0.160, -0.140, float(WORKSPACE_SPEC["z_draw_m"])]),
        ("Back-Left (X_max=0.30, Y_max=+0.14)", [0.300, 0.140, float(WORKSPACE_SPEC["z_draw_m"])]),
        ("Back-Right (X_max=0.30, Y_min=-0.14)", [0.300, -0.140, float(WORKSPACE_SPEC["z_draw_m"])]),
    ]

    marker_slots: dict[str, list[float]] = {
        color: list(MARKER_DOCK_SLOTS_XYZ[color]) for color in AVAILABLE_MARKER_COLORS
    }

    if args.mock:
        pts = np.array([c[1] for c in corners], dtype=np.float64)
    else:
        env = RealWhiteboardEnv(serial_port=args.port, mock_hardware=False)
        env.set_torque(False)
        solver = SO101IKSolver(site_name="pen_tip")
        recorded: list[np.ndarray] = []
        try:
            for label, _ in corners:
                input(f"Touch marker tip lightly to {label} and press Enter...")
                q_6d = env._read_proprio_6d()
                xyz = solver.forward_kinematics(q_6d[:5])
                print(f"  Recorded pen_tip XYZ: {np.round(xyz, 4)}")
                recorded.append(xyz)

            for color in AVAILABLE_MARKER_COLORS:
                ans = input(
                    f"Move gripper to the '{color.upper()}' marker stand slot and press Enter "
                    "(or type 's' to keep default)... "
                )
                if ans.strip().lower() != "s":
                    q_6d = env._read_proprio_6d()
                    xyz = solver.forward_kinematics(q_6d[:5])
                    marker_slots[color] = [round(float(v), 5) for v in xyz]
                    print(f"  Recorded '{color}' slot XYZ: {marker_slots[color]}")
        finally:
            env.close()
        pts = np.asarray(recorded, dtype=np.float64)

    a, b, c = fit_plane_from_points(pts)
    payload = {
        "serial_port": args.port,
        "plane_equation": "Z_board(X, Y) = a * X + b * Y + c",
        "plane_coeffs_abc": [round(a, 6), round(b, 6), round(c, 6)],
        "recorded_corners_xyz": np.round(pts, 5).tolist(),
        "marker_slots_xyz": marker_slots,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved 4-corner + 3-slot calibration to {out_path}: a={a:.5f}, b={b:.5f}, c={c:.5f}")


if __name__ == "__main__":
    main()
