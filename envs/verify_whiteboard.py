"""Deterministic verification script drawing a reference Pelican Riding a Bicycle in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from envs.real_env import RealWhiteboardEnv
from envs.sim_env import SimWhiteboardEnv
from envs.trajectory import compile_primitives_to_waypoints


def get_reference_pelican_strokes() -> list[dict[str, Any]]:
    """Return a clean list of metric 2D primitives depicting a pelican riding a bicycle.

    Coordinate Reminder:
    - +X is UP (top of canvas, range [0.160, 0.300] m)
    - +Y is LEFT, -Y is RIGHT (range [-0.140, +0.140] m)
    - Bicycle faces RIGHT (-Y direction): rear wheel on left (+Y), front wheel on right (-Y).
    """
    rear_wheel_c = [0.192, 0.055]
    front_wheel_c = [0.192, -0.055]
    crank_c = [0.194, 0.008]
    seat_top = [0.224, 0.032]
    head_tube = [0.222, -0.038]
    handlebar_grip = [0.236, -0.028]

    return [
        # 1. Rear and Front Wheels + Inner Hubs
        {"type": "circle", "center_xy": rear_wheel_c, "radius": 0.024, "label": "rear_wheel"},
        {"type": "circle", "center_xy": rear_wheel_c, "radius": 0.004, "label": "rear_hub"},
        {"type": "circle", "center_xy": front_wheel_c, "radius": 0.024, "label": "front_wheel"},
        {"type": "circle", "center_xy": front_wheel_c, "radius": 0.004, "label": "front_hub"},
        # 2. Bicycle Diamond Frame, Fork, Seat Post, and Handlebars
        {
            "type": "polyline",
            "points_xy": [rear_wheel_c, crank_c, head_tube, seat_top, rear_wheel_c],
            "closed": False,
            "label": "bike_frame",
        },
        {"type": "line", "from_xy": crank_c, "to_xy": seat_top, "label": "seat_tube"},
        {
            "type": "line",
            "from_xy": [0.224, 0.042],
            "to_xy": [0.224, 0.022],
            "label": "saddle",
        },
        {
            "type": "polyline",
            "points_xy": [front_wheel_c, head_tube, [0.238, -0.042], handlebar_grip],
            "closed": False,
            "label": "fork_and_handlebars",
        },
        {"type": "circle", "center_xy": crank_c, "radius": 0.007, "label": "chainring"},
        # 3. Pelican Plump Body & Tail perched above the Saddle
        {
            "type": "ellipse",
            "center_xy": [0.244, 0.022],
            "rx": 0.015,
            "ry": 0.026,
            "rotation": -0.22,
            "label": "pelican_body",
        },
        {
            "type": "polyline",
            "points_xy": [[0.242, 0.047], [0.248, 0.062], [0.236, 0.058], [0.238, 0.046]],
            "closed": False,
            "label": "tail_feathers",
        },
        # 4. Pelican Folded Wing & Arm reaching to Handlebars
        {
            "type": "bezier",
            "p0": [0.248, 0.028],
            "p1": [0.256, 0.010],
            "p2": [0.246, -0.010],
            "p3": handlebar_grip,
            "label": "pelican_wing_to_bars",
        },
        # 5. Pelican S-Curved Neck & Head
        {
            "type": "bezier",
            "p0": [0.254, 0.002],
            "p1": [0.265, 0.008],
            "p2": [0.272, -0.002],
            "p3": [0.278, -0.012],
            "label": "neck_back",
        },
        {
            "type": "bezier",
            "p0": [0.246, -0.002],
            "p1": [0.258, 0.000],
            "p2": [0.266, -0.010],
            "p3": [0.272, -0.018],
            "label": "neck_front",
        },
        {
            "type": "circle",
            "center_xy": [0.280, -0.016],
            "radius": 0.008,
            "label": "pelican_head",
        },
        {
            "type": "circle",
            "center_xy": [0.282, -0.018],
            "radius": 0.0018,
            "label": "pelican_eye",
        },
        # 6. Characteristic Long Upper Bill & Large Pouched Lower Beak
        {
            "type": "polyline",
            "points_xy": [
                [0.284, -0.022],
                [0.283, -0.068],
                [0.279, -0.070],
                [0.278, -0.022],
            ],
            "closed": False,
            "label": "upper_beak",
        },
        {
            "type": "bezier",
            "p0": [0.278, -0.022],
            "p1": [0.260, -0.036],
            "p2": [0.262, -0.060],
            "p3": [0.279, -0.068],
            "label": "beak_throat_pouch",
        },
        # 7. Pelican Webbed Legs Pedaling on the Crank
        {
            "type": "polyline",
            "points_xy": [[0.230, 0.020], [0.212, 0.016], [0.200, 0.004], [0.200, -0.003]],
            "closed": False,
            "label": "front_leg_pedal",
        },
        {
            "type": "polyline",
            "points_xy": [[0.230, 0.026], [0.210, 0.022], [0.188, 0.012], [0.188, 0.005]],
            "closed": False,
            "label": "rear_leg_pedal",
        },
    ]


def main() -> None:
    """Run deterministic verification of the whiteboard environment and save visual artifacts."""
    parser = argparse.ArgumentParser(description="Verify SO-ARM101 whiteboard drawing pipeline.")
    parser.add_argument("-d", dest="domain", default="sim", choices=("sim", "real"))
    parser.add_argument("-p", dest="port", default="/dev/tty.usbmodem5B790143431")
    parser.add_argument("-o", dest="out_dir", default="assets/verification")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = (
        SimWhiteboardEnv(record_timelapse=True)
        if args.domain == "sim"
        else RealWhiteboardEnv(serial_port=args.port, mock_hardware=False)
    )

    try:
        env.reset()
        strokes = get_reference_pelican_strokes()
        waypoints_xyz, warnings, hallucinations = compile_primitives_to_waypoints(strokes)
        stats = env.execute_cartesian_path(waypoints_xyz)
        env.move_to_home()
        obs = env.get_obs()

        Image.fromarray(obs["canvas_rgb"]).save(out_dir / "reference_canvas.png")
        Image.fromarray(obs["canvas_annotated_rgb"]).save(out_dir / "reference_canvas_grid.png")
        Image.fromarray(obs["rgb_overhead_cam"]).save(out_dir / "reference_overhead.png")
        Image.fromarray(obs["rgb_third_person_cam"]).save(out_dir / "reference_third_person.png")

        if env.timelapse_frames:
            pil_frames = [Image.fromarray(f) for f in env.timelapse_frames]
            pil_frames[0].save(
                out_dir / "reference_timelapse.gif",
                save_all=True,
                append_images=pil_frames[1:],
                duration=80,
                loop=0,
            )

        print(f"Verification complete in domain='{args.domain}':")
        print(f"  Primitive strokes:      {len(strokes)}")
        print(f"  Waypoints executed:     {stats['waypoints_commanded']}")
        print(f"  Total ink pixels:       {stats['total_ink_pixels']}")
        print(f"  Max IK error (mm):      {stats['max_ik_error_mm']}")
        print(f"  Spatial hallucinations: {hallucinations + stats['spatial_hallucinations']}")
        if warnings or stats["warnings"]:
            print(f"  Warnings:               {warnings + stats['warnings']}")
        print(f"  Saved artifacts to:     {out_dir.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
