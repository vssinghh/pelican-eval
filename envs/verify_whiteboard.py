"""Deterministic verification script drawing a 3-color Pelican Riding a Bicycle in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from envs.real_env import RealWhiteboardEnv
from envs.sim_env import SimWhiteboardEnv
from harness.tools import WhiteboardToolBelt


def get_reference_pelican_strokes() -> list[dict[str, Any]]:
    """Return a list of metric 2D primitives depicting a pelican riding a bicycle using 3 marker colors.

    Color Plan:
    - 'black': Rear & front bicycle wheels, hubs, chainring, and pelican eye.
    - 'red':   Bicycle diamond frame, seat tube, saddle, fork, handlebars, and pelican beak/legs.
    - 'blue':  Pelican body, tail feathers, wing reaching to handlebars, neck, and head.
    """
    rear_wheel_c = [0.192, 0.055]
    front_wheel_c = [0.192, -0.055]
    crank_c = [0.194, 0.008]
    seat_top = [0.224, 0.032]
    head_tube = [0.222, -0.038]
    handlebar_grip = [0.236, -0.028]

    return [
        # Batch 1 (BLACK marker): Wheels, Hubs, Chainring, and Eye
        {
            "type": "circle",
            "center_xy": rear_wheel_c,
            "radius": 0.024,
            "color": "black",
            "label": "rear_wheel",
        },
        {
            "type": "circle",
            "center_xy": rear_wheel_c,
            "radius": 0.004,
            "color": "black",
            "label": "rear_hub",
        },
        {
            "type": "circle",
            "center_xy": front_wheel_c,
            "radius": 0.024,
            "color": "black",
            "label": "front_wheel",
        },
        {
            "type": "circle",
            "center_xy": front_wheel_c,
            "radius": 0.004,
            "color": "black",
            "label": "front_hub",
        },
        {
            "type": "circle",
            "center_xy": crank_c,
            "radius": 0.007,
            "color": "black",
            "label": "chainring",
        },
        {
            "type": "circle",
            "center_xy": [0.282, -0.018],
            "radius": 0.0018,
            "color": "black",
            "label": "pelican_eye",
        },
        # Batch 2 (RED marker): Bicycle Frame, Saddle, Handlebars, Beak & Pouch, and Legs
        {
            "type": "polyline",
            "points_xy": [rear_wheel_c, crank_c, head_tube, seat_top, rear_wheel_c],
            "closed": False,
            "color": "red",
            "label": "bike_frame",
        },
        {
            "type": "line",
            "from_xy": crank_c,
            "to_xy": seat_top,
            "color": "red",
            "label": "seat_tube",
        },
        {
            "type": "line",
            "from_xy": [0.224, 0.042],
            "to_xy": [0.224, 0.022],
            "color": "red",
            "label": "saddle",
        },
        {
            "type": "polyline",
            "points_xy": [front_wheel_c, head_tube, [0.238, -0.042], handlebar_grip],
            "closed": False,
            "color": "red",
            "label": "fork_and_handlebars",
        },
        {
            "type": "polyline",
            "points_xy": [
                [0.284, -0.022],
                [0.283, -0.068],
                [0.279, -0.070],
                [0.278, -0.022],
            ],
            "closed": False,
            "color": "red",
            "label": "upper_beak",
        },
        {
            "type": "bezier",
            "p0": [0.278, -0.022],
            "p1": [0.260, -0.036],
            "p2": [0.262, -0.060],
            "p3": [0.279, -0.068],
            "color": "red",
            "label": "beak_throat_pouch",
        },
        {
            "type": "polyline",
            "points_xy": [[0.230, 0.020], [0.212, 0.016], [0.200, 0.004], [0.200, -0.003]],
            "closed": False,
            "color": "red",
            "label": "front_leg_pedal",
        },
        {
            "type": "polyline",
            "points_xy": [[0.230, 0.026], [0.210, 0.022], [0.188, 0.012], [0.188, 0.005]],
            "closed": False,
            "color": "red",
            "label": "rear_leg_pedal",
        },
        # Batch 3 (BLUE marker): Pelican Body, Tail, Wing, Neck, and Head
        {
            "type": "ellipse",
            "center_xy": [0.244, 0.022],
            "rx": 0.015,
            "ry": 0.026,
            "rotation": -0.22,
            "color": "blue",
            "label": "pelican_body",
        },
        {
            "type": "polyline",
            "points_xy": [[0.242, 0.047], [0.248, 0.062], [0.236, 0.058], [0.238, 0.046]],
            "closed": False,
            "color": "blue",
            "label": "tail_feathers",
        },
        {
            "type": "bezier",
            "p0": [0.248, 0.028],
            "p1": [0.256, 0.010],
            "p2": [0.246, -0.010],
            "p3": handlebar_grip,
            "color": "blue",
            "label": "pelican_wing_to_bars",
        },
        {
            "type": "bezier",
            "p0": [0.254, 0.002],
            "p1": [0.265, 0.008],
            "p2": [0.272, -0.002],
            "p3": [0.278, -0.012],
            "color": "blue",
            "label": "neck_back",
        },
        {
            "type": "bezier",
            "p0": [0.246, -0.002],
            "p1": [0.258, 0.000],
            "p2": [0.266, -0.010],
            "p3": [0.272, -0.018],
            "color": "blue",
            "label": "neck_front",
        },
        {
            "type": "circle",
            "center_xy": [0.280, -0.016],
            "radius": 0.008,
            "color": "blue",
            "label": "pelican_head",
        },
    ]


def main() -> None:
    """Run deterministic verification of the 3-color whiteboard environment and save visual artifacts."""
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
        toolbelt = WhiteboardToolBelt(env)
        strokes = get_reference_pelican_strokes()
        stats = toolbelt._exec_draw_primitives_internal(strokes)
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
        print(f"  Colors used:            {stats['colors_used']}")
        print(f"  Marker swaps at dock:   {stats['marker_swaps']}")
        print(f"  Waypoints executed:     {stats['waypoints_commanded']}")
        print(f"  Total ink pixels:       {stats['total_ink_pixels']}")
        print(f"  Max IK error (mm):      {stats['max_ik_error_mm']}")
        print(f"  Spatial hallucinations: {stats['spatial_hallucinations']}")
        if stats["warnings"]:
            print(f"  Warnings:               {stats['warnings']}")
        print(f"  Saved artifacts to:     {out_dir.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
