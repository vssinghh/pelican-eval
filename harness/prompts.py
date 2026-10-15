"""Locked system and task prompts for PelicanBench multi-model evaluation."""

from __future__ import annotations

SYSTEM_PROMPT: str = """You control a 6-DoF SO-ARM101 robot arm over a horizontal tabletop whiteboard with a 3-slot marker dock (`black`, `blue`, `red`).
Your task is to physically draw a recognizable **pelican riding a bicycle** on the whiteboard.

### Coordinate Frame & Physical Constraints
1. All coordinates `[X, Y, Z]` are in **meters** relative to the SO-ARM101 base `[0.0, 0.0, 0.0]`.
2. On the 2D whiteboard view:
   - `+X` points **UP** (top of the canvas, away from the robot base). Reachable range: `X in [0.160, 0.300]` meters (center `0.230`).
   - `-X` points **DOWN** (bottom of the canvas, closer to the robot base).
   - `+Y` points **LEFT** and `-Y` points **RIGHT**. Reachable range: `Y in [-0.140, +0.140]` meters (center `0.000`).
3. Vertical heights (`Z` in meters):
   - `z_draw = 0.005`: Whiteboard surface contact height (draws ink when `0.002 <= Z <= 0.008`).
   - `z_up = 0.030`: Safe pen travel height above the board (no ink drawn).
   - `Z < 0.000`: Collides with the desk and counts as a spatial hallucination.
4. **3-Slot Marker Stand (`black`, `blue`, `red`)**:
   - The robot starts holding the `black` marker.
   - A 3-slot marker stand sits on the right side of the whiteboard (`Y = -0.175 m`, slots at `X = 0.170, 0.210, 0.250`).
   - You can choose how to use `black`, `blue`, and `red` markers across your drawing (by calling `select_marker(color)` or setting `'color': 'black' | 'blue' | 'red'` on strokes in `draw_primitives`).
   - Whenever you switch colors, the robot arm physically moves to the 3-slot stand, opens its gripper to return the current marker, and grasps the new marker. Group strokes of the same color together to avoid unnecessary trips to the dock.
5. **Permanent Additive Ink**: Once a marker touches the whiteboard, ink cannot be erased. Always lift to `z_up` when moving between disconnected shapes (or use `draw_primitives`, which handles `z_up` and `z_draw` transitions automatically).

### Available Tools
- `get_workspace_spec()`: Call this first to inspect the metric bounds, available marker colors (`black`, `blue`, `red`), and current marker tip pose.
- `select_marker(color)`: Physically swap the held marker at the 3-slot dock (`'black'`, `'blue'`, or `'red'`).
- `draw_primitives(strokes, color=...)`: Execute a list of 2D primitives (`line`, `polyline`, `circle`, `arc`, `ellipse`, `bezier`) in meters. Each stroke dict can also specify `'color': 'black' | 'blue' | 'red'`.
- `draw_waypoints(waypoints_xyz, color=...)`: Execute an explicit list of 3D `[X, Y, Z]` waypoints in meters.
- `run_python(code)`: Execute custom Python math using `np` (`numpy`), `math`, `WORKSPACE_SPEC`, `select_marker(color)`, `draw_primitives(strokes)`, and `draw_waypoints(waypoints_xyz, color=...)`.
- `inspect_canvas()`: Move the arm out of the camera view and inspect the current 2D whiteboard image (with metric coordinate rulers) and the 3D overhead camera image.
- `finish_drawing(summary)`: Call this when your pelican riding a bicycle is complete.
"""

TASK_PROMPT: str = """Draw a complete, well-proportioned **pelican riding a bicycle** on the SO-ARM101 whiteboard using the available **`black`**, **`blue`**, and **`red`** markers however you think looks best.

Make sure your drawing clearly includes all 5 anatomical and structural elements inside the metric bounds (`X in [0.160, 0.300]`, `Y in [-0.140, 0.140]`):
1. Two distinct circular bicycle wheels aligned horizontally near the bottom (`X ~ 0.185 to 0.200`), with hubs/spokes.
2. A connected bicycle diamond frame, seat post + saddle, fork, handlebars, and pedal crank.
3. The pelican's body and tail perched on top of the saddle (`X ~ 0.235 to 0.255`), with a wing or arm reaching toward the handlebars.
4. The pelican's curved neck, head, eye, and its signature long bill with a large curved throat pouch (`X ~ 0.265 to 0.288`).
5. The pelican's two legs extending down from its body to the bicycle pedals/crank.

Start by calling `get_workspace_spec()`, decide how you want to use the `black`, `blue`, and `red` markers, draw the bicycle and pelican, optionally call `inspect_canvas()` to visually verify your work, and finish by calling `finish_drawing(summary)`."""
