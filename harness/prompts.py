"""Locked system and task prompts for PelicanBench multi-model evaluation."""

from __future__ import annotations

SYSTEM_PROMPT: str = """You control a 6-DoF SO-ARM101 robot arm holding a marker over a horizontal tabletop whiteboard.
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
4. **Permanent Additive Ink**: Once the marker touches the whiteboard, ink cannot be erased. Always lift to `z_up` when moving between disconnected shapes (or use `draw_primitives`, which handles `z_up` and `z_draw` transitions automatically).

### Available Tools
- `get_workspace_spec()`: Call this first to inspect the exact metric bounds and current marker tip pose.
- `draw_primitives(strokes)`: Execute a list of 2D primitives (`line`, `polyline`, `circle`, `arc`, `ellipse`, `bezier`) in meters. Automatically lifts to `z_up` before and after each primitive.
- `draw_waypoints(waypoints_xyz)`: Execute an explicit list of 3D `[X, Y, Z]` waypoints in meters.
- `run_python(code)`: Execute custom Python math using `np` (`numpy`), `math`, `WORKSPACE_SPEC`, `draw_primitives(strokes)`, and `draw_waypoints(waypoints_xyz)`.
- `inspect_canvas()`: Render and view the current 2D whiteboard image (with metric coordinate rulers) and the 3D overhead camera image so you can verify your drawing and add any missing details.
- `finish_drawing(summary)`: Call this when your pelican riding a bicycle is complete.
"""

TASK_PROMPT: str = """Draw a complete, well-proportioned **pelican riding a bicycle** on the SO-ARM101 whiteboard.

Make sure your drawing clearly includes all 5 anatomical and structural elements inside the metric bounds (`X in [0.160, 0.300]`, `Y in [-0.140, 0.140]`):
1. Two distinct circular bicycle wheels aligned horizontally near the bottom (`X ~ 0.185 to 0.200`), with hubs/spokes.
2. A connected bicycle diamond frame, seat post + saddle, fork, handlebars, and pedal crank.
3. The pelican's body and tail perched on top of the saddle (`X ~ 0.235 to 0.255`), with a wing or arm reaching toward the handlebars.
4. The pelican's curved neck, head, eye, and its signature long bill with a large curved throat pouch (`X ~ 0.265 to 0.288`).
5. The pelican's two legs extending down from its body to the bicycle pedals/crank.

Start by calling `get_workspace_spec()`, draw the bicycle and pelican using `draw_primitives` or `run_python`, optionally call `inspect_canvas()` to visually verify your work, and finish by calling `finish_drawing(summary)`."""
