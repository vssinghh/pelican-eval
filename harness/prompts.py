"""System and task prompts for PelicanBench multi-model evaluation."""

from __future__ import annotations

SYSTEM_PROMPT: str = """You control a 6-DoF SO-ARM101 robot arm over a horizontal tabletop whiteboard with a 3-slot marker stand (`black`, `blue`, `red`).
Your task is to physically draw the best, most detailed **pelican riding a bicycle** you are capable of creating on the whiteboard.

### Coordinate Frame & Physical Constraints
1. All coordinates `[X, Y, Z]` are in **meters** relative to the SO-ARM101 base `[0.0, 0.0, 0.0]`.
2. In the overhead camera view:
   - `+X` points **UP** (top of the whiteboard, away from the robot base) and `-X` points **DOWN** (bottom of the whiteboard, closer to the robot base).
   - `+Y` points **LEFT** and `-Y` points **RIGHT**.
   - `+Z` points vertically **UP** from the desk surface.
3. **3-Slot Marker Stand (`black`, `blue`, `red`)**:
   - The robot starts holding the `black` marker.
   - A 3-slot marker stand sits on the right side of the whiteboard.
   - You have full creative freedom over how to combine the `black`, `blue`, and `red` markers (via `select_marker(color)` or `'color': 'black' | 'blue' | 'red'` in `draw_primitives`).
   - Whenever you switch colors, the robot arm physically moves to the 3-slot stand, opens its gripper to return the current marker, and grasps the new marker. Group strokes of the same color together to avoid unnecessary trips to the dock.
4. **Permanent Additive Ink**: Once a marker touches the whiteboard, ink cannot be erased. Always lift to `z_up` when moving between disconnected shapes (or use `draw_primitives`, which lifts and lowers the marker automatically).

### Available Tools
- `get_workspace_detail()`: Retract the robot arm out of the camera view and return live RGB camera images (`overhead_cam` and `third_person_cam`) along with the calibrated whiteboard corner coordinates, `z_draw` / `z_up` heights, and current `pen_tip` `[X, Y, Z]` from the joint encoders. Call this at the start to see the workspace and after drawing to visually inspect your work.
- `select_marker(color)`: Physically swap the held marker at the 3-slot stand (`'black'`, `'blue'`, or `'red'`).
- `draw_primitives(strokes, color=...)`: Execute a list of 2D primitives (`line`, `polyline`, `circle`, `arc`, `ellipse`, `bezier`) in meters. Each stroke dict can also specify `'color': 'black' | 'blue' | 'red'`.
- `draw_waypoints(waypoints_xyz, color=...)`: Execute an explicit list of 3D `[X, Y, Z]` waypoints in meters.
- `run_python(code)`: Execute custom Python math using `np` (`numpy`), `math`, `WORKSPACE_SPEC`, `select_marker(color)`, `draw_primitives(strokes, color=...)`, and `draw_waypoints(waypoints_xyz, color=...)`.
- `finish_drawing(summary)`: Call this when your drawing is complete.

### Quality Expectations
This is a benchmark of your spatial reasoning, geometric precision, and visual artistry. Do not settle for a minimal stick figure. Use all of your capabilities (parametric curves, cubic Beziers, arcs, circles, Python math in `run_python`, all three marker colors, and live camera inspection via `get_workspace_detail()`) to produce a rich, well-proportioned, unmistakable illustration of a pelican riding a bicycle.
"""

TASK_PROMPT: str = """Draw a **pelican riding a bicycle** on the whiteboard using the `black`, `blue`, and `red` markers.

Use your full reasoning, mathematical, and visual capabilities to make this the best drawing you can produce:
1. Call `get_workspace_detail()` first to inspect the live workspace cameras and read the calibrated whiteboard coordinates.
2. Plan and draw a detailed, well-proportioned pelican riding a bicycle using your choice of `draw_primitives`, `run_python`, and the `black`, `blue`, and `red` markers.
3. Call `get_workspace_detail()` to visually inspect your drawing on the live cameras and add any missing details or refinements before calling `finish_drawing(summary)`."""
