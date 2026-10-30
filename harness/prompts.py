"""System and task prompts for PelicanBench multi-model evaluation."""

from __future__ import annotations

from pathlib import Path

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


REAL_SYSTEM_PROMPT_HEADER: str = """You control a physical 6-DoF SO-ARM101 robot arm equipped with three live cameras (`Camera 0` overhead, `Camera 1` side, `Camera 2` eye-in-hand wrist) over a tabletop drawing sheet and a marker holder bin.
Your task is to physically draw the best, most detailed **pelican riding a bicycle** you are capable of creating on the paper sheet.

This benchmark evaluates two dimensions:
1. **Embodied Robot Handling:** How accurately you use closed-loop visual servoing and encoder telemetry (`Reg 56` position and `Reg 60` load) to pick up markers and calibrate paper contact using your two robot skills (`so101-pickup-marker` and `so101-calibrate-paper-contact`).
2. **Artistic Taste and Spatial Composition:** How creatively and accurately you design and execute the pelican riding a bicycle once the marker is calibrated.

### Workflow & Human-Assisted Bin Marker Swap
1. **Pick Up Marker (`so101-pickup-marker`):**
   - At the start of the run, an uncapped marker is standing in the front low bin of the holder.
   - Follow the `so101-pickup-marker` skill below using the `so101_cli` tool (`status`, `move-polar`, `gripper`, `lift`) and the 3 returned camera images (`Camera 0`, `Camera 1`, `Camera 2`) to visually align with the barrel, clamp it in the flat pad zone (`2060 <= Reg 56 <= 2350`), and lift it straight up.
2. **Calibrate Paper Contact (`so101-calibrate-paper-contact`):**
   - Immediately after lifting the marker, follow the `so101-calibrate-paper-contact` skill below using `so101_cli` (`move-ik`) to orient the marker vertically (`90 degrees` to the table in `Camera 1`), find the near and far paper contact heights (`z_near` and `z_far`), and save the updated plane coefficients to `config/calibration.json` via `update_calibration(...)`.
3. **Draw with Your Own Taste (`so101_cli` `draw-strokes` / `draw_primitives` / `run_python`):**
   - Once paper contact is calibrated, design and execute your pelican riding a bicycle strokes.
   - You can pass normalized 2D strokes `[{"points": [[u0, v0], [u1, v1], ...]}, ...]` (`u, v` in `[0.15, 0.85]`) to `so101_cli` with `args=["draw-strokes", "<json_strokes>", "<prefix>"]` (which maps every point onto the calibrated plane in `config/calibration.json`, traces the strokes, and automatically retracts to the unobstructed pose and returns all 3 camera images), or use `draw_primitives` / `run_python`.
4. **Switching Marker Colors via Human Bin Swap (`request_marker_swap`):**
   - Whenever you want to switch to another marker color (`black`, `blue`, or `red`), call `request_marker_swap(color="<color>")`.
   - `request_marker_swap` automatically runs `safe-park release` to drop/release the current marker and park the arm safely out of the way, pauses the benchmark timer, and prompts the human operator to place the new uncapped marker color vertically into the front low bin.
   - As soon as `request_marker_swap` returns, the new marker is waiting in the bin: repeat **Step 1 (`so101-pickup-marker`)** and **Step 2 (`so101-calibrate-paper-contact`)** for the new marker, then draw that color layer.
5. **Finish (`finish_drawing`):**
   - After inspecting `Camera 0` to verify your completed pelican riding a bicycle drawing, call `finish_drawing(summary)`.
"""

REAL_TASK_PROMPT: str = """Draw a **pelican riding a bicycle** on the physical paper sheet using the SO-ARM101 robot arm.

Follow this closed-loop workflow:
1. Use the `so101-pickup-marker` skill (`so101_cli` with `status`, `move-polar`, `gripper`, `lift`) to visually servo onto the marker in the front low bin and verify `grasp_state == "BARREL_HELD_IN_FLAT_PAD"`.
2. Use the `so101-calibrate-paper-contact` skill (`so101_cli` with `move-ik` + `update_calibration`) to verify vertical orientation in `Camera 1` and calibrate the paper contact plane in `config/calibration.json`.
3. Design and draw your pelican riding a bicycle (using `so101_cli` `draw-strokes`, `draw_primitives`, or `run_python`). If you want additional marker colors (`black`, `blue`, `red`), call `request_marker_swap(color=...)` to drop the current marker and ask the human to place the new marker in the bin, then pick it up and calibrate it before drawing.
4. Visually verify your completed drawing in `Camera 0` and call `finish_drawing(summary)`."""


def load_real_skills_markdown() -> str:
    """Load the two physical SO-ARM101 skills from .agents/skills/."""
    repo_root = Path(__file__).resolve().parent.parent
    skills_dir = repo_root / ".agents" / "skills"
    skill_names = (
        "so101-pickup-marker",
        "so101-calibrate-paper-contact",
    )
    sections: list[str] = []
    for name in skill_names:
        skill_path = skills_dir / name / "SKILL.md"
        if skill_path.exists():
            sections.append(f"## Skill: `{name}`\n\n{skill_path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(sections)


def get_system_prompt(domain: str = "sim") -> str:
    """Return the system prompt for 'sim' or 'real' (with embedded physical skills)."""
    if domain == "real":
        skills_md = load_real_skills_markdown()
        if skills_md:
            return f"{REAL_SYSTEM_PROMPT_HEADER}\n\n# Loaded Robot Skills\n\n{skills_md}\n"
        return REAL_SYSTEM_PROMPT_HEADER
    return SYSTEM_PROMPT


def get_task_prompt(domain: str = "sim") -> str:
    """Return the initial user task prompt for 'sim' or 'real'."""
    if domain == "real":
        return REAL_TASK_PROMPT
    return TASK_PROMPT
