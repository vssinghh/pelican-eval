"""Unit tests for the hermetic Python sandbox, 6 tools, and hallucination counters."""

from __future__ import annotations

import json

from envs.sim_env import SimWhiteboardEnv
from harness.tools import WhiteboardToolBelt


def test_tool_belt_and_hallucination_tracking() -> None:
    """Verify valid tool execution, spatial hallucination detection, and code sandbox security."""
    env = SimWhiteboardEnv(record_timelapse=False)
    try:
        belt = WhiteboardToolBelt(env)

        # 1. get_workspace_detail returns calibrated bounds and 2 live camera images
        res_spec = belt.dispatch("get_workspace_detail", "{}")
        spec_data = json.loads(res_spec.response_json)
        assert spec_data["x_bounds_m"] == [0.16, 0.3]
        assert len(res_spec.images_base64_png) == 2
        assert belt.total_tool_calls == 1

        # 2. Valid run_python drawing a wheel circle
        py_code = """
wheel = [{"type": "circle", "center_xy": [0.20, 0.04], "radius": 0.02, "label": "wheel"}]
stats = draw_primitives(wheel)
print("Drew pixels:", stats["ink_pixels_added"])
"""
        res_py = belt.dispatch("run_python", {"code": py_code})
        py_data = json.loads(res_py.response_json)
        assert py_data["status"] == "ok"
        assert py_data["ink_pixels_added"] > 100
        assert belt.code_hallucinations == 0
        assert belt.spatial_hallucinations == 0

        # 3. Forbidden import inside run_python increments code_hallucinations
        bad_py = "import os\nos.system('echo hacked')"
        res_bad_py = belt.dispatch("run_python", {"code": bad_py})
        bad_py_data = json.loads(res_bad_py.response_json)
        assert bad_py_data["status"] == "error"
        assert belt.code_hallucinations == 1

        # 4. Air-drawing in draw_waypoints (staying at Z=0.030m, 0 ink drawn) increments spatial_hallucinations
        res_air = belt.dispatch(
            "draw_waypoints",
            {"waypoints_xyz": [[0.20, 0.0, 0.030], [0.24, 0.0, 0.030]]},
        )
        air_data = json.loads(res_air.response_json)
        assert air_data["ink_pixels_added"] == 0
        assert belt.spatial_hallucinations == 1

        # 5. Unknown tool increments code_hallucinations
        belt.dispatch("erase_board", "{}")
        assert belt.code_hallucinations == 2

        # 6. finish_drawing marks episode finished
        res_fin = belt.dispatch("finish_drawing", {"summary": "Done testing."})
        assert res_fin.is_finish
        assert belt.finished
    finally:
        env.close()


def test_three_color_marker_dock_swaps() -> None:
    """Verify autonomous 3-slot marker dock swaps ('black', 'blue', 'red') and invalid color hallucinations."""
    env = SimWhiteboardEnv(record_timelapse=False)
    try:
        belt = WhiteboardToolBelt(env)
        assert env.current_marker_color == "black"

        # 1. Swap to red marker via select_marker tool
        res_red = belt.dispatch("select_marker", {"color": "red"})
        red_data = json.loads(res_red.response_json)
        assert red_data["status"] == "ok"
        assert red_data["swapped"] is True
        assert red_data["active_color"] == "red"
        assert env.marker_swaps == 1

        # 2. Draw strokes with per-stroke color ('red' then 'blue')
        res_draw = belt.dispatch(
            "draw_primitives",
            {
                "strokes": [
                    {"type": "line", "from_xy": [0.20, -0.05], "to_xy": [0.24, -0.05], "color": "red"},
                    {"type": "line", "from_xy": [0.20, 0.05], "to_xy": [0.24, 0.05], "color": "blue"},
                ]
            },
        )
        draw_data = json.loads(res_draw.response_json)
        assert draw_data["status"] == "ok"
        assert draw_data["active_marker_color"] == "blue"
        assert set(draw_data["colors_used"]) == {"red", "blue"}
        assert env.marker_swaps == 2

        # 3. Persistent run_python state across calls + draw_primitives(strokes, color='red')
        belt.dispatch("run_python", {"code": "def my_strokes():\n    return [{'type': 'line', 'from_xy': [0.22, -0.02], 'to_xy': [0.25, -0.02]}]"})
        res_sb_color = belt.dispatch("run_python", {"code": "draw_primitives(my_strokes(), color='red')"})
        sb_color_data = json.loads(res_sb_color.response_json)
        assert sb_color_data["status"] == "ok"
        assert sb_color_data["active_marker_color"] == "red"
        assert belt.code_hallucinations == 0

        # 4. Invalid marker color increments code_hallucinations
        res_bad = belt.dispatch("select_marker", {"color": "purple"})
        bad_data = json.loads(res_bad.response_json)
        assert bad_data["status"] == "error"
        assert belt.code_hallucinations == 1
    finally:
        env.close()
