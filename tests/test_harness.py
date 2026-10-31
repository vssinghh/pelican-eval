"""Unit tests for the hermetic Python sandbox, 6 tools, and hallucination counters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from envs.real_env import RealWhiteboardEnv
from envs.sim_env import SimWhiteboardEnv
from harness.prompts import get_system_prompt, get_task_prompt
from harness.tools import WhiteboardToolBelt, get_tool_schemas


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


def test_real_skills_and_human_bin_swap_tools(tmp_path: Path) -> None:
    """Verify the 2 physical skills load into the real prompt and so101_cli/request_marker_swap dispatch."""
    real_sys = get_system_prompt("real")
    real_task = get_task_prompt("real")
    assert "so101-pickup-marker" in real_sys
    assert "so101-calibrate-paper-contact" in real_sys
    assert "request_marker_swap" in real_sys
    assert "request_marker_swap" in real_task

    schemas = get_tool_schemas("real")
    names = {s["function"]["name"] for s in schemas}
    assert {"so101_cli", "update_calibration", "request_marker_swap"}.issubset(names)
    assert "select_marker" not in names

    tmp_cal = tmp_path / "calibration.json"
    tmp_cal.write_text(
        json.dumps(
            {
                "plane_coeffs_abc": [-0.08, 0.0, 0.0745],
                "single_marker_mode": True,
                "interactive_holder_swap": False,
                "pickup_on_reset": False,
                "target_pitch": 1.65,
            }
        ),
        encoding="utf-8",
    )

    env = RealWhiteboardEnv(calibration_path=tmp_cal, mock_hardware=True)
    try:
        belt = WhiteboardToolBelt(env)
        assert belt.skill_stages["pickup_verified"] is False
        assert belt.skill_stages["calibration_saved"] is False
        assert belt.skill_stages["drawing_executed"] is False

        # 0. Attempting silent color swap via draw_primitives(color='red') in single-marker mode is blocked
        res_unswapped = belt.dispatch(
            "draw_primitives",
            {"strokes": [{"type": "line", "from_xy": [0.20, -0.04], "to_xy": [0.22, -0.02], "color": "red"}]},
        )
        unswapped_data = json.loads(res_unswapped.response_json)
        assert unswapped_data["status"] == "error"
        assert "request_marker_swap" in unswapped_data["error"]
        assert belt.code_hallucinations == 1

        # 1. so101_cli lift verifies pickup in flat pad
        res_cli = belt.dispatch("so101_cli", {"args": ["lift", "0.060", "test_step"]})
        cli_data = json.loads(res_cli.response_json)
        assert cli_data["status"] == "ok"
        assert cli_data["grasp_state"] == "BARREL_HELD_IN_FLAT_PAD"
        assert len(res_cli.images_base64_png) == 2
        assert belt.skill_stages["pickup_verified"] is True
        assert belt.skill_stages["pickups_verified"] == 1

        # 2. update_calibration marks calibration_saved
        res_cal = belt.dispatch(
            "update_calibration",
            {"plane_coeffs_abc": [-0.08, 0.0, 0.0765], "target_pitch": 1.65},
        )
        cal_data = json.loads(res_cal.response_json)
        assert cal_data["status"] == "ok"
        assert belt.skill_stages["calibration_saved"] is True
        assert belt.skill_stages["calibrations_saved"] == 1

        # 3. request_marker_swap to blue resets pickup awaiting flag and increments swaps_requested
        res_swap = belt.dispatch("request_marker_swap", {"color": "blue"})
        swap_data = json.loads(res_swap.response_json)
        assert swap_data["status"] == "ok"
        assert swap_data["requested_color"] == "blue"
        assert env.current_marker_color == "blue"
        assert env.marker_swaps == 1
        assert belt.skill_stages["swaps_requested"] == 1

        # Second pickup after swap increments pickups_verified to 2
        belt.dispatch("so101_cli", {"args": ["lift", "0.060", "pickup_blue"]})
        assert belt.skill_stages["pickups_verified"] == 2

        # 4. so101_cli draw-strokes mirrors normalized (u, v) strokes to canvas and marks drawing_executed
        res_draw = belt.dispatch(
            "so101_cli",
            {"args": ["draw-strokes", '[{"points": [[0.25, 0.30], [0.75, 0.30]]}]', "test_draw"]},
        )
        draw_data = json.loads(res_draw.response_json)
        assert draw_data["status"] == "ok"
        assert env.ink_pixels_drawn > 100
        assert "blue" in env.colors_used
        assert belt.skill_stages["drawing_executed"] is True
        assert belt.skill_stages["drawing_strokes_executed"] == 1
    finally:
        env.close()


def test_real_env_gripper_safe_holding_guard() -> None:
    """Verify RealWhiteboardEnv caps Motor 6 holding torque at 450 and resets Reg 60=0 without opening jaws."""
    env = RealWhiteboardEnv(mock_hardware=True)
    writes_1b: list[tuple[int, int, int]] = []
    writes_2b: list[tuple[int, int, int]] = []
    reg_state = {56: 2130, 60: 0}

    class FakePacketHandler:
        def write1ByteTxRx(self, _port: Any, sid: int, reg: int, val: int) -> tuple[int, int]:
            writes_1b.append((sid, reg, val))
            return 0, 0

        def write2ByteTxRx(self, _port: Any, sid: int, reg: int, val: int) -> tuple[int, int]:
            writes_2b.append((sid, reg, val))
            return 0, 0

        def read2ByteTxRx(self, _port: Any, sid: int, reg: int) -> tuple[int, int, int]:
            if sid == 6 and reg in reg_state:
                return reg_state[reg], 0, 0
            return 2048, 0, 0

    try:
        env.mock_hardware = False
        env._port_handler = object()
        env._packet_handler = FakePacketHandler()

        env._configure_gripper_safe_holding()
        assert (6, 48, 450) in writes_2b

        writes_1b.clear()
        writes_2b.clear()
        hold_goal = env._compute_safe_gripper_hold_tick(1700)
        assert hold_goal == 2020  # max(1700, 2130 - 110)
        assert (6, 48, 450) in writes_2b
        # Overload latch reset must write exact cur_g (2130), never opening jaws
        assert (6, 42, 2130) in writes_2b
        assert (6, 40, 0) in writes_1b
        assert (6, 40, 1) in writes_1b
    finally:
        env.mock_hardware = True
        env._port_handler = None
        env._packet_handler = None
        env.close()


