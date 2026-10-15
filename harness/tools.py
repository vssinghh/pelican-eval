"""Tool definitions, JSON schemas, and hallucination tracking for PelicanBench."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from envs.base import AVAILABLE_MARKER_COLORS, WORKSPACE_SPEC, SO101WhiteboardEnv
from envs.trajectory import compile_primitives_to_waypoints
from harness.python_sandbox import RestrictedPythonSandbox

OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_workspace_spec",
            "description": (
                "Return the metric coordinate frame, reachable whiteboard bounds [X, Y, Z] in meters, "
                "available marker colors ('black', 'blue', 'red'), 3-slot dock coordinates, "
                "pen contact height z_draw, safe travel height z_up, and current marker tip position."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_marker",
            "description": (
                "Move the robot arm to the 3-slot marker stand on the side of the desk, "
                "open the gripper to return the current marker to its slot, and pick up the requested "
                "marker color ('black', 'blue', or 'red')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": list(AVAILABLE_MARKER_COLORS),
                        "description": "Marker color to pick up from the 3-slot stand: 'black', 'blue', or 'red'.",
                    }
                },
                "required": ["color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_primitives",
            "description": (
                "Draw a list of 2D geometric strokes on the whiteboard in meters. "
                "Automatically lifts the marker to z_up before and after each stroke. "
                "Each stroke dict can optionally include 'color': 'black' | 'blue' | 'red' to trigger a physical "
                "marker swap at the 3-slot dock before drawing that stroke. "
                "Supported stroke dicts: "
                "1) {'type': 'line', 'from_xy': [x1, y1], 'to_xy': [x2, y2], 'color': str, 'label': str} "
                "2) {'type': 'polyline', 'points_xy': [[x0, y0], [x1, y1], ...], 'closed': bool, 'color': str, 'label': str} "
                "3) {'type': 'circle', 'center_xy': [cx, cy], 'radius': r, 'color': str, 'label': str} "
                "4) {'type': 'arc', 'center_xy': [cx, cy], 'radius': r, 'start_angle': rad1, 'end_angle': rad2, 'color': str, 'label': str} "
                "5) {'type': 'ellipse', 'center_xy': [cx, cy], 'rx': rx, 'ry': ry, 'rotation': rad, 'color': str, 'label': str} "
                "6) {'type': 'bezier', 'p0': [x0, y0], 'p1': [x1, y1], 'p2': [x2, y2], 'p3': [x3, y3], 'color': str, 'label': str}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strokes": {
                        "type": "array",
                        "description": "List of geometric primitive dictionaries with coordinates in meters.",
                        "items": {"type": "object"},
                    },
                    "color": {
                        "type": "string",
                        "enum": list(AVAILABLE_MARKER_COLORS),
                        "description": "Optional default marker color ('black', 'blue', or 'red') for this batch of strokes.",
                    },
                },
                "required": ["strokes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_waypoints",
            "description": (
                "Move the marker tip sequentially through an explicit list of 3D [X, Y, Z] metric coordinates "
                "in the robot base frame. Use Z = 0.005 to draw on the board and Z = 0.030 to travel in the air."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "waypoints_xyz": {
                        "type": "array",
                        "description": "List of [X, Y, Z] coordinates in meters.",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                    "color": {
                        "type": "string",
                        "enum": list(AVAILABLE_MARKER_COLORS),
                        "description": "Optional marker color ('black', 'blue', or 'red') to select before drawing.",
                    },
                },
                "required": ["waypoints_xyz"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute custom Python 3 code in a math sandbox with pre-imported 'np' (numpy), 'math', "
                "'WORKSPACE_SPEC', 'select_marker(color)', 'draw_primitives(strokes)', and 'draw_waypoints(waypoints_xyz, color=...)'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_canvas",
            "description": (
                "Retract the robot arm out of the camera view and capture the current 2D whiteboard "
                "image (with metric X/Y coordinate rulers) and 3D overhead camera view."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_drawing",
            "description": "Complete the drawing episode, park the robot arm at home, and trigger final scoring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the completed pelican riding a bicycle drawing.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


@dataclass
class ToolExecutionResult:
    """Output from executing a single tool call in WhiteboardToolBelt."""

    tool_name: str
    response_json: str
    images_base64_png: list[str] = field(default_factory=list)
    is_finish: bool = False


def encode_rgb_to_base64_png(rgb_array: np.ndarray, max_width: int = 800) -> str:
    """Encode a uint8 RGB numpy array into a base64 PNG string."""
    pil_img = Image.fromarray(rgb_array)
    if pil_img.width > max_width:
        ratio = float(max_width) / float(pil_img.width)
        new_h = max(1, round(pil_img.height * ratio))
        pil_img = pil_img.resize((max_width, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class WhiteboardToolBelt:
    """Stateful tool dispatcher that binds an SO101WhiteboardEnv to the agent tools."""

    def __init__(self, env: SO101WhiteboardEnv) -> None:
        self.env = env
        self.sandbox = RestrictedPythonSandbox(
            draw_primitives_fn=self._exec_draw_primitives_internal,
            draw_waypoints_fn=self.env.execute_cartesian_path,
            select_marker_fn=self.env.select_marker,
        )

        self.spatial_hallucinations: int = 0
        self.code_hallucinations: int = 0
        self.total_tool_calls: int = 0
        self.tool_call_counts: dict[str, int] = {
            "get_workspace_spec": 0,
            "select_marker": 0,
            "draw_primitives": 0,
            "draw_waypoints": 0,
            "run_python": 0,
            "inspect_canvas": 0,
            "finish_drawing": 0,
        }
        self.hallucination_log: list[str] = []
        self.finished: bool = False
        self.finish_summary: str = ""

    def _exec_draw_primitives_internal(
        self,
        strokes: list[dict[str, Any]],
        default_color: str | None = None,
    ) -> dict[str, Any]:
        """Group 2D primitives by marker color, swap markers at the 3-slot dock, and execute."""
        if not isinstance(strokes, list) or len(strokes) == 0:
            raise ValueError("strokes must be a non-empty list of primitive dictionaries.")

        if default_color is not None:
            self.env.select_marker(default_color)

        # Group contiguous strokes by color so the robot swaps markers at the dock between color batches
        groups: list[tuple[str | None, list[dict[str, Any]]]] = []
        for s in strokes:
            if not isinstance(s, dict):
                raise TypeError(f"Each stroke must be a dictionary, got {type(s).__name__}")
            s_color = s.get("color")
            if s_color is not None:
                s_color = str(s_color).strip().lower()
            if not groups or groups[-1][0] != s_color:
                groups.append((s_color, [s]))
            else:
                groups[-1][1].append(s)

        total_waypoints = 0
        total_interp = 0
        total_ink_added = 0
        max_ik_err_mm = 0.0
        total_spatial = 0
        combined_warnings: list[str] = []
        final_tip = [0.0, 0.0, 0.0]

        for group_color, group_strokes in groups:
            if group_color is not None:
                self.env.select_marker(group_color)
            waypoints_xyz, compile_warnings, compile_hallucinations = compile_primitives_to_waypoints(
                group_strokes
            )
            path_stats = self.env.execute_cartesian_path(waypoints_xyz)
            total_waypoints += int(path_stats["waypoints_commanded"])
            total_interp += int(path_stats["interpolated_steps"])
            total_ink_added += int(path_stats["ink_pixels_added"])
            max_ik_err_mm = max(max_ik_err_mm, float(path_stats["max_ik_error_mm"]))
            total_spatial += compile_hallucinations + int(path_stats["spatial_hallucinations"])
            combined_warnings.extend(compile_warnings)
            combined_warnings.extend(path_stats["warnings"])
            final_tip = list(path_stats["final_pen_tip_xyz"])

        obs_fast = self.env.get_obs()
        return {
            "strokes_executed": len(strokes),
            "waypoints_commanded": total_waypoints,
            "interpolated_steps": total_interp,
            "ink_pixels_added": total_ink_added,
            "total_ink_pixels": int(obs_fast["ink_pixels_drawn"]),
            "active_marker_color": obs_fast.get("active_marker_color", "black"),
            "colors_used": obs_fast.get("colors_used", []),
            "marker_swaps": int(obs_fast.get("marker_swaps", 0)),
            "max_ik_error_mm": round(max_ik_err_mm, 3),
            "final_pen_tip_xyz": final_tip,
            "spatial_hallucinations": total_spatial,
            "warnings": combined_warnings,
        }

    def dispatch(self, tool_name: str, raw_arguments: str | dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool call by name and record hallucination and usage metrics."""
        self.total_tool_calls += 1
        if tool_name in self.tool_call_counts:
            self.tool_call_counts[tool_name] += 1
        else:
            self.tool_call_counts[tool_name] = 1

        # Parse JSON arguments
        if isinstance(raw_arguments, str):
            raw_str = raw_arguments.strip()
            if not raw_str:
                args: dict[str, Any] = {}
            else:
                try:
                    args = json.loads(raw_str)
                except json.JSONDecodeError as exc:
                    self.code_hallucinations += 1
                    msg = f"Malformed JSON arguments for tool '{tool_name}': {exc}"
                    self.hallucination_log.append(msg)
                    return ToolExecutionResult(
                        tool_name=tool_name,
                        response_json=json.dumps({"status": "error", "error": msg}),
                    )
        elif isinstance(raw_arguments, dict):
            args = raw_arguments
        else:
            self.code_hallucinations += 1
            msg = f"Invalid argument container type for tool '{tool_name}'."
            self.hallucination_log.append(msg)
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps({"status": "error", "error": msg}),
            )

        if tool_name == "get_workspace_spec":
            obs = self.env.get_obs()
            payload = dict(WORKSPACE_SPEC)
            payload["current_pen_tip_xyz_m"] = [round(float(v), 4) for v in obs["pen_tip_xyz"]]
            payload["active_marker_color"] = obs.get("active_marker_color", "black")
            payload["colors_used"] = obs.get("colors_used", [])
            payload["marker_swaps"] = int(obs.get("marker_swaps", 0))
            payload["total_ink_pixels_drawn"] = int(obs["ink_pixels_drawn"])
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps(payload),
            )

        if tool_name == "select_marker":
            color = args.get("color", "")
            try:
                swap_res = self.env.select_marker(str(color))
            except Exception as exc:  # noqa: BLE001
                self.code_hallucinations += 1
                msg = f"select_marker error: {exc}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps({"status": "ok", **swap_res}),
            )

        if tool_name == "draw_primitives":
            strokes = args.get("strokes")
            batch_color = args.get("color")
            try:
                stats = self._exec_draw_primitives_internal(strokes, default_color=batch_color)
            except Exception as exc:  # noqa: BLE001
                self.code_hallucinations += 1
                msg = f"draw_primitives error: {exc}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )

            if stats["spatial_hallucinations"] > 0:
                self.spatial_hallucinations += int(stats["spatial_hallucinations"])
                for w in stats["warnings"]:
                    self.hallucination_log.append(f"Spatial: {w}")

            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps({"status": "ok", **stats}),
            )

        if tool_name == "draw_waypoints":
            waypoints = args.get("waypoints_xyz")
            wp_color = args.get("color")
            if not isinstance(waypoints, list) or len(waypoints) == 0:
                self.code_hallucinations += 1
                msg = "draw_waypoints requires a non-empty list 'waypoints_xyz' of [X, Y, Z] coordinates."
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )
            try:
                if wp_color is not None:
                    self.env.select_marker(str(wp_color))
                stats = self.env.execute_cartesian_path(waypoints)
            except Exception as exc:  # noqa: BLE001
                self.code_hallucinations += 1
                msg = f"draw_waypoints error: {exc}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )

            if stats["spatial_hallucinations"] > 0:
                self.spatial_hallucinations += int(stats["spatial_hallucinations"])
                for w in stats["warnings"]:
                    self.hallucination_log.append(f"Spatial: {w}")

            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps({"status": "ok", **stats}),
            )

        if tool_name == "run_python":
            code = args.get("code", "")
            if not isinstance(code, str) or not code.strip():
                self.code_hallucinations += 1
                msg = "run_python requires a non-empty 'code' string."
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )

            sb_res = self.sandbox.run(code)
            self.spatial_hallucinations += sb_res.spatial_hallucinations
            self.code_hallucinations += sb_res.code_hallucinations
            for w in sb_res.warnings:
                self.hallucination_log.append(f"Spatial (run_python): {w}")
            if sb_res.error_message:
                self.hallucination_log.append(f"Code (run_python): {sb_res.error_message}")

            obs_after = self.env.get_obs()
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps(
                    {
                        "status": "ok" if sb_res.success else "error",
                        "stdout": sb_res.stdout,
                        "error": sb_res.error_message,
                        "strokes_executed": sb_res.strokes_executed,
                        "waypoints_executed": sb_res.waypoints_executed,
                        "ink_pixels_added": sb_res.ink_pixels_added,
                        "active_marker_color": obs_after.get("active_marker_color", "black"),
                        "colors_used": obs_after.get("colors_used", []),
                        "marker_swaps": int(obs_after.get("marker_swaps", 0)),
                        "spatial_hallucinations": sb_res.spatial_hallucinations,
                        "warnings": sb_res.warnings,
                    }
                ),
            )

        if tool_name == "inspect_canvas":
            self.env.move_to_home()
            obs = self.env.get_obs()
            b64_canvas = encode_rgb_to_base64_png(obs["canvas_annotated_rgb"], max_width=800)
            b64_overhead = encode_rgb_to_base64_png(obs["rgb_overhead_cam"], max_width=640)
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps(
                    {
                        "status": "ok",
                        "total_ink_pixels": int(obs["ink_pixels_drawn"]),
                        "active_marker_color": obs.get("active_marker_color", "black"),
                        "colors_used": obs.get("colors_used", []),
                        "marker_swaps": int(obs.get("marker_swaps", 0)),
                        "message": (
                            "Attached 2 images: (1) Annotated 2D Whiteboard Canvas with metric X/Y rulers, "
                            "and (2) 3D Overhead Camera view."
                        ),
                    }
                ),
                images_base64_png=[b64_canvas, b64_overhead],
            )

        if tool_name == "finish_drawing":
            self.env.move_to_home()
            self.finished = True
            self.finish_summary = str(args.get("summary", "Completed pelican riding a bicycle."))
            obs = self.env.get_obs()
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps(
                    {
                        "status": "finished",
                        "summary": self.finish_summary,
                        "total_ink_pixels": int(obs["ink_pixels_drawn"]),
                        "colors_used": obs.get("colors_used", []),
                        "marker_swaps": int(obs.get("marker_swaps", 0)),
                    }
                ),
                is_finish=True,
            )

        # Unknown tool name hallucination
        self.code_hallucinations += 1
        msg = (
            f"Unknown tool '{tool_name}'. Available tools: "
            "get_workspace_spec, select_marker, draw_primitives, draw_waypoints, run_python, inspect_canvas, finish_drawing."
        )
        self.hallucination_log.append(msg)
        return ToolExecutionResult(
            tool_name=tool_name,
            response_json=json.dumps({"status": "error", "error": msg}),
        )
