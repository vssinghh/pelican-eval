"""Tool definitions, JSON schemas, and hallucination tracking for PelicanBench."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from envs.base import (
    AVAILABLE_MARKER_COLORS,
    MARKER_COLORS_RGB,
    WORKSPACE_SPEC,
    SO101WhiteboardEnv,
)
from envs.trajectory import compile_primitives_to_waypoints
from harness.python_sandbox import RestrictedPythonSandbox

OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_workspace_detail",
            "description": (
                "Retract the robot arm out of the camera view and return live RGB camera images "
                "(overhead_cam and third_person_cam) along with the calibrated whiteboard corner "
                "coordinates, pen contact height z_draw, safe travel height z_up, and current pen_tip [X, Y, Z]."
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
                            "items": {"number": "float"} if False else {"type": "number"},
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
                "'WORKSPACE_SPEC', 'select_marker(color)', 'draw_primitives(strokes, color=...)', and 'draw_waypoints(waypoints_xyz, color=...)'."
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

REAL_SKILL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "so101_cli",
            "description": (
                "Execute a physical SO-ARM101 closed-loop CLI command from scripts/so101_cli.py "
                "(as taught in the so101-pickup-marker and so101-calibrate-paper-contact skills) "
                "and return joint ticks, FK pinch_site_xyz_m, Reg 56 position, Reg 60 load, grasp_state, "
                "and the 3 live annotated camera images (Camera 0 overhead, Camera 1 side, Camera 2 wrist). "
                "Supported commands in args: "
                "['status', '<prefix>'], "
                "['open-observe', '<prefix>'], "
                "['move-polar', '<pan_tick>', '<reach_m>', '<z_m>', '<pitch_rad>', '<gripper_tick>', '<prefix>'], "
                "['nudge-polar', '<d_pan_tick>', '<d_reach_m>', '<d_z_m>', '<d_pitch_rad>', '<prefix>'], "
                "['move-ik', '<x>', '<y>', '<z>', '<pitch_rad>', '<gripper_tick>', '<prefix>'], "
                "['gripper', '<open|clamp|tick>', '<prefix>'], "
                "['lift', '<dz_m>', '<prefix>'], "
                "['place', '<pan_tick>', '<reach_m>', '<z_insert_m>', '<prefix>'], "
                "['draw-strokes', '<json_strokes_string>', '<prefix>'], "
                "['safe-park', '<release|keep>', '<prefix>']."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Command-line arguments to pass to scripts/so101_cli.py, e.g. "
                            "['move-polar', '2420', '0.202', '0.018', '0.95', '2500', 'step2_align'] or "
                            "['draw-strokes', '[{\"points\": [[0.25, 0.30], [0.75, 0.30]]}]', 'pelican_strokes']."
                        ),
                    }
                },
                "required": ["args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calibration",
            "description": (
                "Save calibrated paper contact plane parameters to config/calibration.json after running "
                "the so101-calibrate-paper-contact skill."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plane_coeffs_abc": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Linear plane coefficients [a, b, c] for z_board(x, y) = a*x + b*y + c.",
                    },
                    "draw_z_pen_down": {
                        "type": "number",
                        "description": "Minimum pen-down Z height in meters (e.g. z_far - 0.0005).",
                    },
                    "draw_z_pen_up": {
                        "type": "number",
                        "description": "Safe pen-up hover Z height in meters (e.g. 0.088).",
                    },
                    "target_pitch": {
                        "type": "number",
                        "description": "Calibrated vertical wrist pitch in radians (e.g. 1.65).",
                    },
                    "board_x_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional [x_min, x_max] paper range in meters.",
                    },
                    "board_y_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional [y_min, y_max] paper range in meters.",
                    },
                },
                "required": ["plane_coeffs_abc"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_marker_swap",
            "description": (
                "Release/drop the currently held marker via 'safe-park release', pause the benchmark timer, "
                "and ask the human operator to place the requested marker color ('black', 'blue', or 'red', "
                "uncapped, tip down) into the front low bin. After this returns, use the so101-pickup-marker "
                "and so101-calibrate-paper-contact skills to pick up and calibrate the new marker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": list(AVAILABLE_MARKER_COLORS),
                        "description": "Marker color to ask the human operator to place in the bin: 'black', 'blue', or 'red'.",
                    }
                },
                "required": ["color"],
            },
        },
    },
]


def get_tool_schemas(domain: str = "sim") -> list[dict[str, Any]]:
    """Return tool schemas for the given execution domain ('sim' or 'real')."""
    if domain == "real":
        real_base_schemas = [
            s for s in OPENAI_TOOL_SCHEMAS if s.get("function", {}).get("name") != "select_marker"
        ]
        return real_base_schemas + REAL_SKILL_TOOL_SCHEMAS
    return OPENAI_TOOL_SCHEMAS


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


def encode_image_path_to_base64_png(img_path: Path, max_width: int = 800) -> str:
    """Load an image file from disk and encode it as a resized base64 PNG string."""
    with Image.open(img_path) as img:
        rgb_img = img.convert("RGB")
        if rgb_img.width > max_width:
            ratio = float(max_width) / float(rgb_img.width)
            new_h = max(1, round(rgb_img.height * ratio))
            rgb_img = rgb_img.resize((max_width, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb_img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


class WhiteboardToolBelt:
    """Stateful tool dispatcher that binds an SO101WhiteboardEnv to the agent tools."""

    def __init__(self, env: SO101WhiteboardEnv) -> None:
        self.env = env
        self.sandbox = RestrictedPythonSandbox(
            draw_primitives_fn=self._exec_draw_primitives_internal,
            draw_waypoints_fn=self.env.execute_cartesian_path,
            select_marker_fn=self._safe_select_marker,
        )

        self.spatial_hallucinations: int = 0
        self.code_hallucinations: int = 0
        self.total_tool_calls: int = 0
        self.paused_human_time_s: float = 0.0
        self.tool_call_counts: dict[str, int] = {
            "get_workspace_detail": 0,
            "select_marker": 0,
            "draw_primitives": 0,
            "draw_waypoints": 0,
            "run_python": 0,
            "finish_drawing": 0,
        }
        is_real_skill_env = hasattr(self.env, "suspend_connections")
        self._awaiting_pickup_verify: bool = is_real_skill_env
        self.skill_stages: dict[str, Any] = {
            "pickup_verified": not is_real_skill_env,
            "pickups_verified": 0 if is_real_skill_env else 1,
            "calibration_saved": not is_real_skill_env,
            "calibrations_saved": 0 if is_real_skill_env else 1,
            "drawing_executed": False,
            "drawing_strokes_executed": 0,
            "swaps_requested": 0,
        }
        self.hallucination_log: list[str] = []
        self.finished: bool = False
        self.finish_summary: str = ""

    def _safe_select_marker(self, color: str) -> dict[str, Any]:
        """Guard physical single-marker mode so models must use request_marker_swap to switch colors."""
        target_color = str(color).strip().lower()
        if target_color not in AVAILABLE_MARKER_COLORS:
            raise ValueError(
                f"Invalid marker color '{color}'. Available colors: {list(AVAILABLE_MARKER_COLORS)}"
            )
        if getattr(self.env, "single_marker_mode", False):
            cur_color = str(getattr(self.env, "current_marker_color", "black")).strip().lower()
            if target_color == cur_color:
                return {
                    "swapped": False,
                    "previous_color": cur_color,
                    "active_color": cur_color,
                    "marker_swaps": int(getattr(self.env, "marker_swaps", 0)),
                    "colors_used": sorted(getattr(self.env, "colors_used", set())),
                }
            raise RuntimeError(
                f"Physical robot is currently holding '{cur_color}' in single-marker bin mode. "
                f"Call request_marker_swap(color='{target_color}') first, then run "
                f"so101-pickup-marker and so101-calibrate-paper-contact before drawing in '{target_color}'."
            )
        swap_res = self.env.select_marker(target_color)
        if swap_res.get("swapped"):
            self.skill_stages["swaps_requested"] += 1
        return swap_res

    def _run_so101_cli_subprocess(self, cli_args: list[str]) -> tuple[dict[str, Any], list[str]]:
        """Execute scripts/so101_cli.py with cli_args and return (json_payload, base64_grid_images)."""
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "scripts" / "so101_cli.py"

        if getattr(self.env, "mock_hardware", False):
            obs = self.env.get_obs()
            b64_0 = encode_rgb_to_base64_png(obs["rgb_overhead_cam"], max_width=640)
            b64_1 = encode_rgb_to_base64_png(obs["rgb_third_person_cam"], max_width=640)
            mock_payload: dict[str, Any] = {
                "status": "ok",
                "mock_hardware": True,
                "command_args": cli_args,
                "ticks": [2048, 1600, 2250, 2850, 973, 2125],
                "pinch_site_xyz_m": [0.206, -0.025, 0.088],
                "gripper_present_position_reg56": 2125,
                "gripper_present_load_reg60": 380,
                "grasp_state": "BARREL_HELD_IN_FLAT_PAD",
            }
            if cli_args and cli_args[0] == "draw-strokes" and len(cli_args) >= 2:
                self._mirror_normalized_strokes_to_canvas(cli_args[1])
            return mock_payload, [b64_0, b64_1]

        if hasattr(self.env, "suspend_connections"):
            self.env.suspend_connections()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path), *[str(a) for a in cli_args]],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        finally:
            if hasattr(self.env, "resume_connections"):
                self.env.resume_connections()

        stdout_str = (proc.stdout or "").strip()
        if proc.returncode != 0:
            err_detail = (proc.stderr or stdout_str or f"exit code {proc.returncode}").strip()
            raise RuntimeError(f"so101_cli failed: {err_detail}")

        json_start = stdout_str.find("{")
        json_end = stdout_str.rfind("}")
        if json_start < 0 or json_end < json_start:
            raise RuntimeError(f"so101_cli returned non-JSON output: {stdout_str}")

        payload: dict[str, Any] = json.loads(stdout_str[json_start : json_end + 1])

        if cli_args and cli_args[0] == "draw-strokes" and len(cli_args) >= 2:
            self._mirror_normalized_strokes_to_canvas(cli_args[1])

        images_b64: list[str] = []
        saved_paths = payload.get("saved_images", [])
        grid_paths = [Path(p) for p in saved_paths if str(p).endswith("_grid.png") and Path(p).exists()]
        raw_paths = [Path(p) for p in saved_paths if not str(p).endswith("_grid.png") and Path(p).exists()]

        for gp in grid_paths:
            images_b64.append(encode_image_path_to_base64_png(gp, max_width=800))

        # Append a timelapse frame if raw cam_0 and cam_1 were saved
        cam0_raw = next((p for p in raw_paths if p.name.endswith("_cam_0.png")), None)
        cam1_raw = next((p for p in raw_paths if p.name.endswith("_cam_1.png")), None)
        if cam0_raw is not None and cam1_raw is not None and hasattr(self.env, "timelapse_frames"):
            try:
                with Image.open(cam1_raw) as im1, Image.open(cam0_raw) as im0:
                    tp = im1.convert("RGB").resize((400, 300))
                    ov = im0.convert("RGB").resize((400, 300))
                    combined = Image.new("RGB", (800, 300))
                    combined.paste(tp, (0, 0))
                    combined.paste(ov, (400, 0))
                    self.env.timelapse_frames.append(np.asarray(combined, dtype=np.uint8))
            except OSError:
                pass

        return payload, images_b64

    def _mirror_normalized_strokes_to_canvas(self, raw_strokes_arg: str) -> None:
        """Mirror normalized (u, v) strokes from draw-strokes onto env._canvas_draw for scoring."""
        try:
            if Path(raw_strokes_arg).is_file():
                data = json.loads(Path(raw_strokes_arg).read_text(encoding="utf-8"))
            else:
                data = json.loads(raw_strokes_arg)
        except Exception:  # noqa: BLE001
            return

        if not isinstance(data, list) or not hasattr(self.env, "_canvas_draw"):
            return

        color_name = getattr(self.env, "current_marker_color", "black")
        rgb = MARKER_COLORS_RGB.get(color_name, (28, 32, 42))
        w = int(getattr(self.env, "canvas_w", 1000))
        h = int(getattr(self.env, "canvas_h", 500))

        mirrored_strokes = 0
        for stroke in data:
            if not isinstance(stroke, dict):
                continue
            pts = stroke.get("points", [])
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            mirrored_strokes += 1
            if hasattr(self.env, "colors_used"):
                self.env.colors_used.add(color_name)
            for idx in range(1, len(pts)):
                u0, v0 = float(pts[idx - 1][0]), float(pts[idx - 1][1])
                u1, v1 = float(pts[idx][0]), float(pts[idx][1])
                c0 = int(np.clip(u0, 0.0, 1.0) * (w - 1))
                r0 = int((1.0 - np.clip(v0, 0.0, 1.0)) * (h - 1))
                c1 = int(np.clip(u1, 0.0, 1.0) * (w - 1))
                r1 = int((1.0 - np.clip(v1, 0.0, 1.0)) * (h - 1))
                self.env._canvas_draw.line([(c0, r0), (c1, r1)], fill=rgb, width=4)
                if hasattr(self.env, "ink_pixels_drawn"):
                    self.env.ink_pixels_drawn += max(1, int(np.hypot(r1 - r0, c1 - c0)))

        if mirrored_strokes > 0:
            self.skill_stages["drawing_strokes_executed"] += mirrored_strokes
            if int(getattr(self.env, "ink_pixels_drawn", 0)) > 0:
                self.skill_stages["drawing_executed"] = True

    def _exec_draw_primitives_internal(
        self,
        strokes: list[dict[str, Any]],
        default_color: str | None = None,
    ) -> dict[str, Any]:
        """Group 2D primitives by marker color, swap markers at the 3-slot dock, and execute."""
        if not isinstance(strokes, list) or len(strokes) == 0:
            raise ValueError("strokes must be a non-empty list of primitive dictionaries.")

        if default_color is not None:
            self._safe_select_marker(default_color)

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
                self._safe_select_marker(group_color)
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

        if total_ink_added > 0:
            self.skill_stages["drawing_strokes_executed"] += len(strokes)
            self.skill_stages["drawing_executed"] = True

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

        if tool_name == "so101_cli":
            cli_args = args.get("args", [])
            if not isinstance(cli_args, list) or len(cli_args) == 0:
                self.code_hallucinations += 1
                msg = "so101_cli requires a non-empty list 'args' of strings."
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )
            str_args = [str(a) for a in cli_args]
            try:
                payload, images_b64 = self._run_so101_cli_subprocess(str_args)
            except Exception as exc:  # noqa: BLE001
                self.code_hallucinations += 1
                msg = f"so101_cli error: {exc}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )

            subcmd = str_args[0].strip().lower()
            arg1 = str_args[1].strip().lower() if len(str_args) >= 2 else ""
            if (
                subcmd == "open-observe"
                or (subcmd == "gripper" and arg1 == "open")
                or (subcmd == "safe-park" and arg1 == "release")
            ):
                self._awaiting_pickup_verify = True
            elif (
                payload.get("grasp_state") == "BARREL_HELD_IN_FLAT_PAD"
                and self._awaiting_pickup_verify
            ):
                self.skill_stages["pickup_verified"] = True
                self.skill_stages["pickups_verified"] += 1
                self._awaiting_pickup_verify = False

            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps({"status": "ok", **payload}),
                images_base64_png=images_b64,
            )

        if tool_name == "update_calibration":
            coeffs = args.get("plane_coeffs_abc")
            if not isinstance(coeffs, list) or len(coeffs) != 3:
                self.code_hallucinations += 1
                msg = "update_calibration requires 'plane_coeffs_abc' as a list of 3 floats [a, b, c]."
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )
            default_cal = Path(__file__).resolve().parent.parent / "config" / "calibration.json"
            cal_path = Path(getattr(self.env, "calibration_path", default_cal))
            try:
                cal_data = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
                cal_data["plane_coeffs_abc"] = [float(coeffs[0]), float(coeffs[1]), float(coeffs[2])]
                for opt_key in ("draw_z_pen_down", "draw_z_pen_up", "target_pitch"):
                    if opt_key in args and args[opt_key] is not None:
                        cal_data[opt_key] = float(args[opt_key])
                for range_key in ("board_x_range", "board_y_range"):
                    if range_key in args and isinstance(args[range_key], list) and len(args[range_key]) == 2:
                        cal_data[range_key] = [float(args[range_key][0]), float(args[range_key][1])]
                cal_path.write_text(json.dumps(cal_data, indent=2) + "\n", encoding="utf-8")
                if hasattr(self.env, "reload_calibration"):
                    self.env.reload_calibration()
                self.skill_stages["calibration_saved"] = True
                self.skill_stages["calibrations_saved"] += 1
            except Exception as exc:  # noqa: BLE001
                self.code_hallucinations += 1
                msg = f"update_calibration error: {exc}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps({"status": "ok", "updated_calibration": cal_data}),
            )

        if tool_name == "request_marker_swap":
            target_color = str(args.get("color", "")).strip().lower()
            if target_color not in AVAILABLE_MARKER_COLORS:
                self.code_hallucinations += 1
                msg = f"Invalid marker color '{target_color}'. Available colors: {list(AVAILABLE_MARKER_COLORS)}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )
            try:
                self._run_so101_cli_subprocess(["safe-park", "release", f"swap_release_{target_color}"])
                if not getattr(self.env, "mock_hardware", False):
                    t_pause_start = time.perf_counter()
                    print(
                        f"\n[MARKER SWAP] Arm is parked and gripper is open. "
                        f"Please place the '{target_color.upper()}' marker (uncapped, tip down) "
                        f"into the front low bin and press Enter..."
                    )
                    try:
                        input()
                    except EOFError:
                        pass
                    self.paused_human_time_s += time.perf_counter() - t_pause_start
                payload, images_b64 = self._run_so101_cli_subprocess(
                    ["status", f"swap_ready_{target_color}"]
                )
                if hasattr(self.env, "current_marker_color"):
                    self.env.current_marker_color = target_color
                if hasattr(self.env, "marker_swaps"):
                    self.env.marker_swaps += 1
                self.skill_stages["swaps_requested"] += 1
                self._awaiting_pickup_verify = True
            except Exception as exc:  # noqa: BLE001
                self.code_hallucinations += 1
                msg = f"request_marker_swap error: {exc}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps(
                    {
                        "status": "ok",
                        "requested_color": target_color,
                        "message": (
                            f"Human operator placed the '{target_color}' marker in the front low bin. "
                            "Now run so101-pickup-marker followed by so101-calibrate-paper-contact."
                        ),
                        **payload,
                    }
                ),
                images_base64_png=images_b64,
            )

        if tool_name == "get_workspace_detail":
            self.env.move_to_home()
            obs = self.env.get_obs()
            b64_overhead = encode_rgb_to_base64_png(obs["rgb_overhead_cam"], max_width=640)
            b64_third_person = encode_rgb_to_base64_png(obs["rgb_third_person_cam"], max_width=640)
            payload = dict(WORKSPACE_SPEC)
            payload["status"] = "ok"
            payload["current_pen_tip_xyz_m"] = [round(float(v), 4) for v in obs["pen_tip_xyz"]]
            payload["active_marker_color"] = obs.get("active_marker_color", "black")
            payload["colors_used"] = obs.get("colors_used", [])
            payload["marker_swaps"] = int(obs.get("marker_swaps", 0))
            payload["total_ink_pixels_drawn"] = int(obs["ink_pixels_drawn"])
            payload["message"] = (
                "Attached 2 live camera images: (1) overhead_cam (top-down RGB view of the robot, "
                "3-slot marker stand, and whiteboard), and (2) third_person_cam (angled 3D RGB view)."
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                response_json=json.dumps(payload),
                images_base64_png=[b64_overhead, b64_third_person],
            )

        if tool_name == "select_marker":
            color = args.get("color", "")
            try:
                swap_res = self._safe_select_marker(str(color))
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
                    self._safe_select_marker(str(wp_color))
                stats = self.env.execute_cartesian_path(waypoints)
            except Exception as exc:  # noqa: BLE001
                self.code_hallucinations += 1
                msg = f"draw_waypoints error: {exc}"
                self.hallucination_log.append(msg)
                return ToolExecutionResult(
                    tool_name=tool_name,
                    response_json=json.dumps({"status": "error", "error": msg}),
                )

            if int(stats.get("ink_pixels_added", 0)) > 0:
                self.skill_stages["drawing_strokes_executed"] += 1
                self.skill_stages["drawing_executed"] = True

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
            if sb_res.waypoints_executed > 0 and sb_res.strokes_executed == 0 and sb_res.ink_pixels_added > 0:
                self.skill_stages["drawing_strokes_executed"] += 1
                self.skill_stages["drawing_executed"] = True

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
            "get_workspace_detail, select_marker, draw_primitives, draw_waypoints, run_python, "
            "so101_cli, update_calibration, request_marker_swap, finish_drawing."
        )
        self.hallucination_log.append(msg)
        return ToolExecutionResult(
            tool_name=tool_name,
            response_json=json.dumps({"status": "error", "error": msg}),
        )
