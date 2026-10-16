"""LiteLLM multi-provider agentic conversation loop with end-to-end timing and cost tracking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

from envs.base import SO101WhiteboardEnv
from envs.verify_whiteboard import get_reference_pelican_strokes
from harness.prompts import SYSTEM_PROMPT, TASK_PROMPT
from harness.tools import OPENAI_TOOL_SCHEMAS, WhiteboardToolBelt

# Suppress verbose LiteLLM debug banners
litellm.suppress_debug_info = True


@dataclass
class AgentRunResult:
    """Complete telemetry and transcript from a single model evaluation run."""

    model: str
    domain: str
    spatial_hallucinations: int
    code_hallucinations: int
    total_hallucinations: int
    total_tool_calls: int
    tool_call_counts: dict[str, int]
    e2e_time_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    turns_used: int
    finished_cleanly: bool
    finish_summary: str
    colors_used: list[str] = field(default_factory=list)
    marker_swaps: int = 0
    hallucination_log: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


class PelicanAgentRunner:
    """Executes the multi-turn LiteLLM tool loop for any configured LLM/VLM provider."""

    def __init__(
        self,
        model: str,
        max_turns: int = 10,
        temperature: float = 0.2,
        timeout_s: int = 120,
        env_file: str | Path | None = None,
    ) -> None:
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)
        else:
            repo_root = Path(__file__).resolve().parent.parent
            default_env = repo_root / ".env"
            if default_env.exists():
                load_dotenv(dotenv_path=default_env, override=False)
            else:
                load_dotenv(override=False)

        self.model = model
        self.max_turns = int(max_turns)
        self.temperature = float(temperature)
        self.timeout_s = int(timeout_s)

    def run_episode(self, env: SO101WhiteboardEnv, domain: str = "sim") -> AgentRunResult:
        """Reset the environment and run the model until finish_drawing() or max_turns."""
        env.reset()
        belt = WhiteboardToolBelt(env)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPT},
        ]
        transcript: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TASK_PROMPT},
        ]

        total_cost_usd = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        turns_used = 0

        t_start = time.perf_counter()

        if self.model in ("reference/expert", "mock/expert"):
            result = self._run_reference_expert_loop(belt, transcript, domain, t_start)
            return result

        for turn_idx in range(self.max_turns):
            turns_used = turn_idx + 1
            completion_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": OPENAI_TOOL_SCHEMAS,
                "tool_choice": "auto",
                "timeout": self.timeout_s,
            }
            # Reasoning models and Gemini 3+ models disallow or deprecate custom temperature
            if not any(
                tag in self.model.lower() for tag in ("o1", "o3", "o4", "reasoner", "gemini-3")
            ):
                completion_kwargs["temperature"] = self.temperature

            response = litellm.completion(**completion_kwargs)

            # Track token usage and USD cost
            usage = getattr(response, "usage", None)
            if usage is not None:
                p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                total_prompt_tokens += p_tok
                total_completion_tokens += c_tok
            else:
                p_tok, c_tok = 0, 0

            try:
                call_cost = float(litellm.completion_cost(completion_response=response) or 0.0)
            except Exception:  # noqa: BLE001
                # Fallback estimate ($3/1M input, $15/1M output) if model is not in local price map
                call_cost = (p_tok * 3.0 + c_tok * 15.0) / 1_000_000.0
            total_cost_usd += call_cost

            choice_msg = response.choices[0].message
            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": choice_msg.content or "",
            }
            tool_calls = getattr(choice_msg, "tool_calls", None)

            if tool_calls:
                serialized_tcs: list[dict[str, Any]] = []
                for tc in tool_calls:
                    serialized_tcs.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
                assistant_dict["tool_calls"] = serialized_tcs
                messages.append(assistant_dict)
                transcript.append(assistant_dict)

                pending_vision_images: list[str] = []
                for tc in tool_calls:
                    t_name = tc.function.name
                    t_args = tc.function.arguments
                    exec_res = belt.dispatch(t_name, t_args)

                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": exec_res.response_json,
                    }
                    messages.append(tool_msg)
                    transcript.append(tool_msg)

                    if exec_res.images_base64_png:
                        pending_vision_images.extend(exec_res.images_base64_png)

                if pending_vision_images and not belt.finished:
                    vision_content: list[dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": (
                                "Here are the live camera images from get_workspace_detail(): "
                                "(1) overhead_cam (top-down RGB view of the robot, 3-slot marker stand, and whiteboard), "
                                "and (2) third_person_cam (angled 3D RGB view). "
                                "Draw or refine strokes on the whiteboard, or call finish_drawing(summary) if complete."
                            ),
                        }
                    ]
                    for b64_img in pending_vision_images:
                        vision_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                            }
                        )
                    messages.append({"role": "user", "content": vision_content})
                    transcript.append(
                        {
                            "role": "user",
                            "content": "[Attached 2 live camera PNG images from get_workspace_detail()]",
                        }
                    )

                if belt.finished:
                    break
            else:
                messages.append(assistant_dict)
                transcript.append(assistant_dict)
                # If the model stopped calling tools without calling finish_drawing, nudge once or exit
                if belt.total_tool_calls > 0 and env.get_obs()["ink_pixels_drawn"] > 0:
                    belt.dispatch(
                        "finish_drawing",
                        {"summary": choice_msg.content or "Finished drawing."},
                    )
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not drawn on the whiteboard yet. Please call get_workspace_detail(), "
                            "draw_primitives(), or run_python() to draw the pelican riding a bicycle, "
                            "then call finish_drawing()."
                        ),
                    }
                )

        if not belt.finished:
            env.move_to_home()

        e2e_time_s = time.perf_counter() - t_start
        final_obs = env.get_obs()
        return AgentRunResult(
            model=self.model,
            domain=domain,
            spatial_hallucinations=belt.spatial_hallucinations,
            code_hallucinations=belt.code_hallucinations,
            total_hallucinations=belt.spatial_hallucinations + belt.code_hallucinations,
            total_tool_calls=belt.total_tool_calls,
            tool_call_counts=dict(belt.tool_call_counts),
            e2e_time_s=round(e2e_time_s, 3),
            cost_usd=round(total_cost_usd, 6),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            turns_used=turns_used,
            finished_cleanly=belt.finished,
            finish_summary=belt.finish_summary,
            colors_used=list(final_obs.get("colors_used", [])),
            marker_swaps=int(final_obs.get("marker_swaps", 0)),
            hallucination_log=list(belt.hallucination_log),
            transcript=transcript,
        )

    def _run_reference_expert_loop(
        self,
        belt: WhiteboardToolBelt,
        transcript: list[dict[str, Any]],
        domain: str,
        t_start: float,
    ) -> AgentRunResult:
        """Execute a deterministic 5-call 3-color reference trajectory through the exact tool belt."""
        strokes = get_reference_pelican_strokes()
        calls = [
            ("get_workspace_detail", {}),
            ("draw_primitives", {"strokes": strokes[:14]}),
            ("get_workspace_detail", {}),
            ("draw_primitives", {"strokes": strokes[14:]}),
            (
                "finish_drawing",
                {"summary": "Completed 3-color reference pelican riding a bicycle across 20 strokes."},
            ),
        ]
        for idx, (t_name, t_args) in enumerate(calls):
            res = belt.dispatch(t_name, t_args)
            transcript.append(
                {
                    "role": "assistant",
                    "tool_call": {"id": f"ref_{idx}", "name": t_name, "arguments": t_args},
                }
            )
            transcript.append(
                {
                    "role": "tool",
                    "name": t_name,
                    "content": json.loads(res.response_json),
                }
            )

        e2e_time_s = time.perf_counter() - t_start
        final_obs = belt.env.get_obs()
        return AgentRunResult(
            model=self.model,
            domain=domain,
            spatial_hallucinations=belt.spatial_hallucinations,
            code_hallucinations=belt.code_hallucinations,
            total_hallucinations=belt.spatial_hallucinations + belt.code_hallucinations,
            total_tool_calls=belt.total_tool_calls,
            tool_call_counts=dict(belt.tool_call_counts),
            e2e_time_s=round(e2e_time_s, 3),
            cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            turns_used=len(calls),
            finished_cleanly=belt.finished,
            finish_summary=belt.finish_summary,
            colors_used=list(final_obs.get("colors_used", [])),
            marker_swaps=int(final_obs.get("marker_swaps", 0)),
            hallucination_log=list(belt.hallucination_log),
            transcript=transcript,
        )
