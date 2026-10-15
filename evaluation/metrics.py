"""5-Metric Aggregator and Leaderboard Formatter for PelicanBench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evaluation.visual_judge import VisualRubricBreakdown
from harness.agent_loop import AgentRunResult


@dataclass
class EpisodeMetrics:
    """Consolidated 5-metric scorecard for a single PelicanBench evaluation episode."""

    model: str
    provider: str
    domain: str
    seed: int
    total_hallucinations: int
    spatial_hallucinations: int
    code_hallucinations: int
    total_tool_calls: int
    tool_call_counts: dict[str, int]
    e2e_time_s: float
    cost_usd: float
    visual_score: float
    visual_breakdown: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    ink_pixels_drawn: int
    finished_cleanly: bool
    canvas_path: str
    timelapse_path: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics dataclass to JSON-serializable dictionary."""
        return asdict(self)


def infer_provider_name(model_id: str) -> str:
    """Extract clean human-readable provider name from a LiteLLM model string."""
    prefix = model_id.split("/")[0].lower() if "/" in model_id else model_id.lower()
    mapping = {
        "anthropic": "Anthropic",
        "gemini": "Google",
        "google": "Google",
        "vertex_ai": "Google",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
        "deepseek": "DeepSeek",
        "reference": "Reference Baseline",
        "mock": "Mock Baseline",
    }
    return mapping.get(prefix, prefix.capitalize())


def build_episode_metrics(
    run_res: AgentRunResult,
    visual_res: VisualRubricBreakdown,
    seed: int,
    ink_pixels_drawn: int,
    canvas_rel_path: str,
    timelapse_rel_path: str,
) -> EpisodeMetrics:
    """Combine agent loop telemetry and visual rubric score into EpisodeMetrics."""
    return EpisodeMetrics(
        model=run_res.model,
        provider=infer_provider_name(run_res.model),
        domain=run_res.domain,
        seed=int(seed),
        total_hallucinations=run_res.total_hallucinations,
        spatial_hallucinations=run_res.spatial_hallucinations,
        code_hallucinations=run_res.code_hallucinations,
        total_tool_calls=run_res.total_tool_calls,
        tool_call_counts=run_res.tool_call_counts,
        e2e_time_s=run_res.e2e_time_s,
        cost_usd=run_res.cost_usd,
        visual_score=visual_res.total_score,
        visual_breakdown=visual_res.to_dict(),
        prompt_tokens=run_res.prompt_tokens,
        completion_tokens=run_res.completion_tokens,
        ink_pixels_drawn=int(ink_pixels_drawn),
        finished_cleanly=run_res.finished_cleanly,
        canvas_path=canvas_rel_path,
        timelapse_path=timelapse_rel_path,
    )


def format_leaderboard_row(m: EpisodeMetrics) -> str:
    """Format a single Markdown table row matching LEADERBOARD.md."""
    hall_str = f"`{m.total_hallucinations} ({m.spatial_hallucinations} / {m.code_hallucinations})`"
    cost_str = f"`\\${m.cost_usd:.4f}`"
    vis_str = f"**`{m.visual_score:.1f} / 10`**"
    links = f"[Canvas]({m.canvas_path}) / [GIF]({m.timelapse_path})"
    return (
        f"| `{m.model}` | {m.provider} | {hall_str} | `{m.total_tool_calls}` | "
        f"`{m.e2e_time_s:.2f}s` | {cost_str} | {vis_str} | {links} |"
    )


def format_markdown_summary_table(metrics_list: list[EpisodeMetrics]) -> str:
    """Format a full Markdown table for a list of evaluated models."""
    header = (
        "| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | "
        "End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Canvas & Replay |\n"
        "| :- | :- | :-: | :-: | :-: | :-: | :-: | :- |"
    )
    rows = [format_leaderboard_row(m) for m in metrics_list]
    return "\n".join([header, *rows])
