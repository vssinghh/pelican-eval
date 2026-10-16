"""Episode Dataset Bundle Recorder and Leaderboard Updater for PelicanBench."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from envs.base import SO101WhiteboardEnv
from evaluation.metrics import (
    EpisodeMetrics,
    build_episode_metrics,
    format_markdown_summary_table,
)
from evaluation.visual_judge import VisualRubricBreakdown
from harness.agent_loop import AgentRunResult


def slugify_model_name(model_id: str) -> str:
    """Convert a provider/model string into a filesystem-safe directory slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", model_id.strip().lower()).strip("_")
    return cleaned or "model"


def save_optimized_gif(
    frames: list[np.ndarray],
    gif_path: Path,
    duration_ms: int = 90,
) -> None:
    """Save RGB numpy frames as a compact, single-palette animated GIF without dither noise."""
    pil_rgb = [Image.fromarray(f) for f in frames]
    base_pal = pil_rgb[-1].quantize(
        colors=256,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )
    pal_frames = [f.quantize(palette=base_pal, dither=Image.Dither.NONE) for f in pil_rgb]
    pal_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


class EpisodeRecorder:
    """Saves all episode artifacts, 500 Hz trajectories, and summary tables."""

    def __init__(self, output_root: str | Path = "results/sim_benchmark") -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def save_episode(
        self,
        env: SO101WhiteboardEnv,
        run_res: AgentRunResult,
        visual_res: VisualRubricBreakdown,
        seed: int = 0,
    ) -> tuple[EpisodeMetrics, Path]:
        """Write metrics.json, transcript.json, overhead_final.png, timelapse.gif, and trajectory_500hz.npz."""
        slug = slugify_model_name(run_res.model)
        ep_dir = self.output_root / f"{slug}_seed{seed}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        obs = env.get_obs()
        overhead_path = ep_dir / "overhead_final.png"
        gif_path = ep_dir / "timelapse.gif"
        npz_path = ep_dir / "trajectory_500hz.npz"

        Image.fromarray(obs["rgb_overhead_cam"]).save(overhead_path)

        timelapse_frames = getattr(env, "timelapse_frames", [])
        if timelapse_frames:
            save_optimized_gif(timelapse_frames, gif_path, duration_ms=90)
        else:
            Image.fromarray(obs["rgb_overhead_cam"]).save(gif_path)

        if hasattr(env, "get_trajectory_arrays"):
            traj = env.get_trajectory_arrays()
            np.savez_compressed(
                npz_path,
                qpos=traj["qpos"],
                qvel=traj["qvel"],
                pen_tip_xyz=traj["pen_tip_xyz"],
                ink_contact=traj["ink_contact"],
            )

        rel_overhead = f"{ep_dir.name}/overhead_final.png"
        rel_gif = f"{ep_dir.name}/timelapse.gif"

        metrics = build_episode_metrics(
            run_res=run_res,
            visual_res=visual_res,
            seed=seed,
            ink_pixels_drawn=int(obs["ink_pixels_drawn"]),
            canvas_rel_path=rel_overhead,
            timelapse_rel_path=rel_gif,
        )

        with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        with open(ep_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": run_res.model,
                    "domain": run_res.domain,
                    "turns_used": run_res.turns_used,
                    "finished_cleanly": run_res.finished_cleanly,
                    "finish_summary": run_res.finish_summary,
                    "hallucination_log": run_res.hallucination_log,
                    "messages": run_res.transcript,
                },
                f,
                indent=2,
            )

        return metrics, ep_dir

    def write_summary_report(self, metrics_list: list[EpisodeMetrics]) -> Path:
        """Write SUMMARY.md and summary_metrics.json aggregating all runs inside output_root."""
        by_key: dict[tuple[str, int], EpisodeMetrics] = {}
        for mf in sorted(self.output_root.glob("*_seed*/metrics.json")):
            try:
                raw = json.loads(mf.read_text(encoding="utf-8"))
                em = EpisodeMetrics(**raw)
                by_key[(em.model, em.seed)] = em
            except (OSError, ValueError, TypeError):
                pass
        for m in metrics_list:
            by_key[(m.model, m.seed)] = m

        merged = list(by_key.values())
        summary_json_path = self.output_root / "summary_metrics.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in merged], f, indent=2)

        table_md = format_markdown_summary_table(merged)
        summary_md_path = self.output_root / "SUMMARY.md"
        content = (
            "# PelicanBench Evaluation Summary\n\n"
            f"{table_md}\n"
        )
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(content)
        return summary_md_path
