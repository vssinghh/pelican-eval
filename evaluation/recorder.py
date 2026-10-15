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
        """Write metrics.json, transcript.json, PNGs, GIF, and trajectory_500hz.npz."""
        slug = slugify_model_name(run_res.model)
        ep_dir = self.output_root / f"{slug}_seed{seed}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        obs = env.get_obs()
        canvas_path = ep_dir / "canvas_final.png"
        grid_path = ep_dir / "canvas_grid.png"
        overhead_path = ep_dir / "overhead_final.png"
        third_person_path = ep_dir / "third_person_final.png"
        gif_path = ep_dir / "timelapse.gif"
        npz_path = ep_dir / "trajectory_500hz.npz"

        Image.fromarray(obs["canvas_rgb"]).save(canvas_path)
        Image.fromarray(obs["canvas_annotated_rgb"]).save(grid_path)
        Image.fromarray(obs["rgb_overhead_cam"]).save(overhead_path)
        Image.fromarray(obs["rgb_third_person_cam"]).save(third_person_path)

        timelapse_frames = getattr(env, "timelapse_frames", [])
        if timelapse_frames:
            pil_frames = [Image.fromarray(f) for f in timelapse_frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=90,
                loop=0,
            )
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

        rel_canvas = f"{ep_dir.name}/canvas_final.png"
        rel_gif = f"{ep_dir.name}/timelapse.gif"

        metrics = build_episode_metrics(
            run_res=run_res,
            visual_res=visual_res,
            seed=seed,
            ink_pixels_drawn=int(obs["ink_pixels_drawn"]),
            canvas_rel_path=rel_canvas,
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
        """Write SUMMARY.md and summary_metrics.json inside output_root."""
        summary_json_path = self.output_root / "summary_metrics.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in metrics_list], f, indent=2)

        table_md = format_markdown_summary_table(metrics_list)
        summary_md_path = self.output_root / "SUMMARY.md"
        content = (
            "# PelicanBench Evaluation Summary\n\n"
            f"{table_md}\n"
        )
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(content)
        return summary_md_path
