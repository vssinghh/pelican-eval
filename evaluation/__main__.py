"""CLI Entry Point for Running PelicanBench Evaluations Across Sim and Real Environments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from envs.real_env import RealWhiteboardEnv
from envs.sim_env import SimWhiteboardEnv
from evaluation.metrics import EpisodeMetrics, format_markdown_summary_table
from evaluation.recorder import EpisodeRecorder
from evaluation.visual_judge import VisualJudge
from harness.agent_loop import PelicanAgentRunner


def load_models_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Load default settings and model ID list from config/models.yaml."""
    if not config_path.exists():
        return {"max_turns": 10, "temperature": 0.2, "timeout_s": 120}, []
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("defaults", {"max_turns": 10, "temperature": 0.2, "timeout_s": 120})
    models = [entry["id"] for entry in data.get("models", []) if "id" in entry]
    return defaults, models


def main() -> None:
    """Run PelicanBench evaluation for one model or all models in config/models.yaml."""
    parser = argparse.ArgumentParser(
        description="PelicanBench: Evaluate LLMs/VLMs on drawing a pelican riding a bicycle."
    )
    parser.add_argument(
        "-d",
        dest="domain",
        default="sim",
        choices=("sim", "real"),
        help="Execution domain: 'sim' (MuJoCo 500 Hz) or 'real' (Physical SO-ARM101).",
    )
    parser.add_argument(
        "-m",
        dest="model",
        default="reference/expert",
        help="LiteLLM model ID (e.g. anthropic/claude-sonnet-4-20250514, gemini/gemini-2.5-pro, openai/gpt-4o, or 'all').",
    )
    parser.add_argument(
        "-o",
        dest="out_dir",
        default=None,
        help="Output directory for dataset bundles (defaults to results/sim_benchmark or results/real_benchmark).",
    )
    parser.add_argument(
        "-s",
        dest="seed",
        type=int,
        default=0,
        help="Episode seed index.",
    )
    parser.add_argument(
        "-t",
        dest="max_turns",
        type=int,
        default=None,
        help="Optional override for maximum agent tool turns.",
    )
    parser.add_argument(
        "-j",
        dest="judge_model",
        default=None,
        help="Optional VLM judge model ID (defaults to deterministic spatial rubric).",
    )
    parser.add_argument(
        "-p",
        dest="port",
        default="/dev/tty.usbmodem5B790143431",
        help="Serial port for physical SO-ARM101 when -d real is selected.",
    )
    parser.add_argument(
        "-k",
        dest="mock_real",
        action="store_true",
        help="Use mock hardware when testing -d real without USB servos connected.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or f"results/{args.domain}_benchmark"
    repo_root = Path(__file__).resolve().parent.parent
    defaults, configured_models = load_models_config(repo_root / "config" / "models.yaml")
    max_turns = int(args.max_turns if args.max_turns is not None else defaults.get("max_turns", 10))
    temperature = float(defaults.get("temperature", 0.2))
    timeout_s = int(defaults.get("timeout_s", 120))

    if args.model.strip().lower() == "all":
        model_list = configured_models
        if not model_list:
            raise ValueError("No models found in config/models.yaml.")
    else:
        model_list = [m.strip() for m in args.model.split(",") if m.strip()]

    env = (
        SimWhiteboardEnv(record_timelapse=True)
        if args.domain == "sim"
        else RealWhiteboardEnv(serial_port=args.port, mock_hardware=args.mock_real)
    )
    judge = VisualJudge(vlm_judge_model=args.judge_model)
    recorder = EpisodeRecorder(output_root=out_dir)
    all_metrics: list[EpisodeMetrics] = []

    try:
        for model_id in model_list:
            print(f"\n[PelicanBench] Evaluating model='{model_id}' on domain='{args.domain}'...")
            runner = PelicanAgentRunner(
                model=model_id,
                max_turns=max_turns,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            run_res = runner.run_episode(env, domain=args.domain)
            obs = env.get_obs()
            visual_res = judge.evaluate(obs["canvas_rgb"])
            ep_metrics, ep_dir = recorder.save_episode(
                env=env,
                run_res=run_res,
                visual_res=visual_res,
                seed=args.seed,
            )
            all_metrics.append(ep_metrics)

            print(f"  Hallucinations: {ep_metrics.total_hallucinations} "
                  f"(Spatial: {ep_metrics.spatial_hallucinations}, Code: {ep_metrics.code_hallucinations})")
            print(f"  Tool Calls:     {ep_metrics.total_tool_calls} {ep_metrics.tool_call_counts}")
            print(f"  Tokens Used:    {ep_metrics.total_tokens:,} "
                  f"(Input: {ep_metrics.prompt_tokens:,}, Output: {ep_metrics.completion_tokens:,})")
            print(f"  End-to-End:     {ep_metrics.e2e_time_s:.2f}s")
            print(f"  Cost (USD):     ${ep_metrics.cost_usd:.5f}")
            print(f"  Visual Score:   {ep_metrics.visual_score:.2f} / 10.0 ({visual_res.rationale})")
            print(f"  Bundle Saved:   {ep_dir.resolve()}")

        summary_path = recorder.write_summary_report(all_metrics)
        print("\n" + format_markdown_summary_table(all_metrics))
        print(f"\nSummary report written to: {summary_path.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
