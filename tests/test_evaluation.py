"""Integration tests for VisualJudge, PelicanAgentRunner, and EpisodeRecorder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm

from envs.real_env import RealWhiteboardEnv
from envs.sim_env import SimWhiteboardEnv
from evaluation.recorder import EpisodeRecorder
from evaluation.visual_judge import VisualJudge
from harness.agent_loop import PelicanAgentRunner


def test_visual_judge_blank_vs_reference_pelican(tmp_path: Path) -> None:
    """Verify VisualJudge scores blank board 0/10 and reference pelican 10/10."""
    env = SimWhiteboardEnv(record_timelapse=True)
    judge = VisualJudge()
    try:
        obs_blank = env.reset()
        score_blank = judge.evaluate(obs_blank["canvas_rgb"])
        assert score_blank.total_score == 0.0

        runner = PelicanAgentRunner(model="reference/expert")
        run_res = runner.run_episode(env, domain="sim")
        assert run_res.finished_cleanly
        assert run_res.total_hallucinations == 0
        assert run_res.total_tool_calls == 5

        obs_done = env.get_obs()
        score_done = judge.evaluate(obs_done["canvas_rgb"])
        assert score_done.total_score >= 9.0

        recorder = EpisodeRecorder(output_root=tmp_path)
        metrics, ep_dir = recorder.save_episode(env, run_res, score_done, seed=0)
        assert (ep_dir / "metrics.json").exists()
        assert (ep_dir / "transcript.json").exists()
        assert (ep_dir / "overhead_final.png").exists()
        assert (ep_dir / "timelapse.gif").exists()
        assert (ep_dir / "trajectory_500hz.npz").exists()
        assert metrics.visual_score >= 9.0
    finally:
        env.close()


def test_litellm_provider_agent_loop_on_real_env_mock(monkeypatch: Any) -> None:
    """Verify PelicanAgentRunner with LiteLLM tool calling on RealWhiteboardEnv."""
    call_seq = [
        SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="get_workspace_detail",
                arguments="{}",
            ),
        ),
        SimpleNamespace(
            id="call_2",
            function=SimpleNamespace(
                name="run_python",
                arguments='{"code": "draw_primitives([{\'type\': \'circle\', \'center_xy\': [0.19, 0.05], \'radius\': 0.02}])"}',
            ),
        ),
        SimpleNamespace(
            id="call_3",
            function=SimpleNamespace(
                name="finish_drawing",
                arguments='{"summary": "Drew test wheel via LiteLLM."}',
            ),
        ),
    ]
    turn_counter = {"idx": 0}

    def _fake_completion(**kwargs: Any) -> Any:
        del kwargs
        idx = turn_counter["idx"]
        turn_counter["idx"] += 1
        tc = call_seq[min(idx, len(call_seq) - 1)]
        msg = SimpleNamespace(content="Executing step", tool_calls=[tc])
        choice = SimpleNamespace(message=msg)
        usage = SimpleNamespace(prompt_tokens=200, completion_tokens=50)
        return SimpleNamespace(choices=[choice], usage=usage)

    monkeypatch.setattr(litellm, "completion", _fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0015)

    real_env = RealWhiteboardEnv(mock_hardware=True)
    try:
        runner = PelicanAgentRunner(model="anthropic/claude-sonnet-4-20250514", max_turns=5)
        res = runner.run_episode(real_env, domain="real")
        assert res.finished_cleanly
        assert res.total_tool_calls == 3
        assert res.total_hallucinations == 0
        assert res.prompt_tokens == 600
        assert res.completion_tokens == 150
        assert abs(res.cost_usd - 0.0045) < 1e-6
    finally:
        real_env.close()
