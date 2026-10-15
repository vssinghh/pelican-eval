# PelicanBench (`pelican-eval`)

**PelicanBench** is an embodied robotics evaluation benchmark and dataset that tests Large Language Models (LLMs) and Vision-Language Models (VLMs) on drawing a **pelican riding a bicycle** using a 6-DoF **SO-ARM101** robot arm (`5` arm joints + `1` marker gripper). Every model runs inside an identical Python `LiteLLM` agentic harness across two execution domains: **Level 1 (`sim` in MuJoCo at `500 Hz`)** and **Level 2 (`real` on a physical SO-ARM101 with a whiteboard and marker)**.

***

## 1. The 2 Execution Levels

Both levels share the same `SO101WhiteboardEnv` interface, metric robot base coordinate frame (`[X, Y, Z]` in meters), and tool definitions:

| Level | Domain Flag | Environment | What It Evaluates |
| :-: | :- | :- | :- |
| **`Level 1`** | `-d sim` | MuJoCo `500 Hz` digital twin (`assets/so101/whiteboard_scene.xml`) | Standardized multi-model benchmark, physics hallucination tracking, and dataset collection |
| **`Level 2`** | `-d real` | Physical **SO-ARM101** follower arm + tabletop whiteboard + marker | Real-world execution of LLM trajectories and closed-loop webcam visual feedback |

***

## 2. The 5 Benchmark Metrics

Every evaluation episode logs a standardized dataset bundle and scores the model on 5 core metrics:

| Metric | Unit | What We Measure |
| :- | :- | :- |
| **1. Hallucinations** | `Count (Spatial / Code)` | Out-of-bounds `[X, Y, Z]` points, table collisions, IK reach failures, air-drawing (`0` ink), and Python/tool errors |
| **2. Tool Calls** | `Count` | Total tool invocations across `get_workspace_spec`, `draw_primitives`, `draw_waypoints`, `run_python`, `inspect_canvas` |
| **3. End-to-End Time** | `Seconds (s)` | Total wall-clock duration from initial prompt dispatch to `finish_drawing()` |
| **4. Cost** | `USD (\$)` | Exact prompt, completion, reasoning, and camera image token cost via `LiteLLM` |
| **5. Visual Output** | `0 to 10` + `PNG/GIF` | 5-part anatomical rubric (`2 pts` each) + rendered 2D canvas, overhead camera view, and timelapse GIF |

***

## 3. Quickstart Commands

```bash
# Install dependencies
uv sync

# 1. Verify Level 1 MuJoCo whiteboard & scripted reference drawing
uv run python -m envs.verify_whiteboard -d sim

# 2. Run a single model evaluation in MuJoCo Sim
uv run python -m evaluation -d sim -m anthropic/claude-sonnet-4-20250514
uv run python -m evaluation -d sim -m gemini/gemini-2.5-pro
uv run python -m evaluation -d sim -m openai/gpt-4o

# 3. Sweep all configured models in config/models.yaml and update leaderboard
uv run python -m evaluation -d sim -m all -o results/sim_benchmark

# 4. Calibrate 4 whiteboard corners and run on the physical SO-ARM101 (Level 2)
uv run python -m envs.calibrate_board -p /dev/tty.usbmodem5B790143431
uv run python -m evaluation -d real -m anthropic/claude-sonnet-4-20250514

# 5. Run unit tests
uv run pytest
```

***

## 4. Documentation
* **[`DESIGN.md`](./DESIGN.md)**: System architecture, repository layout, 6-tool agentic sandbox contract, MuJoCo `500 Hz` ink physics, SO-ARM101 calibration, and dataset format.
* **[`LEADERBOARD.md`](./LEADERBOARD.md)**: Head-to-head model comparison table, cost/latency breakdowns, and visual canvas gallery.
