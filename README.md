# PelicanBench (`pelican-eval`)

**PelicanBench** tests how well AI models (`Claude`, `Gemini`, `GPT`, `DeepSeek`, `Qwen`) control a 6-DoF **SO-ARM101** robot arm (`5` arm joints + `1` gripper holding a marker) to draw a **pelican riding a bicycle** on a whiteboard.

Every model gets the exact same prompt and the same 6 tools across two levels:
1. **Level 1 (`sim`)**: Draw on a virtual whiteboard in **MuJoCo** (`500 Hz` physics simulation).
2. **Level 2 (`real`)**: Draw on a physical whiteboard using a real **SO-ARM101** robot arm and marker.

***

## 1. The 2 Levels

| Level | Flag | Environment | What It Does |
| :-: | :- | :- | :- |
| **`Level 1`** | `-d sim` | MuJoCo `500 Hz` simulation (`assets/so101/whiteboard_scene.xml`) | Runs the simulated SO-ARM101 arm, tracks errors, and saves the drawing and trajectory data |
| **`Level 2`** | `-d real` | Physical **SO-ARM101** arm + whiteboard + marker | Runs the same model and tool calls on the real robot arm over USB |

***

## 2. The 5 Metrics We Measure

| Metric | Unit | What It Measures |
| :- | :- | :- |
| **1. Hallucinations** | `Count (Spatial / Code)` | Points outside the whiteboard, table crashes (`Z < 0`), unreachable arm poses, drawing in mid-air (`0` ink), or broken Python/tool calls |
| **2. Tool Calls** | `Count` | How many tool calls the model used to finish the drawing |
| **3. End-to-End Time** | `Seconds (s)` | Total time from sending the prompt to finishing the drawing |
| **4. Cost** | `USD (\$)` | Total API token cost (text + camera images) tracked by `LiteLLM` |
| **5. Visual Output** | `0 to 10` + `PNG/GIF` | Score out of `10` (`2` points each for wheels, bike frame, pelican head/beak, body/limbs, and riding pose) plus the final image and GIF |

***

## 3. Quickstart Commands

```bash
# Install dependencies
uv sync

# 1. Test the MuJoCo whiteboard with a built-in reference drawing (no API key needed)
uv run python -m envs.verify_whiteboard -d sim

# 2. Add your API key(s) to .env, then run any model in Sim (Level 1)
uv run python -m evaluation -d sim -m anthropic/claude-sonnet-4-20250514
uv run python -m evaluation -d sim -m gemini/gemini-2.5-pro
uv run python -m evaluation -d sim -m openai/gpt-4o

# 3. Run all models listed in config/models.yaml
uv run python -m evaluation -d sim -m all -o results/sim_benchmark

# 4. Calibrate the 4 whiteboard corners and run on the real SO-ARM101 arm (Level 2)
uv run python -m envs.calibrate_board -p /dev/tty.usbmodem5B790143431
uv run python -m evaluation -d real -p /dev/tty.usbmodem5B790143431 -m anthropic/claude-sonnet-4-20250514

# 5. Run tests
uv run pytest
```

***

## 4. Documentation
* **[`DESIGN.md`](./DESIGN.md)**: How the robot workspace, tools, MuJoCo ink physics, and real-robot calibration work.
* **[`LEADERBOARD.md`](./LEADERBOARD.md)**: Model comparison table and output links.
