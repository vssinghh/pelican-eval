# PelicanBench Leaderboard

Head-to-head comparison of Large Language Models and Vision-Language Models controlling the 6-DoF **SO-ARM101** robot arm to draw a **pelican riding a bicycle**.

***

## 1. Level 1 (`sim`): MuJoCo `500 Hz` Digital Twin Benchmark

| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Canvas & Replay |
| :- | :- | :-: | :-: | :-: | :-: | :-: | :- |
| `reference/expert` | Reference Baseline | `0 (0 / 0)` | `5` | `1.37s` | `\$0.0000` | **`10.0 / 10`** | [Canvas](assets/verification/reference_canvas_grid.png) / [GIF](assets/verification/reference_timelapse.gif) |

***

## 2. Level 2 (`real`): Physical SO-ARM101 + Whiteboard Benchmark

| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Physical Board Photo |
| :- | :- | :-: | :-: | :-: | :-: | :-: | :- |
| `reference/expert (calibrated FK)` | Reference Baseline | `0 (0 / 0)` | `5` | `0.13s` | `\$0.0000` | **`10.0 / 10`** | [Canvas](results/real_benchmark/reference_expert_seed0/canvas_final.png) |
