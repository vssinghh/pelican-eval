# PelicanBench Leaderboard

Comparison of AI models controlling the 6-DoF **SO-ARM101** robot arm to draw a **pelican riding a bicycle** on a whiteboard.

***

## 1. Level 1 (`sim`): MuJoCo `500 Hz` Simulation

| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Canvas & Replay |
| :- | :- | :-: | :-: | :-: | :-: | :-: | :- |
| `reference/expert` | Scripted Baseline | `0 (0 / 0)` | `5` | `1.37s` | `\$0.0000` | **`10.0 / 10`** | [Canvas](assets/verification/reference_canvas_grid.png) / [GIF](assets/verification/reference_timelapse.gif) |

***

## 2. Level 2 (`real`): Physical SO-ARM101 + Whiteboard

| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Board Output |
| :- | :- | :-: | :-: | :-: | :-: | :-: | :- |
| `reference/expert` | Scripted Baseline | `0 (0 / 0)` | `5` | `0.13s` | `\$0.0000` | **`10.0 / 10`** | [Canvas](assets/verification/reference_canvas.png) |
