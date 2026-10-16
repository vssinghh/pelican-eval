# PelicanBench Leaderboard

Comparison of AI models controlling the 6-DoF **SO-ARM101** robot arm to draw a **pelican riding a bicycle** on a whiteboard.

***

## 1. Level 1 (`sim`): MuJoCo `500 Hz` Simulation

| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | Tokens (`Total (In / Out)`) | End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Visual Output |
| :- | :- | :-: | :-: | :-: | :-: | :-: | :-: | :- |
| `gemini/gemini-4-argon` | Google | `0 (0 / 0)` | `8` | `99,335 (90,651 / 8,684)` | `153.85s` | `\$0.1438` | **`10.0 / 10`** | [Overhead](results/sim_benchmark/gemini_gemini_4_argon_seed0/overhead_final.png) / [GIF](results/sim_benchmark/gemini_gemini_4_argon_seed0/timelapse.gif) |
| `gemini/gemini-3.8-flash` | Google | `0 (0 / 0)` | `8` | `154,022 (131,904 / 22,118)` | `162.01s` | `\$0.1476` | **`10.0 / 10`** | [Overhead](results/sim_benchmark/gemini_gemini_3_8_flash_seed0/overhead_final.png) / [GIF](results/sim_benchmark/gemini_gemini_3_8_flash_seed0/timelapse.gif) |
| `reference/expert` | Scripted Baseline | `0 (0 / 0)` | `5` | `0 (0 / 0)` | `1.37s` | `\$0.0000` | **`10.0 / 10`** | [Overhead](assets/verification/reference_overhead.png) / [GIF](assets/verification/reference_timelapse.gif) |

***

## 2. Level 2 (`real`): Physical SO-ARM101 + Whiteboard

| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | Tokens (`Total (In / Out)`) | End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Visual Output |
| :- | :- | :-: | :-: | :-: | :-: | :-: | :-: | :- |
| `reference/expert` | Scripted Baseline | `0 (0 / 0)` | `5` | `0 (0 / 0)` | `0.13s` | `\$0.0000` | **`10.0 / 10`** | [Overhead](assets/verification/reference_overhead.png) / [GIF](assets/verification/reference_timelapse.gif) |
