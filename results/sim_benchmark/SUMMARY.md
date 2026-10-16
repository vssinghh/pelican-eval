# PelicanBench Evaluation Summary

| Model | Provider | Hallucinations (`Spatial / Code`) | Tool Calls | Tokens (`Total (In / Out)`) | End-to-End Time (`s`) | Cost (`USD`) | Visual Score (`0..10`) | Visual Output |
| :- | :- | :-: | :-: | :-: | :-: | :-: | :-: | :- |
| `gemini/gemini-3.8-flash` | Google | `0 (0 / 0)` | `8` | `154,022 (131,904 / 22,118)` | `162.01s` | `\$0.1476` | **`10.0 / 10`** | [Overhead](gemini_gemini_3_8_flash_seed0/overhead_final.png) / [GIF](gemini_gemini_3_8_flash_seed0/timelapse.gif) |
| `gemini/gemini-4-argon` | Google | `0 (0 / 0)` | `8` | `99,335 (90,651 / 8,684)` | `153.85s` | `\$0.1438` | **`10.0 / 10`** | [Overhead](gemini_gemini_4_argon_seed0/overhead_final.png) / [GIF](gemini_gemini_4_argon_seed0/timelapse.gif) |
| `reference/expert` | Reference Baseline | `0 (0 / 0)` | `5` | `0 (0 / 0)` | `2.13s` | `\$0.0000` | **`10.0 / 10`** | [Overhead](reference_expert_seed0/overhead_final.png) / [GIF](reference_expert_seed0/timelapse.gif) |
