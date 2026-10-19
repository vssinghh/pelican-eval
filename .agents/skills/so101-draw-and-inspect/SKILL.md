---
name: so101-draw-and-inspect
description: >-
  Execute calibrated 2D drawing strokes on the physical SO-ARM101 robot arm, retract to the
  unobstructed inspection pose, and visually verify the drawn ink lines in Camera 0 while
  confirming zero marker slip via servo register Reg 56.
---

# SO-ARM101 Closed-Loop Drawing and Visual Inspection (`so101-draw-and-inspect`)

This skill guides you to draw normalized 2D strokes (`(u, v)` in `[0, 1] x [0, 1]`) on the physical paper using the contact calibration stored in [calibration.json](../../config/calibration.json), retract the arm out of the overhead camera's line of sight, and inspect the resulting drawing.

***

## 1. Canvas Coordinate Convention

All strokes use normalized coordinates `(u, v)` in `[0.0, 1.0]`:
* **`u` (`0.0` to `1.0`):** Horizontal axis across the sheet in `Camera 0` (`u = 0.0` is left/`+Y`, `u = 1.0` is right/`-Y`).
* **`v` (`0.0` to `1.0`):** Vertical axis up the sheet in `Camera 0` (`v = 0.0` is bottom/near base `x_min`, `v = 1.0` is top/far from base `x_max`).
* Keep primary drawing strokes within `[0.15, 0.85] x [0.15, 0.85]` so the arm stays well inside its smooth kinematic workspace.

***

## 2. Step-by-Step Closed-Loop Execution Procedure

### Step 1: Verify Pre-Conditions Before Drawing
1. Confirm `so101-calibrate-paper-contact` has been run for the currently held marker so [calibration.json](../../config/calibration.json) holds the live `plane_coeffs_abc` and `target_pitch`.
2. Record the initial `gripper_present_position_reg56` value (`2075 <= Reg 56 <= 2135`) so you can verify zero slip after the stroke batch completes.

### Step 2: Execute a Batch of Strokes and Retract for Inspection
Pass a JSON list of strokes (either inline or via a `.json` file path) to `draw-strokes` in [so101_cli.py](../../scripts/so101_cli.py):
```bash
uv run python scripts/so101_cli.py draw-strokes '[{"points": [[0.35, 0.50], [0.65, 0.50]]}]' batch1_inspect
```
What `draw-strokes` does automatically:
1. Maps each `(u, v)` waypoint onto the calibrated tilted board plane `z_board(x, y) = a*x + b*y + c` with a `2 mm` safety Z-floor (`z_floor = draw_z_pen_down - 0.002`) so the marker never slams into the table.
2. Interpolates densely (`<= 2.5%` canvas step per sub-waypoint) at constant vertical pitch (`target_pitch`) and horizontal wrist roll (`Motor 5 = 973`).
3. Lifts the marker `+28 mm` above the paper between disconnected strokes.
4. Retracts the arm to the right-side unobstructed pose (`[2480, 1150, 2950, 1950, 973, 1700]`) so `Camera 0` (`overhead_cam`) has a 100% clear view of the entire sheet.

### Step 3: Closed-Loop Visual and Encoder Verification
After every `draw-strokes` call, inspect `outputs/live/batch1_inspect_cam_0.png` and the JSON status readout:
1. **Check `grasp_state` and `Reg 56`:** Confirm `gripper_present_position_reg56` changed by no more than `5 ticks` from before the stroke.
2. **Check Ink Visibility in `Camera 0`:**
   * **Lines are crisp and continuous:** Proceed to the next stroke batch (or `so101-swap-marker` if switching colors).
   * **Lines are faint or missing in a region:** The paper height is slightly lower there. Lower `plane_coeffs_abc[2]` (`c`) and `draw_z_pen_down` in [calibration.json](../../config/calibration.json) by `-0.002 m` (`2 mm`) and re-run the missing stroke.
   * **Marker dragged during pen-up travel:** Increase `draw_z_pen_up` or re-run `so101-calibrate-paper-contact`.
