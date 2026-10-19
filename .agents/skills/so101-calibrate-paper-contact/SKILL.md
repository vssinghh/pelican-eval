---
name: so101-calibrate-paper-contact
description: >-
  Calibrate vertical marker orientation (90 degrees to the table) and exact paper contact
  height (Z-plane) on the physical SO-ARM101 robot arm after picking up a marker. Use this
  skill after so101-pickup-marker and before drawing so the marker tip leaves crisp lines
  without dragging or hovering in mid-air.
---

# SO-ARM101 Paper Contact and Vertical Pitch Calibration (`so101-calibrate-paper-contact`)

Every time the robot picks up a marker, the grasp height along the barrel varies by a few millimeters. This skill guides you to orient the grasped marker dead vertical (`90.0°`) over the drawing sheet, find the exact paper contact height `z_contact` using `Camera 1` (`side_cam`) and `Camera 0` (`overhead_cam`), and save the calibrated plane into [calibration.json](../../config/calibration.json).

***

## 1. Why Vertical Orientation (`90.0°`) Matters

* **Anti-Slip Palm Bracing:** When the marker stands vertical (`90.0°` to the table), the top butt end of the marker presses directly against the solid underside of the gripper palm. Upward reaction forces from the paper cannot push the marker upward through the jaws.
* **Uniform Line Width:** A vertical felt tip writes equally well in all `360°` planar stroke directions (`+X`, `-X`, `+Y`, `-Y`).
* **Typical Pitch Angle:** For a marker picked up vertically from the holder, `target_pitch = q1 + q2 + q3` is approximately **`0.95 rad` (`54.4°`)**. Always verify the actual marker angle in `Camera 1`.

***

## 2. Step-by-Step Closed-Loop Calibration Procedure

Use [so101_cli.py](../../scripts/so101_cli.py) to move and inspect both cameras at each step.

### Step 1: Verify the Marker Is Still Clamped
Before moving over the paper, check `status`:
```bash
uv run python scripts/so101_cli.py status cal_check
```
Confirm `grasp_state == "MARKER_CLAMPED"` (`2075 <= gripper_present_position_reg56 <= 2135`). If not, invoke `so101-pickup-marker` first.

### Step 2: Move to a Safe Hover Pose Over the Center of the Paper
Look at `Camera 0` (`cal_check_cam_0_grid.png`) to confirm where the lined sheet sits in front of the robot (typically centered near `x = 0.23 m, y = 0.00 m`). Move `pinch_site` to a safe hover height (`z = 0.020 m`) over the paper center with `pitch = 0.95 rad` and gripper clamped (`1700`):
```bash
uv run python scripts/so101_cli.py move-ik 0.230 0.000 0.020 0.95 1700 cal_hover
```
Inspect `outputs/live/cal_hover_cam_1_grid.png` (`Camera 1` side view) and `outputs/live/cal_hover_cam_0_grid.png` (`Camera 0` overhead view):
1. **Verify Vertical Angle in `Camera 1`:** Check whether the marker barrel is parallel to the vertical yellow pixel grid lines (`x = 400, 600, 800`).
   * If the marker tip points slightly forward (away from the base), increase `pitch` by `+0.05 to +0.10 rad`.
   * If the marker tip points slightly backward (toward the base), decrease `pitch` by `-0.05 to -0.10 rad`.
2. **Estimate Tip-to-Paper Gap in `Camera 1`:** Look at the vertical gap between the blue/red/green felt tip and its shadow on the white paper.

### Step 3: Step Down in Z Until the Tip Meets Its Shadow
Lower `z` in small steps (`0.010 m` when more than `2 cm` above the sheet, then `0.003 m` near the surface) at `(x = 0.230, y = 0.000)`:
```bash
uv run python scripts/so101_cli.py move-ik 0.230 0.000 <candidate_z> <verified_pitch> 1700 cal_touch
```
After each step, inspect `outputs/live/cal_touch_cam_1_grid.png`:
* **In Air:** A visible gap remains between the felt tip and its shadow on the paper. Lower `z` by another `0.003 m` to `0.005 m`.
* **Exact Contact (`z_center`):** The felt tip touches its shadow on the paper without bowing the paper or tilting the marker barrel.
* **Too Deep:** The paper bows downward or the marker tilts in the jaws. Raise `z` by `+0.003 m`.

### Step 4: Lift and Verify the Ink Dot in `Camera 0`
Lift straight up to `z = z_center + 0.025 m` and retract slightly so `Camera 0` can see `(0.230, 0.000)`:
```bash
uv run python scripts/so101_cli.py move-ik 0.190 -0.060 0.030 <verified_pitch> 1700 cal_verify_dot
```
Inspect `outputs/live/cal_verify_dot_cam_0_grid.png`:
* Confirm a visible colored ink dot appears at the paper center. If no dot appeared, lower `z_center` by `0.002 m` and test once more.
* Confirm `gripper_present_position_reg56` is unchanged (zero slip).

### Step 5: Save Calibration Parameters to `config/calibration.json`
The physical SO-ARM101 arm exhibits slight kinematic sag as it reaches forward (`dz/dx = -0.08`, i.e., `z_board(x, y) = a*x + b*y + c` where `a = -0.08`, `b = 0.0`). Given your verified `z_center` at `x_center = 0.230`:
1. Compute `c = z_center - (-0.08 * 0.230) = z_center + 0.0184`.
2. Update [config/calibration.json](../../config/calibration.json) with:
   * `"plane_coeffs_abc": [-0.08, 0.0, <computed_c>]`
   * `"target_pitch": <verified_pitch>`
   * `"draw_z_pen_down": <z_center>`
   * `"draw_z_pen_up": <z_center + 0.028>`
