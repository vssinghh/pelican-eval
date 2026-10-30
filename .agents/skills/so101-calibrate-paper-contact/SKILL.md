---
name: so101-calibrate-paper-contact
description: >-
  Calibrate vertical marker orientation (90 degrees to the table) and exact paper contact
  height (Z-plane) on the physical SO-ARM101 robot arm after picking up a marker. Use this
  skill after so101-pickup-marker and before drawing so the marker tip leaves crisp lines
  without dragging or hovering in mid-air.
---

# SO-ARM101 Paper Contact and Vertical Pitch Calibration (`so101-calibrate-paper-contact`)

Every time the robot picks up a marker, the grasp height along the barrel varies by a few millimeters. This skill guides you to orient the grasped marker vertical (`90.0°` to the table) over the drawing sheet, find the exact paper contact height `z_contact` using `Camera 1` (`side_cam`) and `Camera 0` (`overhead_cam`), and save the calibrated plane into [calibration.json](../../config/calibration.json).

***

## 1. Why Vertical Orientation (`90.0°`) and High Initial Hover Matter

* **Anti-Slip Palm Bracing:** When the marker stands vertical (`90.0°` to the table), the top end of the marker presses directly against the solid inner wall of the gripper palm. Upward reaction forces from the paper cannot push the marker upward through the jaws.
* **Uniform Line Width:** A vertical felt tip writes equally well in all planar stroke directions (`+X`, `-X`, `+Y`, `-Y`).
* **Critical `pinch_site` Height Offset:** In [so101_cli.py](../../scripts/so101_cli.py), `move-ik` and `move-polar` solve for `pinch_site` (the center of the gripper jaws), while the marker tip extends `5.5 cm to 6.5 cm` below `pinch_site` when held vertically (`pitch = 1.65 rad`).
  * **NEVER** command `pinch_site` below `z = 0.055 m` during initial hover! Always begin hovering at **`z = 0.085 m to 0.090 m`**, which places the marker tip `~2.5 cm` safely above the paper.
* **Hardware Joint Limits (`Motor 3` and `Motor 4`):**
  * `Motor 4` (`wrist_flex`) physically saturates at **`2924` ticks (`1.520 rad`)**.
  * `Motor 3` (`elbow_flex`) must remain `>= 2087` ticks (`0.08 rad`) to avoid the straight-elbow singularity.
  * Within the primary paper drawing region (`x in [0.202, 0.226]`, `y in [-0.055, -0.010]`), `target_pitch = 1.65 rad` keeps `Motor 4 <= 2921` ticks and orients the marker vertical (`90.0°` to the table in `Camera 1`).

***

## 2. Step-by-Step Closed-Loop Calibration Procedure

Use [so101_cli.py](../../scripts/so101_cli.py) to move and inspect both cameras at each step.

### Step 1: Verify the Marker Is Clamped in the Flat Pad
Before moving over the paper, check `status`:
```bash
uv run python scripts/so101_cli.py status cal_check
```
Confirm `grasp_state == "BARREL_HELD_IN_FLAT_PAD"` (`2060 <= gripper_present_position_reg56 <= 2350`). If `grasp_state == "EMPTY_CLOSED"`, invoke `so101-pickup-marker` first.

### Step 2: Move to a Safe High Hover Pose (`z = 0.088 m`) Over the Near Paper Zone
Move `pinch_site` to a safe high hover (`z = 0.088 m`) over the near-center portion of the lined sheet (`x = 0.206 m, y = -0.020 m`) with vertical `pitch = 1.65 rad` and gripper clamped (`1700`):
```bash
uv run python scripts/so101_cli.py move-ik 0.206 -0.020 0.088 1.65 1700 cal_hover
```
Inspect `outputs/live/cal_hover_cam_1_grid.png` (`Camera 1` side view) and `outputs/live/cal_hover_cam_0_grid.png` (`Camera 0` overhead view):
1. **Verify Vertical Angle in `Camera 1`:** Confirm the marker barrel is parallel to the vertical yellow pixel grid lines (`x = 400, 500, 600`).
2. **Estimate Tip-to-Paper Gap in `Camera 1`:** Look at the vertical gap between the felt tip and the lined paper surface (`y_px ≈ 760..820`).

### Step 3: Step Down in Z at Two Points (Near and Far) Until the Tip Touches Its Shadow
1. **Near Point (`x = 0.206, y = -0.020`):**
   Lower `z` in small steps (`0.010 m` while clearly in air, then `0.002 m` near the surface, typically reaching contact around `z_near ≈ 0.061 m`):
   ```bash
   uv run python scripts/so101_cli.py move-ik 0.206 -0.020 <candidate_z> 1.65 1700 cal_touch_near
   ```
   Lift to `z = 0.088 m` and verify a crisp ink dot in `Camera 1` and `Camera 0`.
2. **Far Point (`x = 0.224, y = -0.045`):**
   Hover at `(0.224, -0.045, 0.088)`, then step down to find exact contact at the far reach (typically `z_far ≈ 0.0595 m`):
   ```bash
   uv run python scripts/so101_cli.py move-ik 0.224 -0.045 <candidate_z> 1.65 1700 cal_touch_far
   uv run python scripts/so101_cli.py move-ik 0.224 -0.045 0.088 1.65 1700 cal_far_dot_check
   ```
   Confirm a second crisp ink dot appears on the paper and `gripper_present_position_reg56` remains stable (`|ΔReg 56| <= 3 ticks`).

### Step 4: Save Calibration Parameters to `config/calibration.json` and Verify with `draw-strokes`
1. Fit the linear plane `z_board(x, y) = a*x + b*y + c` (with `a = -0.08`, `b = 0.0`, and `c = z_near + 0.08 * 0.206`, typically `c ≈ 0.0770`).
2. Update [config/calibration.json](../../config/calibration.json) with:
   * `"board_x_range": [0.202, 0.226]`
   * `"board_y_range": [-0.055, -0.010]`
   * `"plane_coeffs_abc": [-0.08, 0.0, <computed_c>]`
   * `"target_pitch": 1.65`
   * `"draw_z_pen_down": <z_far - 0.0005>`
   * `"draw_z_pen_up": 0.0880`
3. Execute a verification stroke via `draw-strokes`, which automatically retracts to the right-side unobstructed pose (`[2650, 1000, 2750, 2550, 973, 1700]`) and captures `Camera 0` for visual confirmation.
