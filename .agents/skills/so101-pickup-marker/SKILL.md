---
name: so101-pickup-marker
description: >-
  Pick up a marker from a cup or figurine holder on the physical SO-ARM101 robot arm using
  closed-loop triple-camera visual servoing (overhead, side, and eye-in-hand wrist cameras)
  and gripper encoder verification. Use this skill whenever the agent needs to locate, align
  with, grasp, and lift a marker without relying on hardcoded coordinates.
---

# SO-ARM101 Closed-Loop Marker Pickup (`so101-pickup-marker`)

This skill guides you to pick up an upright marker from its holder on the physical SO-ARM101 follower arm (`/dev/tty.usbmodem5B790143431`) using live triple-camera feedback (`Camera 0` overhead, `Camera 1` side, `Camera 2` eye-in-hand wrist) and servo register readouts (`Reg 56`). Never rely on hardcoded workspace coordinates because the lightweight holder and paper shift between runs.

***

## 1. Physical Kinematics, Decoupled Polar Control, and Camera Map

Always use the decoupled cylindrical primitive `move-polar` in [so101_cli.py](../../scripts/so101_cli.py) rather than Cartesian `move-ik` when approaching the marker:

```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> <reach_m> <z_m> 0.95 <gripper_tick> <out_prefix>
```

Why `move-polar` with `pitch_rad = 0.95`:
* It locks `Motor 1` (`shoulder_pan`) at the exact integer `<pan_tick>` so advancing `<reach_m>` moves the jaws straight along the wrist camera's optical axis without sweeping sideways.
* It locks `Motor 5` (`wrist_roll`) at `973` (`-pi/2 rad`), keeping the wrist camera on top, the **fixed jaw on the left**, and the **moving jaw opening to the right**.
* Setting `pitch_rad = 0.95` keeps the gripper jaws horizontal (`q1 + q2 + q3 = 0.95 rad`) for a clean perpendicular pinch on a vertical barrel.

| Control Parameter | Physical Axis | Visual Effect in Primary Camera | Primary Camera to Observe |
| :- | :- | :- | :- |
| **`pan_tick` (`Motor 1`)** | Base azimuth yaw | **Increasing `pan_tick`** rotates the arm clockwise, which shifts the marker **LEFT (`-x`)** in `Camera 2`. **Decreasing `pan_tick`** rotates counter-clockwise, shifting the marker **RIGHT (`+x`)** in `Camera 2`. Note: STS3215 servos have an `~8 tick` static friction deadband; changes smaller than `10 ticks` may not move the joint. | `Camera 2` (`wrist_cam`) |
| **`z_m` (Elevation)** | Jaw height above table | Raising `z_m` moves the jaw tips **UP (`-y`)** in `Camera 1`. Note: extending `reach_m` forward increases gravity load on `Motor 2`, causing `~2 to 4 mm` of vertical droop per `2 cm` of reach. | `Camera 1` (`side_cam`) |
| **`reach_m` (Radial reach)** | Distance from base axis | Increasing `reach_m` moves the jaws forward along the ray defined by `pan_tick` (**LEFT / `-x`** in `Camera 1`, and brings the marker **DOWN / `+y`** toward the jaws in `Camera 2`). | `Camera 1` and `Camera 2` |
| **`gripper_tick` (`Motor 6`)** | Pinch jaw | **`3145` = Wide Open**. **`2035` = Closed Empty (Sighting Pose)**. **`1700` = Commanded Firm Grasp** (`Reg 48 = 750`, `Reg 34 = 75%`). | `Reg 56` and `Camera 2` |

***

## 2. Four Physical Rules for 100% Repeatable Grasping

Before moving the arm, internalize these four physical invariants of the SO-ARM101 gripper and camera geometry. View the linked reference images in `references/` to verify what each state looks like.

### Rule 1: The Grasp-Height Rule (Always Approach `1 to 2 cm` Below the Top Cap)
* **Never approach near the bottom of the marker (`z_m < 0.030 m`).** When the jaws are low near the holder, the upper `8 cm` of the marker sticks up right in front of the `Camera 2` lens, creating a giant out-of-focus black shadow that hides the jaws (see [bad_too_low_occlusion_cam2.png](references/bad_too_low_occlusion_cam2.png)).
* **Always set `z_m` (`~0.045 to 0.052 m`) so the jaw tips approach `1 to 2 cm` below the top cap of the marker in `Camera 1`** (see [good_top_height_cam1.png](references/good_top_height_cam1.png)). At this height:
  1. Nothing extends above the jaws to block `Camera 2` or create vertical parallax.
  2. The circular top cap of the marker in `Camera 2` is at the exact height of the jaws.
  3. The jaws stay `4+ cm` above the figurine or cup holder, preventing collisions with the holder.

### Rule 2: The Single-Acting Jaw Rule (Align Marker to the Right of the Fixed Jaw Wall in `Camera 2`)
* In `Camera 2`, the **left jaw is fixed** and the **right jaw is the pivoting finger** that swings open to the right. When the gripper is wide open (`3145`), the moving jaw swings out of view to the bottom-right, leaving only the fixed jaw visible on the left.
* **Never align the marker directly in front of the black plastic tip of the fixed jaw.** If you advance with the marker in front of the fixed jaw tip, the rounded tip will ram the back of the marker and push it aside (see [bad_fixed_jaw_hit_cam2.png](references/bad_fixed_jaw_hit_cam2.png)).
* **Always align `pan_tick` so the circular top cap of the marker sits immediately to the RIGHT of the fixed jaw's flat grey rubber pad line** (see [good_sight_alignment_cam2.png](references/good_sight_alignment_cam2.png)).
* **Closed-Jaw Rifle-Sight Trick:** During pre-approach (`2 to 3 cm` short of the marker), you can set `gripper_tick = 2035` (closed) so both jaw tips meet in `Camera 2` like a rifle front sight. Adjust `pan_tick` until the left edge of the marker cap lines up with the seam between the two closed jaw tips, then open the gripper (`3145`) in place before advancing.

### Rule 3: The Enclosure Depth Rule (Advance Until the Rounded Tip Passes the Marker Cap)
* The gripping surface of the jaw is the **flat grey rubber pad** behind the rounded black tip, not the tip itself.
* When advancing `reach_m` from the pre-grasp pose:
  1. In `Camera 2`, advance until the **rounded black tip of the fixed jaw extends past the front (top) edge of the marker cap**, so the circular marker cap sits directly beside the flat grey rubber pad on the right side of the fixed jaw.
  2. In `Camera 1`, verify that the black jaw tip extends `~1 cm` past the marker shaft at that height.
  3. Raise `z_m` by `+0.002 to 0.004 m` as you increase `reach_m` by `+0.020 m` to compensate for `Motor 2` gravity droop.

### Rule 4: The Closed-Jaw Self-Correction Rule (Inspect Before Opening on a Miss)
* After running `gripper clamp`, check `gripper_present_position_reg56` in the JSON output:
  * **`2075 <= Reg 56 <= 2135` (`grasp_state == "MARKER_CLAMPED"`):** The `10 mm` marker barrel is physically clamped between the rubber pads.
  * **`Reg 56 <= 2045` (`grasp_state == "EMPTY_CLOSED"`):** The jaws closed empty.
* **If `EMPTY_CLOSED` occurs, DO NOT open the gripper in place** (opening the pivoting jaw next to the marker can knock the marker over). Instead:
  1. Inspect `Camera 2` while the jaws are still closed (`2035`) to read the exact miss direction from the closed-jaw pointer:
     * **Marker cap is in front of (`smaller y` than) the closed jaw tips:** Reach was too short.
     * **Marker is pushed to the left of the fixed jaw:** `pan_tick` was too large (fixed jaw hit the marker).
     * **Marker is to the right of the closed moving jaw:** `pan_tick` was too small.
  2. **Retract `reach_m` by `-0.025 m` first while keeping `gripper_tick = 2035` (closed).**
  3. Open the gripper (`3145`) at the retracted pose, apply the `pan_tick` or `reach_m` correction, and re-advance.

***

## 3. Step-by-Step Closed-Loop Execution Procedure

### Step 1: Observe the Workspace and Locate the Marker
Run `status` to auto-discover the follower serial port, read current joint ticks, and capture grid-annotated frames from all three cameras (`cam_0`, `cam_1`, `cam_2`):
```bash
uv run python scripts/so101_cli.py status step1_observe
```
* If `status` reports `"FOLLOWER_USB_NOT_FOUND"` or `"SERVOS_NOT_RESPONDING"`, stop and ask the user to check the follower USB-C cable and 12V power supply.
* Inspect `outputs/live/step1_observe_cam_0_grid.png` and `outputs/live/step1_observe_cam_1_grid.png` to confirm the marker is upright in the holder and estimate its coarse azimuth (`pan_tick`, typically `2380 to 2450` when the holder is on the right side of the paper) and radial distance (`reach_m`, typically `0.18 to 0.20 m` for pre-approach).

### Step 2: Move to High Pre-Approach Pose (`1 to 2 cm` Below Marker Cap) and Align Azimuth
Move to a retracted pre-approach pose (`reach_m` `2 to 3 cm` short of the marker, `z_m` near the top cap around `0.048 to 0.052 m`, `pitch_rad = 0.95`, gripper closed `2035` for sighting or open `3145`):
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> 0.195 0.050 0.95 2035 step2_sight
```
*(Note: `so101_cli.py` automatically routes large `Motor 1` pans through a safe high overhead arc so the arm never sweeps low across the table.)*

Inspect `outputs/live/step2_sight_cam_1_grid.png` and `outputs/live/step2_sight_cam_2_grid.png`:
1. In `Camera 1`, verify the jaw tips are `1 to 2 cm` below the top cap of the marker (compare with [good_top_height_cam1.png](references/good_top_height_cam1.png)). Adjust `z_m` if needed.
2. In `Camera 2`, verify the circular top cap of the marker sits immediately to the **right** of the fixed jaw's inner wall / closed-jaw seam (compare with [good_sight_alignment_cam2.png](references/good_sight_alignment_cam2.png)).
   * If the marker cap is too far right in `Camera 2`, **increase `pan_tick`** (by at least `12 to 25 ticks` to clear servo deadband).
   * If the marker cap is in front of or to the left of the fixed jaw wall in `Camera 2`, **decrease `pan_tick`** (by at least `12 to 25 ticks`).

### Step 3: Open Jaws in Place and Advance Radially to Enclose the Cap
Once `pan_tick` and `z_m` are aligned, open the gripper (`3145`) at the pre-approach reach, then advance `reach_m` by `+0.020 to 0.030 m` (while adding `+0.002 to 0.004 m` to `z_m` to offset droop):
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> 0.195 0.050 0.95 3145 step3_open
uv run python scripts/so101_cli.py move-polar <pan_tick> <enclose_reach_m> 0.053 0.95 3145 step3_enclose
```
Inspect `outputs/live/step3_enclose_cam_2_grid.png` and `outputs/live/step3_enclose_cam_1_grid.png`:
* Confirm in `Camera 2` that the rounded tip of the fixed jaw extends past the front of the marker cap and the marker cap sits flush beside the flat grey rubber pad.
* Confirm in `Camera 1` that the jaw tip extends `~1 cm` past the marker barrel.

### Step 4: Clamp and Verify via Servo Register `Reg 56`
Command the firm grasp:
```bash
uv run python scripts/so101_cli.py gripper clamp step4_clamp
```
* **If `grasp_state == "MARKER_CLAMPED"` (`2075 <= gripper_present_position_reg56 <= 2135`):** Proceed to Step 5.
* **If `grasp_state == "EMPTY_CLOSED"` (`gripper_present_position_reg56 <= 2045`):** Follow **Rule 4** above (inspect `step4_clamp_cam_2_grid.png` while closed, retract `reach_m` to `0.195` with gripper closed `2035`, open at `0.195`, adjust `pan_tick` or `reach_m`, and repeat Step 3).

### Step 5: Lift Vertically Clear of the Holder
Once `Reg 56` confirms `MARKER_CLAMPED`, lift vertically by increasing `z_m` to `0.090 m` while keeping `pan_tick` locked and the gripper clamped (`1700`):
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> <enclose_reach_m> 0.090 0.95 1700 step5_lift
```
Inspect `outputs/live/step5_lift_cam_1_grid.png` and `outputs/live/step5_lift_cam_2_grid.png` to confirm the bottom tip of the marker is completely clear of the holder and `gripper_present_position_reg56` remains in `[2075, 2135]`.
