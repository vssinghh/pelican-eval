---
name: so101-pickup-marker
description: >-
  Pick up a marker from a cup or figurine holder on the physical SO-ARM101 robot arm using
  closed-loop triple-camera visual servoing (overhead, side, and eye-in-hand wrist cameras)
  and gripper encoder verification. Use this skill whenever the agent needs to locate, align
  with, grasp, and lift a marker without relying on hardcoded coordinates.
---

# SO-ARM101 Closed-Loop Marker Pickup (`so101-pickup-marker`)

This skill guides you to pick up an upright marker from its holder on the physical SO-ARM101 follower arm (`/dev/tty.usbmodem5B790143431`) using live triple-camera feedback (`Camera 0` overhead, `Camera 1` side, `Camera 2` eye-in-hand wrist) and servo register readouts (`Reg 56` position and `Reg 60` load). Never rely on hardcoded workspace coordinates because the lightweight holder and paper shift between runs.

***

## 1. Physical Kinematics, Decoupled Polar Control, and Camera Map

Always use the decoupled cylindrical primitive `move-polar` in [so101_cli.py](../../scripts/so101_cli.py) rather than Cartesian `move-ik` when approaching the marker:

```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> <reach_m> <z_m> 0.95 <gripper_tick> <out_prefix>
```

Why `move-polar` with `pitch_rad = 0.95`:
* It locks `Motor 1` (`shoulder_pan`) at the exact integer `<pan_tick>` so advancing `<reach_m>` moves the jaws straight along the wrist camera optical axis without sweeping sideways.
* It locks `Motor 5` (`wrist_roll`) at `973` (`-pi/2 rad`), keeping the wrist camera on top, the **fixed jaw on the left**, and the **moving jaw opening to the right**.
* Setting `pitch_rad = 0.95` keeps the gripper jaws horizontal (`q1 + q2 + q3 = 0.95 rad`) for a clean perpendicular pinch on a vertical barrel.
* `so101_cli.py` automatically runs a closed-loop steady-state tick trim on `Motors 1..5` at the end of every `move-polar` command so gravity droop and static friction deadband are eliminated (`<= 3 ticks` steady-state error).

| Control Parameter | Physical Axis | Visual Effect and Pixel Jacobian | Primary Camera to Observe |
| :- | :- | :- | :- |
| **`pan_tick` (`Motor 1`)** | Base azimuth yaw | **Decreasing `pan_tick`** (`-10 ticks`) rotates the arm counter-clockwise, shifting the marker **RIGHT (`+36 px` in `+x_cam2`)** into the open jaw gap. **Increasing `pan_tick`** (`+10 ticks`) rotates clockwise, shifting the marker **LEFT (`-36 px` in `-x_cam2`)** toward the left fixed jaw. | `Camera 2` (`wrist_cam`) |
| **`z_m` (Elevation)** | Jaw height above table | Raising `z_m` moves the jaw tips **UP (`-y_cam1`)** in `Camera 1`. Because `Camera 2` looks diagonally forward and down (`pitch_rad = 0.95`), descending vertically (`-1 mm` in `z_m`) at constant `reach_m` shifts the marker **UP (`-7.0 px` in `-y_cam2`)** in `Camera 2`. | `Camera 1` (`side_cam`) and `Camera 2` |
| **`reach_m` (Radial reach)** | Distance from base axis | Increasing `reach_m` (`+1 mm`) moves the jaws forward along the ray defined by `pan_tick` (**LEFT / `-x_cam1`** in `Camera 1`) and shifts the marker **DOWN (`+7.9 px` in `+y_cam2`)** toward the jaws in `Camera 2`. | `Camera 1` and `Camera 2` |
| **`gripper_tick` (`Motor 6`)** | Pinch jaw | **`3145` = Wide Open**. **`2035` = Closed Empty (Sighting Pose)**. **`1700` = Commanded Firm Grasp** (`Reg 48 = 750`, `Reg 34 = 75%`). | `Reg 56`, `Reg 60`, and `Camera 2` |

***

## 2. Five Physical Rules for 100% Repeatable Grasping

Before moving the arm, internalize these five physical invariants of the SO-ARM101 gripper and camera geometry. View the linked reference images in `references/` to verify what each state looks like.

### Rule 1: The Grasp-Height Rule (Grasp `1 to 2 cm` Below the Top Cap at `z_m = 0.048 to 0.053 m`)
* **Never approach near the bottom of the marker (`z_m < 0.035 m`).** When the jaws are low near the holder, the upper `8 cm` of the marker sticks up right in front of the `Camera 2` lens, creating a giant out-of-focus black shadow that hides the jaws (see [bad_too_low_occlusion_cam2.png](references/bad_too_low_occlusion_cam2.png)).
* **Always set the final grasp height `z_m` (`0.048 to 0.053 m`) so the jaw tips sit `1 to 2 cm` below the top cap of the marker in `Camera 1`** (see [good_top_height_cam1.png](references/good_top_height_cam1.png)). At this height:
  1. Nothing extends above the jaws to block `Camera 2` or create severe vertical parallax.
  2. The circular top cap of the marker in `Camera 2` is at the exact height of the flat grey rubber pads.
  3. The jaws stay `4+ cm` above the figurine or cup holder, preventing collisions with the holder.

### Rule 2: The Single-Acting Jaw Rule (Keep `x_cam2` in `868..955` to Clear the Left Fixed Jaw)
* In `Camera 2`, the **left jaw is fixed** (`x_cam2 = 800..868`) and the **right jaw is the pivoting finger** that swings open to the right (`x_cam2 > 955`). When the gripper is wide open (`3145`), the moving jaw swings out of view to the bottom-right, leaving only the fixed jaw visible on the left.
* **Never align the marker directly in front of the black plastic tip of the fixed jaw (`x_cam2 < 868`).** If you advance or descend with the marker at `x_cam2 < 868`, the fixed jaw will hit the marker and push it aside (see [bad_fixed_jaw_hit_cam2.png](references/bad_fixed_jaw_hit_cam2.png)).
* **Always adjust `pan_tick` so the circular top cap of the marker sits at `x_cam2 = 885..935` inside the green `FLAT PAD ZONE (x=868..955, y=730..830)` box** drawn on every `*_cam_2_grid.png` image (see [good_sight_alignment_cam2.png](references/good_sight_alignment_cam2.png)).
  * If `x_cam2 < 880` (too close to the left fixed jaw), **decrease `pan_tick`** (`-10 ticks` shifts the marker `+36 px` right).
  * If `x_cam2 > 945` (too far right), **increase `pan_tick`** (`+10 ticks` shifts the marker `-36 px` left).

### Rule 3: The Flat Pad Zone vs. Rear V-Gap Rule (`y_cam2 = 730..830`, Never `> 835`)
* The gripping surface of the SO-ARM101 jaw is the **flat grey high-friction rubber pad** (`y_cam2 = 730..830` in `Camera 2`, marked by the green `FLAT PAD ZONE` box). Immediately behind the flat rubber pads (`y_cam2 > 835`, marked by the red `V-GAP LIMIT` line) is a wider V-shaped cutout near the jaw pivot.
* **Never over-advance `reach_m` past the red `V-GAP LIMIT (y=835)`.** If the marker barrel enters `y_cam2 > 835`, it slips into the rear V-gap where the jaws close nearly empty (`Reg 56 <= 2058`, low load) and the marker wobbles loosely.
* **Always position `reach_m` so that at grasp height (`z_m = 0.048 to 0.053 m`), the circular marker cap sits inside the green `FLAT PAD ZONE (x=868..955, y=730..830)`.**

### Rule 4: The `Camera 1` + `Camera 2` Parallax Rule When Descending from Hover
* Because `Camera 2` is mounted on top of the wrist looking diagonally forward and down (`pitch_rad = 0.95`), descending vertically (`-z_m`) at constant `reach_m` shifts the marker **UP (`-y_cam2`)** in `Camera 2` at **`-7.0 px` per `-1 mm` of vertical descent**.
* Therefore, if you inspect alignment while hovering `30 mm` above the grasp plane (`z_m = 0.080 to 0.085 m`):
  1. In `Camera 2`, the marker top cap should appear lower in the frame (`y_cam2 = 880..910`) so that descending `-30 mm` to `z_m = 0.050 to 0.053 m` shifts it up by `-210 px` directly into `y_cam2 = 730..830` (`FLAT PAD ZONE`).
  2. Always cross-check `Camera 1` (`side_cam`) before descending: verify that the flat grey rubber pads on the upper jaw sit directly above the vertical marker barrel (or approach from `1.5 to 2.5 cm` short of the barrel at `z_m = 0.050 m` and then advance `reach_m` horizontally into the `FLAT PAD ZONE`).

### Rule 5: The Closed-Jaw Self-Correction Rule (Inspect Before Opening on a Miss)
* After clamping (`gripper clamp` or `move-polar ... 1700`), verify both servo registers in the JSON output:
  * **`2075 <= Reg 56 <= 2150` and `Reg 60 >= 200` (`grasp_state == "MARKER_HELD"`):** The `10 mm` marker barrel is firmly clamped between the flat grey rubber pads.
  * **`Reg 56 <= 2055` (`grasp_state == "EMPTY_CLOSED"`):** The jaws closed empty (or deep in the V-gap if `Reg 56` is `2048..2058`).
* **If a miss occurs, DO NOT open the gripper in place near the marker.** Note that `so101_cli.py gripper open` automatically retracts `reach_m` by `-2.5 cm` with the jaws closed before opening whenever `reach_m > 0.210 m` and `z_m < 0.075 m`, preventing the pivoting right jaw from knocking the marker over.

***

## 3. Step-by-Step Closed-Loop Execution Procedure

### Step 1: Observe the Workspace and Locate the Marker
Run `status` to auto-discover the follower serial port, read current joint ticks, and capture grid-annotated frames from all three cameras (`cam_0`, `cam_1`, `cam_2`):
```bash
uv run python scripts/so101_cli.py status step1_observe
```
* If `status` reports `"FOLLOWER_USB_NOT_FOUND"` or `"SERVOS_NOT_RESPONDING"`, stop and ask the user to check the follower USB-C cable and 12V power supply.
* Inspect `outputs/live/step1_observe_cam_0_grid.png` and `outputs/live/step1_observe_cam_1_grid.png` to confirm the marker is upright in the holder and estimate its coarse azimuth (`pan_tick`, typically `2400 to 2530` when the holder is on the right side of the paper) and radial distance (`reach_m`, typically `0.180 to 0.205 m`).

### Step 2: Move to Pre-Approach Pose and Align Azimuth (`pan_tick`)
Move to a pre-approach pose (`2 cm` short of the marker radially at `z_m = 0.050 m`, or hovering above at `z_m = 0.080 m`, with `pitch_rad = 0.95` and gripper closed `2035` for sighting or open `3145`):
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> 0.180 0.050 0.95 3145 step2_align
```
*(Note: `so101_cli.py` automatically routes large `Motor 1` pans through a high-clearance overhead arc at `z >= 0.115 m` so the arm never sweeps low across the figurine or paper.)*

Inspect `outputs/live/step2_align_cam_1_grid.png` and `outputs/live/step2_align_cam_2_grid.png`:
1. In `Camera 1`, verify the jaw tips are `1 to 2 cm` below the top cap of the marker (`z_m = 0.048 to 0.053 m`, compare with [good_top_height_cam1.png](references/good_top_height_cam1.png)).
2. In `Camera 2`, check the horizontal `x_cam2` position of the circular marker top cap relative to the green `FLAT PAD ZONE (x=868..955, y=730..830)` box:
   * Target `x_cam2 = 885..935` (centered in the open gap to the right of the fixed jaw wall).
   * If `x_cam2 < 880` (marker too far left in `Camera 2`), **decrease `pan_tick`** (`-10 ticks` shifts the marker `+36 px` right).
   * If `x_cam2 > 940` (marker too far right in `Camera 2`), **increase `pan_tick`** (`+10 ticks` shifts the marker `-36 px` left).

### Step 3: Advance Radially into the `FLAT PAD ZONE (x=868..955, y=730..830)`
With `pan_tick` locked and gripper open (`3145`), advance `reach_m` so the marker barrel enters the green `FLAT PAD ZONE` between the flat grey rubber pads:
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> <grasp_reach_m> 0.050 0.95 3145 step3_enclose
```
Inspect `outputs/live/step3_enclose_cam_2_grid.png` and `outputs/live/step3_enclose_cam_1_grid.png`:
* Confirm in `Camera 2` that the marker barrel/cap sits squarely inside the green `FLAT PAD ZONE (x=868..955, y=730..830)` box and **above** the red `V-GAP LIMIT (y=835)` line. (Use `+1 mm reach_m = +7.9 px y_cam2` to fine-tune `reach_m` if needed.)
* Confirm in `Camera 1` that the flat grey rubber pads overlap the upper marker barrel `1 to 2 cm` below the top cap.

### Step 4: Clamp Between the Flat Rubber Pads and Verify `Reg 56` + `Reg 60`
Command the firm grasp (`1700`) via `gripper clamp` (which actuates only `Motor 6` while preserving the trimmed pose of `Motors 1..5`) or `move-polar`:
```bash
uv run python scripts/so101_cli.py gripper clamp step4_clamp
```
Inspect `outputs/live/step4_clamp_cam_2_grid.png` and the JSON telemetry:
* **Success criteria:**
  * `2075 <= gripper_present_position_reg56 <= 2150` (`grasp_state == "MARKER_HELD"`)
  * `gripper_present_load_reg60 >= 200`
  * In `Camera 2`, both left and right flat grey rubber pads are visibly pressed against the marker barrel inside the green `FLAT PAD ZONE (x=868..955, y=730..830)`.
* **If `Reg 56 <= 2058`:** The marker is either missed (`EMPTY_CLOSED`) or too deep in the rear V-gap (`y_cam2 > 835`). Retract `reach_m` with the gripper closed, open at a safe retracted reach, adjust `reach_m` or `pan_tick`, and repeat Step 3.

### Step 5: Lift Vertically Clear of the Holder
Once `Reg 56` (`2075..2150`), `Reg 60` (`>= 200`), and `Camera 2` confirm a flat-pad grasp, extract the marker straight up by raising `z_m` to `0.125 to 0.135 m` while keeping `pan_tick` locked and the gripper clamped (`1700`):
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> <grasp_reach_m> 0.130 0.85 1700 step5_lift
```
Inspect `outputs/live/step5_lift_cam_0_grid.png`, `outputs/live/step5_lift_cam_1_grid.png`, and `outputs/live/step5_lift_cam_2_grid.png` to confirm the bottom tip of the marker is completely clear of the figurine holder in mid-air and `gripper_present_position_reg56` remains in `[2075, 2150]`.
