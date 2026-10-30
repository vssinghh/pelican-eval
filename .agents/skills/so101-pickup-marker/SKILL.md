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

## 2. Six Physical Rules for 100% Repeatable Grasping

Before moving the arm, internalize these six physical invariants of the SO-ARM101 gripper and camera geometry. View the linked reference images in `references/` to verify what each state looks like.

### Rule 1: The Lower-Barrel Grasp Height and Leaning-Cup Compensation Rule (`z_m = 0.020 to 0.038 m`)
* **Target the lower white barrel (`2.0 to 3.5 cm` below the top cap, `0.8 to 1.5 cm` above the holder rim) rather than grasping high near the cap.** Grasping lower on the main cylindrical body gives the flat rubber pads maximum contact area, prevents the marker from pivoting near its top cap, and keeps the lower tip shorter below the jaws for stable drawing calibration.
* **Compensate for marker tilt in cup compartments when lowering `z_m`:** Inside a wide hexagonal cup cell, a marker leans at an angle (`~60 to 75 deg`). Because the barrel slopes diagonally from the cup rim up to the top cap, lowering `z_m` by `-6 mm` to grasp lower on the barrel requires two coordinated adjustments:
  1. **Increase `reach_m` by `+5 to +7 mm`**: Lowering `z_m` by `-6 mm` shifts the barrel UP (`-42 px` in `-y_cam2`) in the diagonally downward-looking `Camera 2` frame, AND the sloping barrel is physically further forward near the cup rim. Advancing `reach_m` by `+5 to +7 mm` (`+40 to +55 px` in `+y_cam2`) keeps the lower barrel section centered inside the `FLAT PAD ZONE (y_cam2 = 700..825)`.
  2. **Adjust `pan_tick` (`+10 to +20 ticks` if leaning right)** so the lower barrel cross-section at jaw height (not the upper red cap) aligns with the fixed left jaw pad (`x_cam2 = 868`).
* **Never confuse the open moving right jaw tip in `Camera 1` for the marker top cap.** When the gripper is open (`2500` to `3145`), the background moving right jaw projects `~80 px` higher (`y_cam1 ≈ 225..260`) than the foreground fixed left jaw (`y_cam1 ≈ 340..400`). Always measure jaw height using the **foreground fixed left jaw** (`y_cam1 ≈ 370..405` at `z_m = 0.020..0.026 m`) and verify that the red cap and upper white collar extend clearly `70 to 110 px` (`2.0 to 3.5 cm`) above the foreground fixed left jaw while the jaws stay `8 to 15 mm` above the cup rim (see [good_top_height_cam1.png](references/good_top_height_cam1.png)).

### Rule 2: The Single-Acting Jaw and Open-Throat Standoff Rule (`x_cam2 = 900..1020` Before Advancing)
* In `Camera 2`, the **left jaw is fixed** (`x_cam2 = 750..868`, tip extending forward to `y_cam2 = 565`) and the **right jaw is the pivoting finger** that swings open to the right (`x_cam2 > 955`). When the gripper is wide open (`3145`), the moving jaw swings out of view to the bottom-right, leaving only the fixed jaw visible on the left.
* **Never advance `reach_m` while the marker is in front of or grazing the fixed left jaw tip (`x_cam2 < 885` at `y_cam2 < 565`).** Even at `x_cam2 = 850..875`, advancing `reach_m` causes the rounded front tip of the fixed left jaw (`x_cam2 = 750..868`) to collide head-on with the marker barrel and push it forward/down underneath the jaw (see [bad_fixed_jaw_hit_cam2.png](references/bad_fixed_jaw_hit_cam2.png)).
* **Two-stage azimuth alignment in `Camera 2`:**
  1. **At pre-enclosure standoff (`y_cam2 < 565`)**: Adjust `pan_tick` so the marker barrel/cap sits well inside the **wide-open throat to the right of the fixed jaw tip (`x_cam2 = 900..1020`)** before advancing `reach_m`.
  2. **At enclosure reach (`y_cam2 = 730..830`)**: Advance `reach_m` (optionally with `+10..+15 ticks` of `pan_tick`) so the marker lands squarely inside the green `FLAT PAD ZONE (x=868..955, y=730..830)` box drawn on every `*_cam_2_grid.png` image (see [good_sight_alignment_cam2.png](references/good_sight_alignment_cam2.png)).
  * If `x_cam2 < 885` (too close to the left fixed jaw), **decrease `pan_tick`** (`-10 ticks` shifts the marker `+18..+36 px` right depending on reach).
  * If `x_cam2 > 965` once at enclosure depth (`y_cam2 = 730..830`), **increase `pan_tick`** (`+10 ticks` shifts the marker `-18..-36 px` left against the flat rubber pad).

### Rule 3: The Flat Pad Zone vs. Rear V-Gap Rule (`y_cam2 = 730..830`, Never `> 835`)
* The gripping surface of the SO-ARM101 jaw is the **flat grey high-friction rubber pad** (`y_cam2 = 730..830` in `Camera 2`, marked by the green `FLAT PAD ZONE` box). Immediately behind the flat rubber pads (`y_cam2 > 835`, marked by the red `V-GAP LIMIT` line) is a wider V-shaped cutout near the jaw pivot.
* **Never over-advance `reach_m` past the red `V-GAP LIMIT (y=835)`.** If the marker barrel enters `y_cam2 > 835`, it slips into the rear V-gap where the jaws close nearly empty (`Reg 56 <= 2058`, low load) and the marker wobbles loosely.
* **Always position `reach_m` so that at grasp height (`z_m = 0.048 to 0.053 m`), the circular marker cap sits inside the green `FLAT PAD ZONE (x=868..955, y=730..830)`.**

### Rule 4: The `Camera 1` + `Camera 2` Parallax Rule When Descending from Hover
* Because `Camera 2` is mounted on top of the wrist looking diagonally forward and down (`pitch_rad = 0.95`), descending vertically (`-z_m`) at constant `reach_m` shifts the marker **UP (`-y_cam2`)** in `Camera 2` at **`-7.0 px` per `-1 mm` of vertical descent**.
* Therefore, if you inspect alignment while hovering `30 mm` above the grasp plane (`z_m = 0.080 to 0.085 m`):
  1. In `Camera 2`, the marker top cap should appear lower in the frame (`y_cam2 = 880..910`) so that descending `-30 mm` to `z_m = 0.050 to 0.053 m` shifts it up by `-210 px` directly into `y_cam2 = 730..830` (`FLAT PAD ZONE`).
  2. Always cross-check `Camera 1` (`side_cam`) before descending: verify that the flat grey rubber pads on the upper jaw sit directly above the vertical marker barrel (or approach from `1.5 to 2.5 cm` short of the barrel at `z_m = 0.050 m` and then advance `reach_m` horizontally into the `FLAT PAD ZONE`).

### Rule 5: The Slow 3-Stage Closure and Fixed-Jaw Pad Pre-Contact Rule (`2500 -> 2310 -> 2150 -> 1700`)
* **Never snap the gripper directly from open (`3145` or `2500`) to `clamp` (`1700`) in a single step on a leaning marker inside a cup cell.** When a marker rests freely in a wide cup compartment, a single fast sweep of the pivoting right jaw can push or roll the barrel sideways or into the unpadded rear V-gap before the rubber pads engage.
* **Always align the marker's left edge flush against the fixed left jaw's grey rubber pad (`x_cam2 = 868`, `y_cam2 = 700..825`) with the gripper at `2500` before closing** (see [good_stage0_flush_left_pad_cam2.png](references/good_stage0_flush_left_pad_cam2.png)). When the marker barrel is already supported on the left by the stationary rubber pad, the closing right jaw pushes the barrel directly against a high-friction backstop instead of knocking it sideways.
* **Execute closure slowly (`2.0 s` ramp in `so101_cli.py`) across three verified stages:**
  1. **Stage 1 (`gripper 2310`)**: Pre-close the moving right jaw to `x_cam2 ≈ 1230` (`2 to 4 mm` from the right side of the barrel) and verify in `Camera 2` that the marker remains undisturbed inside `y_cam2 = 700..825` (see [good_stage1_soft_close_2310_cam2.png](references/good_stage1_soft_close_2310_cam2.png)).
  2. **Stage 2 (`gripper 2150`)**: Soft-contact close until the moving right jaw's rubber pad gently touches the right side of the barrel (`Reg 56 ≈ 2165..2200`, `Reg 60 ≈ 80..120`), capturing the barrel between both pads with zero impact deflection (see [good_stage2_touch_close_2150_cam2.png](references/good_stage2_touch_close_2150_cam2.png)).
  3. **Stage 3 (`gripper clamp` / `1700`)**: Apply full holding preload and verify `2060 <= Reg 56 <= 2350` and `Reg 60 >= 150` (`BARREL_HELD_IN_FLAT_PAD`, see [good_stage3_clamp_1700_cam2.png](references/good_stage3_clamp_1700_cam2.png)).

### Rule 6: The Closed-Jaw Self-Correction Rule (Inspect Before Opening on a Miss)
* After clamping (`gripper clamp` or `move-polar ... 1700`), verify both servo registers in the JSON output:
  * **`2060 <= Reg 56 <= 2650` and `Reg 60 >= 150` (`grasp_state == "BARREL_HELD_IN_FLAT_PAD"`):** The marker barrel (`8 mm` to `16 mm` diameter) is firmly clamped between the flat grey rubber pads (typically `Reg 56 = 2085..2110` for `10 mm` barrels or `2130..2350` for thicker `12 to 15 mm` barrels, with `Reg 60 = 200..450`).
  * **`1960 <= Reg 56 <= 2059` (`grasp_state == "BARREL_IN_REAR_V_GAP"` or `"PUSHED_INTO_INNER_V_GAP"`):** The marker slipped past `y_cam2 = 835` into the rear V-gap cutout.
  * **`Reg 56 <= 2055` and low load (`grasp_state == "EMPTY_CLOSED"`):** The jaws closed empty.
* **If a miss occurs, DO NOT open the gripper wide (`3145`) in place near the marker.** Open only to `2500` after retracting `reach_m` by `-20 to -25 mm` so the pivoting right jaw never sweeps into the marker or cup walls.

***

## 3. Step-by-Step Closed-Loop Execution Procedure

### Step 1: Observe the Workspace and Locate the Marker
Run `status` to auto-discover the follower serial port, read current joint ticks, and capture grid-annotated frames from all three cameras (`cam_0`, `cam_1`, `cam_2`):
```bash
uv run python scripts/so101_cli.py status step1_observe
```
* If `status` reports `"FOLLOWER_USB_NOT_FOUND"` or `"SERVOS_NOT_RESPONDING"`, stop and ask the user to check the follower USB-C cable and 12V power supply.
* Inspect `outputs/live/step1_observe_cam_0_grid.png` and `outputs/live/step1_observe_cam_1_grid.png` to confirm the marker is in the holder, identify which cup compartment holds it, and estimate its coarse azimuth (`pan_tick`, typically `2400 to 2530`) and radial distance (`reach_m`, typically `0.185 to 0.215 m`).

### Step 2: Move to Pre-Approach Standoff and Align Azimuth (`pan_tick`)
Move to a pre-approach standoff (`2.0 to 2.5 cm` short of the marker radially at lower-barrel height `z_m = 0.022 to 0.026 m` for cup holders or `0.036 to 0.045 m` for tall figurines, with `pitch_rad = 0.92..0.95` and gripper pre-narrowed to `2500`):
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> 0.185 0.024 0.92 2500 step2_align
```
*(Note: `so101_cli.py` automatically routes large `Motor 1` pans through a high-clearance overhead arc at `z >= 0.115 m` so the arm never sweeps low across the holder or paper.)*

Inspect `outputs/live/step2_align_cam_1_grid.png` and `outputs/live/step2_align_cam_2_grid.png`:
1. In `Camera 1`, verify the foreground fixed left jaw sits at the **lower white barrel** (`2.0 to 3.5 cm` below the top cap, `8 to 15 mm` above the cup rim).
2. In `Camera 2`, check the horizontal `x_cam2` position of the lower barrel section at jaw height (`y_cam2 = 500..650` at standoff):
   * Target `x_cam2 = 885..950` for the barrel's left edge so it clears the fixed left jaw tip (`x_cam2 = 868`, `y_cam2 = 565`) during radial advance.
   * If `x_cam2 < 880` (marker too far left in `Camera 2`), **decrease `pan_tick`** (`-10 ticks` shifts the marker `+36 px` right).
   * If `x_cam2 > 960` (marker too far right in `Camera 2`), **increase `pan_tick`** (`+10 ticks` shifts the marker `-36 px` left).

### Step 3: Advance Radially into the `FLAT PAD ZONE` and Seat Flush Against the Fixed Left Pad
Advance `reach_m` at lower-barrel height (`z_m = 0.020 to 0.025 m`) so the lower white barrel enters `y_cam2 = 700..825` (remembering that lower `z_m` on a leaning marker requires `+5 to +8 mm` more `reach_m` than grasping near the cap), then fine-tune `pan_tick` so the left edge of the barrel sits **flush against the fixed left jaw's grey rubber pad (`x_cam2 = 868`)**:
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> <grasp_reach_m> 0.022 0.92 2500 step3_enclose
```
Inspect `outputs/live/step3_enclose_cam_2_grid.png` and `outputs/live/step3_enclose_cam_1_grid.png`:
* Confirm in `Camera 2` that the lower marker barrel cross-section at the jaw plane sits squarely inside `y_cam2 = 700..825` (**above** the red `V-GAP LIMIT (y=835)` line) and its left edge touches the fixed left jaw pad (`x_cam2 = 868`).
* **3D Perspective Note for Lower-Barrel Grasps (`Camera 2` vs `Camera 1`)**: When grasping `2.0 to 3.5 cm` below the top cap (`z_m = 0.013 to 0.025 m`), the top cap of the marker extends **above** the top surface of the gripper jaws (`y_cam1 < y_jaws` in `Camera 1`). Because `Camera 2` is mounted on the wrist above the jaws looking downward at an angle, the top cap protruding above the jaws is closer to the `Camera 2` lens and projects across the bottom of the `Camera 2` frame (`y_cam2 = 850..1080`). Do not mistake that upper cap projection for the jaw-plane cross-section: always verify jaw height in `Camera 1` (`y_cam1` below the cap) and check the barrel's alignment between the two grey rubber pads specifically along rows `y_cam2 = 730..830` in `Camera 2`.
* Confirm in `Camera 1` that the foreground fixed left jaw overlaps the lower white barrel (`2.0 to 3.5 cm` below the cap) with `8 to 15 mm` clearance above the cup rim.

### Step 4: Execute 3-Stage Slow Closure (`2310 -> 2150 -> 1700`) and Verify `Reg 56` + `Reg 60`
Close the gripper slowly in three stages using `gripper` (which ramps over `2.0 s` and actuates only `Motor 6` while preserving the trimmed pose of `Motors 1..5`):
```bash
uv run python scripts/so101_cli.py gripper 2310 step4a_soft_close
uv run python scripts/so101_cli.py gripper 2150 step4b_touch_close
uv run python scripts/so101_cli.py gripper clamp step4c_clamp
```
Inspect `step4a_soft_close_cam_2_grid.png`, `step4b_touch_close_cam_2_grid.png`, and `step4c_clamp_cam_2_grid.png` along with the JSON telemetry:
* **After Stage 1 (`2310`)**: Confirm the moving right jaw pad has approached to `x_cam2 ≈ 1230` without shifting the marker.
* **After Stage 2 (`2150`)**: Confirm the moving right jaw pad gently touches the right side of the barrel (`Reg 56 ≈ 2165..2200`, `Reg 60 ≈ 80..120`) while the left side remains flush against `x_cam2 = 868`.
* **After Stage 3 (`clamp` / `1700`)**:
  * `2060 <= gripper_present_position_reg56 <= 2350` (`grasp_state == "BARREL_HELD_IN_FLAT_PAD"`)
  * `gripper_present_load_reg60 >= 150` (typically `200..450`)
  * In `Camera 2`, both left and right flat grey rubber pads are visibly pressed against the lower white barrel inside `y_cam2 = 700..825`.
* **If `Reg 56 <= 2059`:** The marker slipped into the rear V-gap (`y_cam2 > 835`) or was missed. Retract `reach_m` by `-20 mm` before opening to `2500`, reduce `reach_m` by `4 to 6 mm` at enclosure so `y_cam2 <= 820`, and repeat Step 3.

### Step 5: Lift Vertically Clear of the Holder
Once `Reg 56` (`2060..2650`), `Reg 60` (`>= 150`), and `Camera 2` confirm a flat-pad grasp, extract the marker straight up by raising `z_m` to `0.125 to 0.135 m` while keeping `pan_tick` locked and the gripper clamped (`1700`):
```bash
uv run python scripts/so101_cli.py move-polar <pan_tick> <grasp_reach_m> 0.130 0.85 1700 step5_lift
```
Inspect `outputs/live/step5_lift_cam_0_grid.png`, `outputs/live/step5_lift_cam_1_grid.png`, and `outputs/live/step5_lift_cam_2_grid.png` to confirm the bottom tip of the marker is completely clear of the figurine holder in mid-air and `gripper_present_position_reg56` remains in `[2060, 2650]`.

***

## 4. Failure Handling and Human-in-the-Loop Escalation

The horizontal pinch configuration (`wrist_roll = 973`) is designed strictly to grasp an **upright** marker from an intact holder. Never loop indefinitely or attempt to chase a fallen marker across the desk. Distinguish between minor alignment misses (autonomous retry) and unrecoverable workspace faults (human assistance).

### 4.1 When to Escalate Immediately to the Human Operator (Zero Blind Retries)
Stop autonomous retries immediately and ask the human for help if any of the following conditions appear in `Camera 0`, `Camera 1`, `Camera 2`, or the JSON telemetry:
1. **Holder Tilted, Knocked Over, or Lifted**: The figurine or cup holder is lying on its side, tilted noticeably, or being lifted/dragged along with the marker during `Step 5` extraction (for example, if the marker was wedged inside the figurine).
2. **Marker Fallen Flat, Dropped, or Missing**: The marker is lying horizontally on the table, has rolled out of camera view, or slipped out of the gripper during lift (`Reg 56` drops to `<= 2055` during `Step 5`).
3. **Marker Caught at an Unusable Angle**: After lifting (`Step 5`), `Camera 1` shows the marker skewed sharply (`> 20 degrees` from vertical) or caught by its top tip such that it cannot be calibrated perpendicular to the paper.
4. **Two Consecutive Grasp Misses (`>= 2` failed clamp attempts)**: Even if the marker and holder remain upright, if two closed-loop adjustments fail to achieve `2075 <= Reg 56 <= 2150` and `Reg 60 >= 200`, stop and ask the human rather than risking a collision on a third attempt.
5. **USB or Servo Fault**: `FOLLOWER_USB_NOT_FOUND`, `SERVOS_NOT_RESPONDING`, `Reg 56 = -1`, or servo overload trip.

### 4.2 Safe Park and Operator Prompt Protocol
Before asking the human to reach into the workspace, always clear the arm out of the way so the operator has unobstructed hand access and no wedged object is flung upward:

1. **Execute `safe-park`**:
   * If the marker is still in the holder, wedged, or dropped on the table (`release` mode opens the gripper in place first before lifting to the high park pose `[2048, 1000, 3000, 2600, 973, 3145]`):
     ```bash
     uv run python scripts/so101_cli.py safe-park release park_for_human
     ```
   * If the marker is already lifted into mid-air (`z_m >= 0.110 m`) but held at a bad angle and you want the arm to keep holding it until the human takes or straightens it (`keep` mode preserves the current gripper clamp while moving to the high park pose):
     ```bash
     uv run python scripts/so101_cli.py safe-park keep park_for_human
     ```
2. **Prompt the Human Clearly**:
   In your message to the user, include:
   * **What happened**: A brief diagnosis referencing `outputs/live/park_for_human_cam_0_grid.png` and `outputs/live/park_for_human_cam_1_grid.png` (for example: *"The blue figurine tilted during approach and the marker is now leaning at an angle"*).
   * **Safety confirmation**: Confirm that the arm has parked safely in the high home pose away from the holder.
   * **Requested action**: Specify the exact reset needed (for example: *"Please stand the blue figurine upright and place the marker vertically in its holder, then reply so I can resume"*).
3. **Resume from Step 1 (`status`)**:
   Once the human confirms that the holder and marker have been reset, always restart from **Step 1 (`status`)** to re-observe the workspace from scratch. Never reuse stale `pan_tick` or `reach_m` coordinates after a manual reset.
