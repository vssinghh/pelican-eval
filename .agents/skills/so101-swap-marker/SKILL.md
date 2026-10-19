---
name: so101-swap-marker
description: >-
  Release the currently held marker, prompt the human operator to place a new marker color
  into the holder, and chain into closed-loop pickup and paper contact calibration on the
  physical SO-ARM101 robot arm.
---

# SO-ARM101 Human-in-the-Loop Marker Color Swap (`so101-swap-marker`)

Use this skill whenever a drawing requires switching marker colors (for example, switching between `blue`, `red`, `green`, or `black`) or returning the marker after finishing a drawing.

***

## Step-by-Step Marker Swap Procedure

### Step 1: Move to the Safe Right-Side Handoff Pose
Move the arm to the right side of the workspace (`Motor 1 = 2480`), well clear of the drawing paper so releasing the marker cannot leave an accidental mark on the sheet:
```bash
uv run python scripts/so101_cli.py move-ticks 2480 1350 2800 1950 973 1700 swap_handoff
```

### Step 2: Open the Gripper to Release the Current Marker
Open the gripper wide (`tick = 3145`):
```bash
uv run python scripts/so101_cli.py gripper open swap_released
```
Inspect `outputs/live/swap_released_cam_1.png` and confirm `grasp_state == "OPEN"` (`gripper_present_position_reg56 >= 2500`).

### Step 3: Retract Inward So the Holder Area Is Clear for the User's Hand
Retract the forearm inward (`Motor 2 = 1150`, `Motor 3 = 2950`, gripper open `3145`) so the user can easily place the new marker into the holder without bumping the robot jaws:
```bash
uv run python scripts/so101_cli.py move-ticks 2480 1150 2950 1950 973 3145 swap_parked
```

### Step 4: Prompt the User for the Next Marker Color
Ask the user clearly:
> "Please place the **<COLOR>** marker vertically (tip down) into the holder and let me know when it is ready."

Stop calling tools and wait for the user's reply.

### Step 5: Chain into `so101-pickup-marker` and `so101-calibrate-paper-contact`
Once the user confirms the new marker is in the holder:
1. Follow [so101-pickup-marker](../so101-pickup-marker/SKILL.md) to visually locate the holder in `Camera 0` and `Camera 1`, align `pinch_site`, grasp the barrel, and verify `2075 <= Reg 56 <= 2135`.
2. Follow [so101-calibrate-paper-contact](../so101-calibrate-paper-contact/SKILL.md) to re-verify vertical `90.0°` pitch and exact paper contact height `z_center` for the newly grasped marker before executing strokes with [so101-draw-and-inspect](../so101-draw-and-inspect/SKILL.md).
