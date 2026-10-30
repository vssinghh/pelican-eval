"""Closed-loop primitive CLI for the physical SO-ARM101 follower arm.

Provides sensor readouts (joint ticks, FK pinch_site pose, gripper load/position,
and annotated camera frames) and smooth interpolated joint/IK motion primitives
without hardcoded workspace coordinates.

Usage:
    uv run python scripts/so101_cli.py status [out_prefix]
    uv run python scripts/so101_cli.py open-observe [out_prefix]
    uv run python scripts/so101_cli.py move-ticks <t1> <t2> <t3> <t4> <t5> <t6> [out_prefix]
    uv run python scripts/so101_cli.py move-polar <pan_tick> <reach_m> <z_m> <pitch_rad> <gripper_tick> [out_prefix]
    uv run python scripts/so101_cli.py nudge-polar <d_pan_tick> <d_reach_m> <d_z_m> <d_pitch_rad> [out_prefix]
    uv run python scripts/so101_cli.py move-ik <x> <y> <z> <pitch_rad> <gripper_tick> [out_prefix]
    uv run python scripts/so101_cli.py gripper <open|clamp|tick> [out_prefix]
    uv run python scripts/so101_cli.py lift [dz_m] [out_prefix]
    uv run python scripts/so101_cli.py place <pan_tick> <reach_m> <z_insert_m> [out_prefix]
    uv run python scripts/so101_cli.py draw-strokes <json_string_or_path> [out_prefix]
    uv run python scripts/so101_cli.py safe-park [release|keep] [out_prefix]
    uv run python scripts/so101_cli.py relax
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import serial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.ik_solver import SO101IKSolver

DEFAULT_PORT = "/dev/tty.usbmodem5B790143431"
KNOWN_LEADER_SUBSTR = "5B8E116931"
BAUD = 1_000_000
ZERO_OFFSETS = np.array([2027.5, 1851.0, 2035.0, 1930.0, 1997.0, 2048.0], dtype=np.float64)
RAD_PER_TICK = (2.0 * np.pi) / 4096.0


def open_robot_serial() -> serial.Serial:
    """Open the SO-ARM101 follower serial port, excluding the leader arm port."""
    all_usb = sorted(glob.glob("/dev/tty.usbmodem*"))
    candidates: list[str] = []
    if Path(DEFAULT_PORT).exists():
        candidates.append(DEFAULT_PORT)
    for p in all_usb:
        if p not in candidates and KNOWN_LEADER_SUBSTR not in p:
            candidates.append(p)

    if not candidates:
        leader_only = [p for p in all_usb if KNOWN_LEADER_SUBSTR in p]
        msg = (
            "Leader arm USB is connected, but Follower arm USB (/dev/tty.usbmodem5B790143431) "
            "is missing from the USB tree. Please reseat the USB-C cable on the follower arm board."
            if leader_only
            else "No /dev/tty.usbmodem* device found. Please check the SO-ARM101 USB cable."
        )
        print(
            json.dumps(
                {
                    "error": "FOLLOWER_USB_NOT_FOUND",
                    "detected_usb_ports": all_usb,
                    "message": msg,
                },
                indent=2,
            )
        )
        sys.exit(2)

    errors: list[str] = []
    for port in candidates:
        try:
            ser = serial.Serial(port, BAUD, timeout=0.05)
            if read_u16(ser, 1, 56) >= 0:
                return ser
            ser.close()
        except (OSError, serial.SerialException) as exc:
            errors.append(f"{port}: {exc}")

    print(
        json.dumps(
            {
                "error": "SERVOS_NOT_RESPONDING",
                "candidates_checked": candidates,
                "details": errors,
                "message": (
                    "Follower USB serial port found, but Motors 1..6 did not respond. "
                    "Please check that the 12V power supply is switched on."
                ),
            },
            indent=2,
        )
    )
    sys.exit(2)


def _checksum(packet_body: list[int]) -> int:
    return (~sum(packet_body)) & 0xFF


def read_u16(ser: serial.Serial, motor_id: int, reg: int) -> int:
    body = [motor_id, 4, 2, reg, 2]
    ser.reset_input_buffer()
    ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
    resp = ser.read(8)
    if len(resp) == 8 and resp[0] == 0xFF and resp[1] == 0xFF:
        return int(resp[5] | (resp[6] << 8))
    return -1


def write_u8(ser: serial.Serial, motor_id: int, reg: int, val: int) -> None:
    body = [motor_id, 4, 3, reg, val & 0xFF]
    ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
    ser.read(6)


def write_u16(ser: serial.Serial, motor_id: int, reg: int, val: int) -> None:
    lo = val & 0xFF
    hi = (val >> 8) & 0xFF
    body = [motor_id, 5, 3, reg, lo, hi]
    ser.write(bytes([0xFF, 0xFF, *body, _checksum(body)]))
    ser.read(6)


def read_all_ticks(ser: serial.Serial) -> list[int]:
    ticks: list[int] = []
    for m in range(1, 7):
        val = read_u16(ser, m, 56)
        if val < 0:
            time.sleep(0.01)
            val = read_u16(ser, m, 56)
        ticks.append(val)
    return ticks


def ticks_to_rad(ticks: list[int]) -> np.ndarray:
    return (np.array(ticks[:5], dtype=np.float64) - ZERO_OFFSETS[:5]) * RAD_PER_TICK


def rad_to_ticks(q_rad: np.ndarray) -> list[int]:
    raw = np.round(ZERO_OFFSETS[:5] + (q_rad[:5] / RAD_PER_TICK)).astype(int)
    return [int(np.clip(v, 200, 3900)) for v in raw]


def compute_fk(solver: SO101IKSolver, q_rad: np.ndarray) -> tuple[np.ndarray, float]:
    pos = solver.forward_kinematics(q_rad[:5])
    pitch_rad = float(q_rad[1] + q_rad[2] + q_rad[3])
    return pos, pitch_rad


def _interpolate_ticks(
    ser: serial.Serial,
    start_ticks: list[int],
    end_ticks: list[int],
    steps: int,
    dt: float,
) -> None:
    start = np.array(start_ticks, dtype=np.float64)
    end = np.array(end_ticks, dtype=np.float64)
    for s in range(1, steps + 1):
        alpha = s / float(steps)
        w = 0.5 * (1.0 - np.cos(np.pi * alpha))
        interp = np.round((1.0 - w) * start + w * end).astype(int)
        for m in range(1, 7):
            write_u16(ser, m, 42, int(interp[m - 1]))
        time.sleep(dt)


def smooth_move_ticks(
    ser: serial.Serial,
    target_ticks: list[int],
    steps: int = 45,
    dt: float = 0.025,
) -> list[int]:
    cur_ticks = read_all_ticks(ser)
    for m in range(1, 6):
        if cur_ticks[m - 1] < 0:
            raise RuntimeError(f"Failed to read Motor {m} Present_Position (Reg 56)")
        write_u16(ser, m, 48, 800)
        write_u8(ser, m, 40, 1)
    # Configure Motor 6 (gripper) for continuous holding torque without tripping overload comparator
    target_ticks = list(target_ticks)
    write_u8(ser, 6, 55, 0)
    write_u8(ser, 6, 34, 75)
    write_u8(ser, 6, 36, 95)
    write_u8(ser, 6, 55, 1)
    write_u16(ser, 6, 48, 450)
    if int(cur_ticks[5]) < 2600 and int(target_ticks[5]) <= 1850:
        hold_goal = max(int(target_ticks[5]), int(cur_ticks[5]) - 110)
        if read_u16(ser, 6, 60) == 0:
            write_u8(ser, 6, 40, 0)
            write_u16(ser, 6, 42, int(cur_ticks[5]))
            time.sleep(0.04)
            write_u8(ser, 6, 40, 1)
            time.sleep(0.02)
        else:
            write_u8(ser, 6, 40, 1)
        write_u16(ser, 6, 42, hold_goal)
        cur_ticks[5] = hold_goal
        target_ticks[5] = hold_goal
    else:
        write_u8(ser, 6, 40, 1)

    pan_delta = abs(int(target_ticks[0]) - int(cur_ticks[0]))
    max_lift_tick = max(int(cur_ticks[1]), int(target_ticks[1]))
    max_elbow_tick = max(int(cur_ticks[2]), int(target_ticks[2]))
    if pan_delta > 80 and (max_lift_tick > 1150 or max_elbow_tick > 2750):
        # High-clearance overhead arc (z >= 0.115 m):
        # 1) Lift vertically at current pan if starting low
        # 2) Pan Motor 1 at high elevation (shoulder_lift <= 1050, elbow_flex <= 2650)
        # 3) Descend vertically at target pan
        wp = list(cur_ticks)
        if int(cur_ticks[1]) > 1150 or int(cur_ticks[2]) > 2750:
            safe_up_start = [
                int(cur_ticks[0]),
                min(int(cur_ticks[1]), 1050),
                min(int(cur_ticks[2]), 2650),
                2450,
                973,
                int(cur_ticks[5]),
            ]
            _interpolate_ticks(ser, wp, safe_up_start, steps=max(20, steps // 2), dt=dt)
            time.sleep(0.08)
            wp = safe_up_start
        safe_up_end = [
            int(target_ticks[0]),
            min(int(target_ticks[1]), 1050),
            min(int(target_ticks[2]), 2650),
            int(target_ticks[3]),
            973,
            int(target_ticks[5]),
        ]
        _interpolate_ticks(ser, wp, safe_up_end, steps=max(25, steps // 2), dt=dt)
        time.sleep(0.08)
        _interpolate_ticks(ser, safe_up_end, target_ticks, steps=max(20, steps // 2), dt=dt)
    else:
        _interpolate_ticks(ser, cur_ticks, target_ticks, steps=steps, dt=dt)
    time.sleep(0.20)

    # Closed-loop steady-state tick trim on Motors 1..5 to eliminate gravity sag and static friction deadband
    cmd_ticks = [int(v) for v in target_ticks]
    for _ in range(2):
        actual = read_all_ticks(ser)
        if any(actual[m] < 0 for m in range(5)):
            break
        errs = [int(target_ticks[m]) - int(actual[m]) for m in range(5)]
        if max(abs(e) for e in errs) <= 3:
            break
        next_cmd = list(cmd_ticks)
        for m in range(5):
            if abs(errs[m]) > 2:
                corr = int(np.clip( np.round(0.75 * errs[m]), -30, 30 ))
                next_cmd[m] = int(np.clip(cmd_ticks[m] + corr, int(target_ticks[m]) - 55, int(target_ticks[m]) + 55))
        _interpolate_ticks(ser, cmd_ticks, next_cmd, steps=8, dt=0.020)
        cmd_ticks = next_cmd
        time.sleep(0.12)
    return cmd_ticks


def draw_grid_overlay(frame: np.ndarray, step_px: int = 100, cam_idx: int = -1) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    for x in range(0, w, step_px):
        color = (0, 255, 255) if x % 200 == 0 else (140, 140, 140)
        cv2.line(annotated, (x, 0), (x, h), color, 1)
        if x > 0:
            cv2.putText(
                annotated,
                str(x),
                (x + 3, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
    for y in range(0, h, step_px):
        color = (0, 255, 255) if y % 200 == 0 else (140, 140, 140)
        cv2.line(annotated, (0, y), (w, y), color, 1)
        if y > 0:
            cv2.putText(
                annotated,
                str(y),
                (5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
    if cam_idx == 2:
        # Fixed-jaw flat grey rubber pad target zone (x=868..955, y=730..830) and rear V-gap limit (y=835)
        cv2.rectangle(annotated, (868, 730), (955, 830), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "FLAT PAD ZONE (x=868..955, y=730..830)",
            (865, 720),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
        )
        cv2.line(annotated, (820, 835), (980, 835), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            "V-GAP LIMIT (y=835)",
            (985, 840),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 255),
            2,
        )
    return annotated


def capture_cameras(out_prefix: str, out_dir: Path) -> list[str]:
    saved: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_idx in (0, 1, 2):
        cap = cv2.VideoCapture(cam_idx)
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            raw_path = out_dir / f"{out_prefix}_cam_{cam_idx}.png"
            grid_path = out_dir / f"{out_prefix}_cam_{cam_idx}_grid.png"
            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(str(grid_path), draw_grid_overlay(frame, cam_idx=cam_idx))
            saved.append(str(raw_path))
            saved.append(str(grid_path))
    return saved


def report_status(
    ser: serial.Serial,
    solver: SO101IKSolver,
    out_prefix: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, object]:
    ticks = read_all_ticks(ser)
    q_rad = ticks_to_rad(ticks)
    pos, pitch_rad = compute_fk(solver, q_rad)
    g_pos = ticks[5]
    g_load = read_u16(ser, 6, 60)

    if 2060 <= g_pos <= 2650:
        grasp_state = "BARREL_HELD_IN_FLAT_PAD"
    elif 2056 <= g_pos <= 2059:
        grasp_state = "BARREL_IN_REAR_V_GAP"
    elif g_pos <= 2055:
        grasp_state = "EMPTY_CLOSED"
    elif g_pos >= 2700:
        grasp_state = "OPEN"
    else:
        grasp_state = "PARTIAL_OR_TRANSITION"

    info: dict[str, object] = {
        "ticks": ticks,
        "q_rad": [round(float(v), 4) for v in q_rad],
        "pinch_site_xyz_m": [round(float(v), 4) for v in pos],
        "radial_reach_m": round(float(np.hypot(pos[0], pos[1])), 4),
        "pitch_rad": round(pitch_rad, 4),
        "gripper_present_position_reg56": g_pos,
        "gripper_present_load_reg60": g_load,
        "grasp_state": grasp_state,
    }
    if out_prefix and out_dir is not None:
        info["saved_images"] = capture_cameras(out_prefix, out_dir)
    print(json.dumps(info, indent=2))
    return info


def solve_polar_ticks(
    solver: SO101IKSolver,
    pan_tick: int,
    reach_m: float,
    z_m: float,
    pitch_rad: float = 0.95,
) -> list[int]:
    """Solve for [t1..t5] given exact Motor 1 pan_tick, horizontal reach_m, height z_m, and pitch_rad."""
    q0 = float((pan_tick - ZERO_OFFSETS[0]) * RAD_PER_TICK)
    q1, q2 = -0.40, 1.25
    eps = 1e-4
    for _ in range(80):
        q3 = pitch_rad - q1 - q2
        pos = solver.forward_kinematics(np.array([q0, q1, q2, q3, -np.pi / 2.0]))
        r_cur = float(np.hypot(pos[0], pos[1]))
        z_cur = float(pos[2])
        err = np.array([reach_m - r_cur, z_m - z_cur], dtype=np.float64)
        if float(np.linalg.norm(err)) < 1e-4:
            break
        p1 = solver.forward_kinematics(
            np.array([q0, q1 + eps, q2, pitch_rad - (q1 + eps) - q2, -np.pi / 2.0])
        )
        p2 = solver.forward_kinematics(
            np.array([q0, q1, q2 + eps, pitch_rad - q1 - (q2 + eps), -np.pi / 2.0])
        )
        j_mat = np.array(
            [
                [(np.hypot(p1[0], p1[1]) - r_cur) / eps, (np.hypot(p2[0], p2[1]) - r_cur) / eps],
                [(p1[2] - z_cur) / eps, (p2[2] - z_cur) / eps],
            ],
            dtype=np.float64,
        )
        dq = j_mat.T @ np.linalg.solve(j_mat @ j_mat.T + 1e-5 * np.eye(2), err)
        dq = np.clip(dq, -0.15, 0.15)
        q1 = float(np.clip(q1 + dq[0], -1.65, 1.65))
        q2 = float(np.clip(q2 + dq[1], 0.08, 1.65))
    q3 = float(np.clip(pitch_rad - q1 - q2, -1.65, 1.520))
    if pitch_rad - q1 - q2 > 1.520:
        for _ in range(40):
            pos = solver.forward_kinematics(np.array([q0, q1, q2, q3, -np.pi / 2.0]))
            r_cur = float(np.hypot(pos[0], pos[1]))
            z_cur = float(pos[2])
            err = np.array([reach_m - r_cur, z_m - z_cur], dtype=np.float64)
            if float(np.linalg.norm(err)) < 1e-4:
                break
            p1 = solver.forward_kinematics(
                np.array([q0, q1 + eps, q2, q3, -np.pi / 2.0])
            )
            p2 = solver.forward_kinematics(
                np.array([q0, q1, q2 + eps, q3, -np.pi / 2.0])
            )
            j_mat = np.array(
                [
                    [(np.hypot(p1[0], p1[1]) - r_cur) / eps, (np.hypot(p2[0], p2[1]) - r_cur) / eps],
                    [(p1[2] - z_cur) / eps, (p2[2] - z_cur) / eps],
                ],
                dtype=np.float64,
            )
            dq = j_mat.T @ np.linalg.solve(j_mat @ j_mat.T + 1e-5 * np.eye(2), err)
            dq = np.clip(dq, -0.15, 0.15)
            q1 = float(np.clip(q1 + dq[0], -1.65, 1.65))
            q2 = float(np.clip(q2 + dq[1], 0.08, 1.65))
    ticks = rad_to_ticks(np.array([q0, q1, q2, q3, -np.pi / 2.0]))
    ticks[0] = int(pan_tick)
    ticks[4] = 973
    return ticks


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    solver = SO101IKSolver(site_name="pinch_site")
    ser = open_robot_serial()
    default_out = Path(__file__).resolve().parent.parent / "outputs" / "live"

    try:
        if cmd == "status":
            prefix = sys.argv[2] if len(sys.argv) >= 3 else "status"
            report_status(ser, solver, prefix, default_out)

        elif cmd == "open-observe":
            prefix = sys.argv[2] if len(sys.argv) >= 3 else "observe"
            target_ticks = solve_polar_ticks(solver, 2450, 0.140, 0.055, 0.95)
            target_ticks.append(3145)
            smooth_move_ticks(ser, target_ticks)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "move-ticks":
            if len(sys.argv) < 8:
                raise ValueError("Usage: move-ticks <t1> <t2> <t3> <t4> <t5> <t6> [out_prefix]")
            target = [int(sys.argv[i]) for i in range(2, 8)]
            prefix = sys.argv[8] if len(sys.argv) >= 9 else "move_ticks"
            smooth_move_ticks(ser, target)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "move-polar":
            if len(sys.argv) < 7:
                raise ValueError(
                    "Usage: move-polar <pan_tick> <reach_m> <z_m> <pitch_rad> <gripper_tick> [out_prefix]"
                )
            pan_tick = int(sys.argv[2])
            reach_m = float(sys.argv[3])
            z_m = float(sys.argv[4])
            pitch_rad = float(sys.argv[5])
            g_tick = int(sys.argv[6])
            prefix = sys.argv[7] if len(sys.argv) >= 8 else "move_polar"
            target_ticks = solve_polar_ticks(solver, pan_tick, reach_m, z_m, pitch_rad)
            target_ticks.append(g_tick)
            smooth_move_ticks(ser, target_ticks)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "nudge-polar":
            if len(sys.argv) < 6:
                raise ValueError(
                    "Usage: nudge-polar <d_pan_tick> <d_reach_m> <d_z_m> <d_pitch_rad> [out_prefix]"
                )
            d_pan = int(sys.argv[2])
            d_reach = float(sys.argv[3])
            d_z = float(sys.argv[4])
            d_pitch = float(sys.argv[5])
            prefix = sys.argv[6] if len(sys.argv) >= 7 else "nudge_polar"
            cur_ticks = read_all_ticks(ser)
            cur_q = ticks_to_rad(cur_ticks)
            pos, cur_pitch = compute_fk(solver, cur_q)
            cur_reach = float(np.hypot(pos[0], pos[1]))
            target_ticks = solve_polar_ticks(
                solver,
                int(cur_ticks[0]) + d_pan,
                cur_reach + d_reach,
                float(pos[2]) + d_z,
                cur_pitch + d_pitch,
            )
            target_ticks.append(int(cur_ticks[5]))
            smooth_move_ticks(ser, target_ticks)
            report_status(ser, solver, prefix, default_out)

        elif cmd in ("move-ik", "move-xyz"):
            if len(sys.argv) < 7:
                raise ValueError(
                    f"Usage: {cmd} <x> <y> <z> <pitch_rad> <gripper_tick> [out_prefix]"
                )
            x, y, z, pitch = (float(sys.argv[i]) for i in range(2, 6))
            g_tick = int(sys.argv[6])
            prefix = sys.argv[7] if len(sys.argv) >= 8 else "move_ik"
            cur_ticks = read_all_ticks(ser)
            cur_q = ticks_to_rad(cur_ticks)
            ik_res = solver.solve(
                target_xyz=np.array([x, y, z], dtype=np.float64),
                target_pitch=pitch,
                q_init=cur_q,
                exact_pitch=True,
            )
            target_ticks = rad_to_ticks(ik_res.q_5d)
            target_ticks[4] = 973  # Keep wrist_roll horizontal (-pi/2 rad)
            target_ticks.append(g_tick)
            smooth_move_ticks(ser, target_ticks)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "lift":
            dz = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.095
            prefix = sys.argv[3] if len(sys.argv) >= 4 else "lift"
            cur_ticks = read_all_ticks(ser)
            cur_q = ticks_to_rad(cur_ticks)
            pos, _ = compute_fk(solver, cur_q)
            cur_reach = float(np.hypot(pos[0], pos[1]))
            target_ticks = solve_polar_ticks(
                solver,
                int(cur_ticks[0]),
                cur_reach,
                float(pos[2]) + dz,
                0.85,
            )
            target_ticks.append(1700)
            smooth_move_ticks(ser, target_ticks)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "place":
            if len(sys.argv) < 5:
                raise ValueError("Usage: place <pan_tick> <reach_m> <z_insert_m> [out_prefix]")
            pan_tick = int(sys.argv[2])
            reach_m = float(sys.argv[3])
            z_insert_m = float(sys.argv[4])
            prefix = sys.argv[5] if len(sys.argv) >= 6 else "place"
            above_ticks = solve_polar_ticks(solver, pan_tick, reach_m, max(z_insert_m + 0.080, 0.125), 0.85)
            above_ticks.append(1700)
            smooth_move_ticks(ser, above_ticks, steps=35, dt=0.025)
            insert_ticks = solve_polar_ticks(solver, pan_tick, reach_m, z_insert_m, 0.95)
            insert_ticks.append(1700)
            smooth_move_ticks(ser, insert_ticks, steps=35, dt=0.025)
            # Open gripper in place
            cur_ticks = read_all_ticks(ser)
            start_g = int(cur_ticks[5])
            for s in range(1, 21):
                alpha = s / 20.0
                w = 0.5 * (1.0 - np.cos(np.pi * alpha))
                g_interp = int(round((1.0 - w) * start_g + w * 3145))
                write_u16(ser, 6, 42, g_interp)
                time.sleep(0.020)
            time.sleep(0.25)
            retract_ticks = solve_polar_ticks(solver, pan_tick, max(0.145, reach_m - 0.045), z_insert_m + 0.020, 0.95)
            retract_ticks.append(3145)
            smooth_move_ticks(ser, retract_ticks, steps=35, dt=0.025)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "gripper":
            if len(sys.argv) < 3:
                raise ValueError("Usage: gripper <open|clamp|tick> [out_prefix]")
            arg = sys.argv[2]
            if arg == "open":
                g_tick = 3145
            elif arg == "clamp":
                g_tick = 1700
            else:
                g_tick = int(arg)
            prefix = sys.argv[3] if len(sys.argv) >= 4 else f"gripper_{arg}"
            cur_ticks = read_all_ticks(ser)
            if g_tick >= 2500 and cur_ticks[5] < 2500:
                cur_q = ticks_to_rad(cur_ticks)
                pos, pitch_rad = compute_fk(solver, cur_q)
                r_cur = float(np.hypot(pos[0], pos[1]))
                if r_cur > 0.210 and float(pos[2]) < 0.075:
                    # Safe retract by 2.5 cm with jaws closed before opening so we never knock the marker
                    retract_ticks = solve_polar_ticks(
                        solver,
                        int(cur_ticks[0]),
                        max(0.190, r_cur - 0.025),
                        float(pos[2]),
                        float(np.clip(pitch_rad, 0.85, 1.05)),
                    )
                    retract_ticks.append(int(cur_ticks[5]))
                    smooth_move_ticks(ser, retract_ticks, steps=25, dt=0.025)
                    cur_ticks = read_all_ticks(ser)
            # Actuate ONLY Motor 6 so Motors 1..5 keep their gravity-trimmed Goal_Position (Reg 42)
            write_u8(ser, 6, 55, 0)
            write_u8(ser, 6, 34, 75)
            write_u8(ser, 6, 36, 95)
            write_u8(ser, 6, 55, 1)
            is_closing = g_tick < int(cur_ticks[5])
            write_u16(ser, 6, 48, 350 if is_closing else 600)
            write_u8(ser, 6, 40, 0)
            time.sleep(0.01)
            write_u8(ser, 6, 40, 1)
            start_g = int(cur_ticks[5])
            n_steps = max(40, int(abs(g_tick - start_g) // 16)) if is_closing else 25
            step_dt = 0.035 if is_closing else 0.025
            for s in range(1, n_steps + 1):
                alpha = s / float(n_steps)
                w = 0.5 * (1.0 - np.cos(np.pi * alpha))
                g_interp = int(round((1.0 - w) * start_g + w * g_tick))
                write_u16(ser, 6, 42, g_interp)
                time.sleep(step_dt)
            if is_closing:
                write_u16(ser, 6, 48, 450)
                settled_g = read_u16(ser, 6, 56)
                if g_tick <= 1850 and settled_g > 2050:
                    write_u16(ser, 6, 42, max(g_tick, settled_g - 110))
            time.sleep(0.35)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "draw-strokes":
            if len(sys.argv) < 3:
                raise ValueError(
                    "Usage: draw-strokes <json_string_or_path> [out_prefix]"
                )
            raw_arg = sys.argv[2]
            prefix = sys.argv[3] if len(sys.argv) >= 4 else "draw_strokes"
            if Path(raw_arg).is_file():
                payload = json.loads(Path(raw_arg).read_text())
            else:
                payload = json.loads(raw_arg)

            cur_ticks = read_all_ticks(ser)
            cur_q = ticks_to_rad(cur_ticks)

            if isinstance(payload, dict) and "strokes" in payload:
                z_travel = float(payload.get("z_travel", 0.060))
                default_pitch = float(payload.get("pitch_rad", 0.80))
                g_tick = int(payload.get("gripper_tick", 1700))
                strokes_3d = payload["strokes"]

                for stroke in strokes_3d:
                    if not stroke:
                        continue
                    pt0 = stroke[0]
                    x0, y0, z0 = float(pt0[0]), float(pt0[1]), float(pt0[2])
                    p0_pitch = float(pt0[3]) if len(pt0) >= 4 else default_pitch

                    # 1. Hover above start of stroke at z_travel
                    p0_up = np.array([x0, y0, z_travel], dtype=np.float64)
                    cur_q = solver.solve(
                        p0_up, target_pitch=p0_pitch, q_init=cur_q, exact_pitch=True
                    ).q_5d
                    t_up = rad_to_ticks(cur_q)
                    t_up[4] = 973
                    t_up.append(g_tick)
                    smooth_move_ticks(ser, t_up, steps=25, dt=0.020)

                    # 2. Lower smoothly to calibrated paper contact height z0
                    p0_dn = np.array([x0, y0, z0], dtype=np.float64)
                    cur_q = solver.solve(
                        p0_dn, target_pitch=p0_pitch, q_init=cur_q, exact_pitch=True
                    ).q_5d
                    t_dn = rad_to_ticks(cur_q)
                    t_dn[4] = 973
                    t_dn.append(g_tick)
                    cmd_dn = smooth_move_ticks(ser, t_dn, steps=18, dt=0.020)
                    trim_offset = [int(cmd_dn[m]) - int(t_dn[m]) for m in range(5)]

                    # 3. Trace dense points along the stroke while preserving closed-loop trim
                    for i in range(1, len(stroke)):
                        prev_pt = stroke[i - 1]
                        curr_pt = stroke[i]
                        p_a = np.array(
                            [float(prev_pt[0]), float(prev_pt[1]), float(prev_pt[2])],
                            dtype=np.float64,
                        )
                        p_b = np.array(
                            [float(curr_pt[0]), float(curr_pt[1]), float(curr_pt[2])],
                            dtype=np.float64,
                        )
                        pitch_a = (
                            float(prev_pt[3]) if len(prev_pt) >= 4 else default_pitch
                        )
                        pitch_b = (
                            float(curr_pt[3]) if len(curr_pt) >= 4 else default_pitch
                        )
                        dist = float(np.linalg.norm(p_b[:2] - p_a[:2]))
                        n_sub = max(5, int(np.ceil(dist / 0.003)))
                        for s in range(1, n_sub + 1):
                            alpha = s / float(n_sub)
                            p_sub = (1.0 - alpha) * p_a + alpha * p_b
                            pitch_sub = (1.0 - alpha) * pitch_a + alpha * pitch_b
                            cur_q = solver.solve(
                                p_sub,
                                target_pitch=pitch_sub,
                                q_init=cur_q,
                                exact_pitch=True,
                            ).q_5d
                            t_sub = rad_to_ticks(cur_q)
                            t_sub[4] = 973
                            for m in range(1, 6):
                                cmd_m = int(
                                    np.clip(
                                        int(t_sub[m - 1]) + trim_offset[m - 1],
                                        int(t_sub[m - 1]) - 55,
                                        int(t_sub[m - 1]) + 55,
                                    )
                                )
                                write_u16(ser, m, 42, cmd_m)
                            write_u16(ser, 6, 42, int(cmd_dn[5]))
                            time.sleep(0.035)
                        time.sleep(0.06)

                    # 4. Lift vertically to z_travel at end of stroke
                    end_pt = stroke[-1]
                    x_e, y_e = float(end_pt[0]), float(end_pt[1])
                    pitch_e = float(end_pt[3]) if len(end_pt) >= 4 else default_pitch
                    p_end_up = np.array([x_e, y_e, z_travel], dtype=np.float64)
                    cur_q = solver.solve(
                        p_end_up, target_pitch=pitch_e, q_init=cur_q, exact_pitch=True
                    ).q_5d
                    t_end_up = rad_to_ticks(cur_q)
                    t_end_up[4] = 973
                    t_end_up.append(g_tick)
                    smooth_move_ticks(ser, t_end_up, steps=18, dt=0.020)

                # 5. Retract to unobstructed inspection pose (pan=1750, reach=0.15, z=0.16, pitch=-0.20)
                inspect_ticks = solve_polar_ticks(solver, 1750, 0.15, 0.16, -0.20)
                inspect_ticks.append(g_tick)
                smooth_move_ticks(ser, inspect_ticks, steps=35, dt=0.025)
                time.sleep(0.5)
                report_status(ser, solver, prefix, default_out)
            else:
                strokes = payload
                cal_path = (
                    Path(__file__).resolve().parent.parent
                    / "config"
                    / "calibration.json"
                )
                cal = json.loads(cal_path.read_text())
                x_min, x_max = cal["board_x_range"]
                y_min, y_max = cal["board_y_range"]
                a, b, c = cal["plane_coeffs_abc"]
                pitch = float(cal["target_pitch"])
                z_floor = float(cal.get("draw_z_pen_down", -0.016)) - 0.002

                def map_pt(u: float, v: float, pen_down: bool) -> np.ndarray:
                    # u in [0, 1] -> left-to-right on paper (+Y to -Y)
                    # v in [0, 1] -> bottom-to-top on paper (x_min to x_max)
                    x = float(x_min + np.clip(v, 0.0, 1.0) * (x_max - x_min))
                    y = float(y_max - np.clip(u, 0.0, 1.0) * (y_max - y_min))
                    z_contact = max(float(a * x + b * y + c), z_floor)
                    z = z_contact if pen_down else z_contact + 0.028
                    return np.array([x, y, z], dtype=np.float64)

                for stroke in strokes:
                    pts = stroke["points"]
                    if not pts:
                        continue
                    # 1. Hover above start of stroke
                    p0_up = map_pt(pts[0][0], pts[0][1], pen_down=False)
                    cur_q = solver.solve(
                        p0_up, target_pitch=pitch, q_init=cur_q, exact_pitch=True
                    ).q_5d
                    t_up = rad_to_ticks(cur_q)
                    t_up[4] = 973
                    t_up.append(1700)
                    smooth_move_ticks(ser, t_up, steps=25, dt=0.020)

                    # 2. Lower to paper contact
                    p0_dn = map_pt(pts[0][0], pts[0][1], pen_down=True)
                    cur_q = solver.solve(
                        p0_dn, target_pitch=pitch, q_init=cur_q, exact_pitch=True
                    ).q_5d
                    t_dn = rad_to_ticks(cur_q)
                    t_dn[4] = 973
                    t_dn.append(1700)
                    cmd_dn = smooth_move_ticks(ser, t_dn, steps=18, dt=0.020)
                    trim_offset = [int(cmd_dn[m]) - int(t_dn[m]) for m in range(5)]

                    # 3. Trace dense points along the stroke while preserving closed-loop trim
                    for i in range(1, len(pts)):
                        u0, v0 = float(pts[i - 1][0]), float(pts[i - 1][1])
                        u1, v1 = float(pts[i][0]), float(pts[i][1])
                        dist = float(np.hypot(u1 - u0, v1 - v0))
                        n_sub = max(3, int(np.ceil(dist / 0.025)))
                        for s in range(1, n_sub + 1):
                            alpha = s / float(n_sub)
                            p_sub = map_pt(
                                u0 + alpha * (u1 - u0),
                                v0 + alpha * (v1 - v0),
                                pen_down=True,
                            )
                            cur_q = solver.solve(
                                p_sub, target_pitch=pitch, q_init=cur_q, exact_pitch=True
                            ).q_5d
                            t_sub = rad_to_ticks(cur_q)
                            t_sub[4] = 973
                            t_sub.append(1700)
                            for m in range(1, 6):
                                cmd_m = int(
                                    np.clip(
                                        int(t_sub[m - 1]) + trim_offset[m - 1],
                                        int(t_sub[m - 1]) - 55,
                                        int(t_sub[m - 1]) + 55,
                                    )
                                )
                                write_u16(ser, m, 42, cmd_m)
                            write_u16(ser, 6, 42, int(cmd_dn[5]))
                            time.sleep(0.040)

                    # 4. Lift pen at end of stroke
                    p_end_up = map_pt(pts[-1][0], pts[-1][1], pen_down=False)
                    cur_q = solver.solve(
                        p_end_up, target_pitch=pitch, q_init=cur_q, exact_pitch=True
                    ).q_5d
                    t_end_up = rad_to_ticks(cur_q)
                    t_end_up[4] = 973
                    t_end_up.append(1700)
                    smooth_move_ticks(ser, t_end_up, steps=15, dt=0.020)

                # 5. Retract to unobstructed pose so Camera 0 sees the full sheet
                inspect_ticks = solve_polar_ticks(solver, 1750, 0.15, 0.16, -0.20)
                inspect_ticks.append(1700)
                smooth_move_ticks(ser, inspect_ticks, steps=35, dt=0.025)
                report_status(ser, solver, prefix, default_out)

        elif cmd == "safe-park":
            mode = sys.argv[2] if len(sys.argv) >= 3 else "release"
            prefix = sys.argv[3] if len(sys.argv) >= 4 else "safe_park"
            cur_ticks = read_all_ticks(ser)
            target_g = int(cur_ticks[5]) if mode == "keep" else 3145
            if mode != "keep" and int(cur_ticks[5]) < 2900:
                # Open Motor 6 in place first so a wedged marker or tipped holder is never flung upward
                write_u8(ser, 6, 55, 0)
                write_u8(ser, 6, 34, 75)
                write_u8(ser, 6, 36, 75)
                write_u8(ser, 6, 55, 1)
                write_u16(ser, 6, 48, 750)
                write_u8(ser, 6, 40, 1)
                start_g = int(cur_ticks[5])
                for s in range(1, 21):
                    alpha = s / 20.0
                    w = 0.5 * (1.0 - np.cos(np.pi * alpha))
                    g_interp = int(round((1.0 - w) * start_g + w * 3145))
                    write_u16(ser, 6, 42, g_interp)
                    time.sleep(0.020)
                time.sleep(0.25)
                cur_ticks = read_all_ticks(ser)
            park_ticks = [2048, 1000, 3000, 2600, 973, target_g]
            smooth_move_ticks(ser, park_ticks, steps=40, dt=0.025)
            report_status(ser, solver, prefix, default_out)

        elif cmd == "relax":
            for m in range(1, 7):
                write_u8(ser, m, 40, 0)
            print(json.dumps({"status": "TORQUE_DISABLED_ALL_6_MOTORS"}))

        else:
            raise ValueError(f"Unknown command: {cmd}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
