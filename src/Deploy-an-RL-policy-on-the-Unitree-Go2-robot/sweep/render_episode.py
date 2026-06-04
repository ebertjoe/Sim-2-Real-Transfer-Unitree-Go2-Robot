#!/usr/bin/env python3
"""
render_episode.py — Run a single episode with the MuJoCo viewer open.

Supports both single-gait episodes and gait-switch episodes.

Usage (single gait):
    python3 sweep/render_episode.py \
        --xml    resources/go2/scene_flat.xml \
        --policy resources/go2/policy.pt \
        --gait   4 \
        --vx 0.0 --vy 0.0 --wz 0.0

Usage (gait switch):
    python3 sweep/render_episode.py \
        --xml      resources/go2/scene_flat.xml \
        --policy   resources/go2/policy.pt \
        --gait-from 4 --gait-to 1 \
        --switch-at 300 \
        --vx 0.6 --vy 0.0 --wz 0.0

Gait IDs:
    0=bound  1=trot  2=hop  3=amble
    4=pronk  5=limp  6=stand  7=run
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sim_runner import (
    SimRunner, GAIT_TABLE, STEP_DT, DECIMATION,
    DEFAULT_SETTLE_STEPS, GAIT_SETTLE_STEPS,
    MUJOCO_TO_INTERNAL, INTERNAL_TO_MUJOCO,
    DEFAULT_JOINT_POS_INTERNAL, ACTION_SCALE, OBS_CLIP,
    GRAVITY_W, HIP_POS_B, NUM_JOINTS,
    quat_rotate_inverse, quat_to_rotmat, quat_to_rpy, read_foot_contact,
    FALL_HEIGHT, FALL_HEIGHT_GRACE,
)
from run_switch_sweep import _RaibertGait


def parse_args():
    p = argparse.ArgumentParser(description="Render a single or switch episode")
    p.add_argument("--xml",       required=True)
    p.add_argument("--policy",    required=True)
    p.add_argument("--gait",      type=int, default=None,
                   help="Gait ID for single-gait episode (0-7)")
    p.add_argument("--gait-from", type=int, default=None,
                   help="Starting gait ID for switch episode")
    p.add_argument("--gait-to",   type=int, default=None,
                   help="Target gait ID for switch episode")
    p.add_argument("--switch-at", type=int, default=300,
                   help="Policy step to perform the switch (default: 300 = 3s)")
    p.add_argument("--vx",        type=float, default=0.6)
    p.add_argument("--vy",        type=float, default=0.0)
    p.add_argument("--wz",        type=float, default=0.0)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--steps",     type=int,   default=1000,
                   help="Max policy steps after settle (default: 1000 = 10s)")
    p.add_argument("--no-realtime", action="store_true",
                   help="Run as fast as possible (default: real time)")
    return p.parse_args()


def run_single(args, runner, viewer):
    """Run a single-gait episode using SimRunner directly."""
    gait_id   = args.gait
    gait_name = GAIT_TABLE[gait_id]["name"]
    settle    = GAIT_SETTLE_STEPS.get(gait_id, DEFAULT_SETTLE_STEPS)

    print(f"\n  Mode    : single gait")
    print(f"  Gait    : {gait_name} (id={gait_id})")
    print(f"  Vel     : vx={args.vx}  vy={args.vy}  wz={args.wz}")
    print(f"  Settle  : {settle} steps ({settle * STEP_DT:.1f}s)")
    print(f"  Max     : {args.steps} steps ({args.steps * STEP_DT:.1f}s)\n")

    runner.reset(gait_id=gait_id, vel_cmd=[args.vx, args.vy, args.wz], seed=args.seed)
    print(f"Reset done. Height: {runner.d.qpos[2]:.4f}m\n")
    print(f"{'Step':>6}  {'Height':>7}  {'vx_act':>7}  {'vx_err':>7}  {'Status'}")
    print("-" * 52)

    step = 0
    while viewer.is_running() and not runner.done:
        t0 = time.time()
        runner.step()
        viewer.sync()

        if step % 50 == 0:
            h      = runner.d.qpos[2]
            lv     = runner.d.sensordata[52:55]
            vx_err = abs(float(lv[0]) - args.vx)
            status = "RUNNING" if not runner.done else runner.termination_reason
            print(f"{step:>6}  {h:>7.4f}  {float(lv[0]):>+7.3f}  {vx_err:>7.3f}  {status}")

        if not args.no_realtime:
            wait = STEP_DT - (time.time() - t0)
            if wait > 0: time.sleep(wait)
        step += 1

    m = runner.get_episode_metrics()
    print(f"\n=== SUMMARY ===")
    print(f"  Result       : {'SURVIVED' if m['survived'] else 'FAILED'}")
    print(f"  Termination  : {m['termination_reason']}")
    print(f"  Survival     : {m['survival_time_s']:.2f}s")
    print(f"  Mean height  : {m['mean_height']:.4f}m")
    print(f"  vx error     : {m['mean_vx_error']:.4f} m/s")
    print(f"  Contact acc  : {m['mean_contact_acc']:.3f}")


def run_switch(args, runner, viewer):
    """Run a gait-switch episode with live viewer."""
    gf         = args.gait_from
    gt         = args.gait_to
    switch_at  = args.switch_at
    vx, vy, wz = args.vx, args.vy, args.wz
    settle     = GAIT_SETTLE_STEPS.get(gf, DEFAULT_SETTLE_STEPS)

    print(f"\n  Mode      : gait switch")
    print(f"  From      : {GAIT_TABLE[gf]['name']} (id={gf})")
    print(f"  To        : {GAIT_TABLE[gt]['name']} (id={gt})")
    print(f"  Switch at : step {switch_at} ({switch_at * STEP_DT:.1f}s)")
    print(f"  Vel       : vx={vx}  vy={vy}  wz={wz}")
    print(f"  Settle    : {settle} steps ({settle * STEP_DT:.1f}s)")
    print(f"  Max       : {args.steps} steps ({args.steps * STEP_DT:.1f}s)\n")

    # Reset using SimRunner's two-phase reset with gait_from
    runner.reset(gait_id=gf, vel_cmd=[vx, vy, wz], seed=args.seed)
    print(f"Reset done. Height: {runner.d.qpos[2]:.4f}m\n")

    # Own Raibert instance so we can switch it mid-episode
    raibert       = _RaibertGait(gf)
    raibert._t_exec = raibert._period * 0.5
    current_gait  = gf
    switch_done   = False
    low_h_count   = 0
    tau           = runner._tau.copy()

    STAND_GAIT_ID = 6

    print(f"{'Step':>6}  {'Height':>7}  {'vx_act':>7}  {'vx_err':>7}  {'Gait':<8}  {'Status'}")
    print("-" * 62)

    # Track pre-switch height for recovery display
    pre_heights = []
    step = 0

    while viewer.is_running() and step < args.steps:
        t0 = time.time()

        # ── Switch ────────────────────────────────────────────────────────
        if step == switch_at and not switch_done:
            current_gait = gt
            raibert.switch(gt)
            switch_done = True
            print(f"\n{'>'*20} SWITCH: {GAIT_TABLE[gf]['name']} → {GAIT_TABLE[gt]['name']} {'<'*20}\n")

        # ── Read state ────────────────────────────────────────────────────
        jp     = runner.d.qpos[7:19].astype(np.float32)[MUJOCO_TO_INTERNAL]
        jv     = runner.d.qvel[6:18].astype(np.float32)[MUJOCO_TO_INTERNAL]
        tq     = np.clip(runner.d.sensordata[24:36].astype(np.float32), -23.5, 23.5)[MUJOCO_TO_INTERNAL]
        q_wxyz = runner.d.qpos[3:7].astype(np.float32)
        pg     = quat_rotate_inverse(q_wxyz, GRAVITY_W)
        av     = runner.d.sensordata[40:43].astype(np.float32)
        lv     = runner.d.sensordata[52:55].astype(np.float32)
        bh     = float(runner.d.qpos[2])
        fc     = read_foot_contact(runner.d, runner.m, runner._calf_body_ids)

        # ── Gait obs ──────────────────────────────────────────────────────
        vc_obs   = np.zeros(3, dtype=np.float32) if current_gait == STAND_GAIT_ID \
                   else np.array([vx, vy, wz], dtype=np.float32)
        gait_obs = raibert.step(v_B=lv, v_cmd=np.array([vx, vy, wz]), q_wxyz=q_wxyz)

        obs = np.concatenate([
            pg, jp, av, jv, lv, vc_obs, tq, fc,
            np.array([bh], dtype=np.float32),
            gait_obs["desFeetContact"],
            gait_obs["refFootZ"],
            gait_obs["refFootX"],
            gait_obs["refFootY"],
        ], dtype=np.float32)
        obs = np.clip(obs, -OBS_CLIP, OBS_CLIP)

        # ── Policy ────────────────────────────────────────────────────────
        with torch.no_grad():
            act = runner.policy(
                torch.from_numpy(obs).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)

        tgt_i = DEFAULT_JOINT_POS_INTERNAL + act * ACTION_SCALE
        tgt_m = tgt_i[INTERNAL_TO_MUJOCO]
        for i in range(NUM_JOINTS):
            q  = float(runner.d.qpos[7 + i])
            dq = float(runner.d.qvel[6 + i])
            tau[i] = float(np.clip((tgt_m[i] - q) * 25.0 - dq * 0.5, -23.5, 23.5))
        for _ in range(DECIMATION):
            runner.d.ctrl[:] = tau
            mujoco.mj_step(runner.m, runner.d)

        viewer.sync()

        # ── Fall detection ────────────────────────────────────────────────
        if bh < FALL_HEIGHT:
            low_h_count += 1
        else:
            low_h_count = 0
        if low_h_count >= FALL_HEIGHT_GRACE:
            print(f"\n[FALL] height={bh:.4f}m at step {step}")
            break

        # ── Accumulate pre-switch height baseline ─────────────────────────
        if max(0, switch_at - 100) <= step < switch_at:
            pre_heights.append(bh)

        # ── Console output every 25 steps ─────────────────────────────────
        if step % 25 == 0:
            vx_err = abs(float(lv[0]) - vx)
            gname  = GAIT_TABLE[current_gait]["name"]
            status = "PRE " if step < switch_at else "POST"

            # Show recovery progress after switch
            rec_str = ""
            if switch_done and pre_heights:
                pre_mean = np.mean(pre_heights)
                h_diff   = abs(bh - pre_mean)
                rec_str  = f"  Δh={h_diff:.3f}"

            print(f"{step:>6}  {bh:>7.4f}  {float(lv[0]):>+7.3f}  {vx_err:>7.3f}  {gname:<8}  {status}{rec_str}")

        if not args.no_realtime:
            wait = STEP_DT - (time.time() - t0)
            if wait > 0: time.sleep(wait)

        step += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== SWITCH EPISODE SUMMARY ===")
    print(f"  From        : {GAIT_TABLE[gf]['name']} → {GAIT_TABLE[gt]['name']}")
    print(f"  Switch at   : step {switch_at} ({switch_at * STEP_DT:.1f}s)")
    print(f"  Completed   : {step} steps  ({'fell' if low_h_count >= FALL_HEIGHT_GRACE else 'survived'})")
    if pre_heights:
        pre_mean = float(np.mean(pre_heights))
        print(f"  Pre-switch height (mean): {pre_mean:.4f}m")


def main():
    args = parse_args()

    # Validate args
    switch_mode = args.gait_from is not None and args.gait_to is not None
    single_mode = args.gait is not None

    if not switch_mode and not single_mode:
        print("[error] Provide either --gait (single) or --gait-from + --gait-to (switch)")
        sys.exit(1)
    if switch_mode and single_mode:
        print("[error] Provide either --gait or --gait-from/--gait-to, not both")
        sys.exit(1)

    print("\nLoading MuJoCo model and policy...")
    runner = SimRunner(xml_path=args.xml, policy_path=args.policy)

    viewer = mujoco.viewer.launch_passive(runner.m, runner.d)
    viewer.cam.azimuth   = 135
    viewer.cam.elevation = -20
    viewer.cam.distance  = 3.5

    print("Running two-phase reset (PD standup + policy warm-up)...")

    if switch_mode:
        run_switch(args, runner, viewer)
    else:
        run_single(args, runner, viewer)

    viewer.close()


if __name__ == "__main__":
    main()