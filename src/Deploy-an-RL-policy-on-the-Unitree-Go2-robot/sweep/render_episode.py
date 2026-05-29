#!/usr/bin/env python3
"""
render_episode.py — Run a single episode with the MuJoCo viewer open.

Useful for visually inspecting policy behaviour for a specific
(gait, velocity) combination — especially for debugging unexpected
failures in the sweep.

Usage:
    python3 sweep/render_episode.py \
        --xml    resources/go2/scene_flat.xml \
        --policy resources/go2/policy.pt \
        --gait   4 \
        --vx     0.0 \
        --vy     0.0 \
        --wz     0.0 \
        [--seed  0] \
        [--steps 1000]

Gait IDs:
    0=bound  1=trot  2=hop  3=amble
    4=pronk  5=limp  6=stand  7=run
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sim_runner import (
    SimRunner, GAIT_TABLE, STEP_DT, PHYSICS_DT, DECIMATION,
    DEFAULT_SETTLE_STEPS, GAIT_SETTLE_STEPS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Render a single sweep episode")
    p.add_argument("--xml",    required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--gait",   type=int, default=1,
                   help="Gait ID 0-7 (default: 1=trot)")
    p.add_argument("--vx",     type=float, default=0.6)
    p.add_argument("--vy",     type=float, default=0.0)
    p.add_argument("--wz",     type=float, default=0.0)
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--steps",  type=int,   default=1000,
                   help="Max policy steps (default: 1000 = 10s)")
    p.add_argument("--realtime", action="store_true", default=True,
                   help="Run at real time (default: True)")
    return p.parse_args()


def main():
    args = parse_args()

    gait_name  = GAIT_TABLE[args.gait]["name"]
    settle     = GAIT_SETTLE_STEPS.get(args.gait, DEFAULT_SETTLE_STEPS)

    print(f"\nRendering episode:")
    print(f"  gait    : {gait_name} (id={args.gait})")
    print(f"  vel_cmd : vx={args.vx}  vy={args.vy}  wz={args.wz}")
    print(f"  seed    : {args.seed}")
    print(f"  settle  : {settle} steps ({settle * STEP_DT:.1f}s)")
    print(f"  max     : {args.steps} steps ({args.steps * STEP_DT:.1f}s)")
    print()

    # ── Build runner (loads model + policy) ──────────────────────────────
    runner = SimRunner(xml_path=args.xml, policy_path=args.policy)

    # ── Launch viewer ─────────────────────────────────────────────────────
    viewer = mujoco.viewer.launch_passive(runner.m, runner.d)
    viewer.cam.azimuth   = 135
    viewer.cam.elevation = -20
    viewer.cam.distance  = 3.0

    # ── Reset to policy stand pose ────────────────────────────────────────
    print("Running two-phase reset (PD standup + policy warm-up)...")
    runner.reset(
        gait_id = args.gait,
        vel_cmd = [args.vx, args.vy, args.wz],
        seed    = args.seed,
    )
    print(f"Reset complete. Starting height: {runner.d.qpos[2]:.4f}m")
    print(f"Joint pos: {runner.d.qpos[7:19].round(3).tolist()}")
    print()
    print("Running episode — close viewer window to stop early.")
    print(f"{'Step':>6}  {'Height':>7}  {'vx_act':>7}  {'vx_err':>7}  {'Status'}")
    print("-" * 50)

    step = 0
    while viewer.is_running() and not runner.done:
        t0 = time.time()

        runner.step()
        viewer.sync()

        # Console output every 50 steps
        if step % 50 == 0:
            h  = runner.d.qpos[2]
            lv = runner.d.sensordata[52:55]
            vx_err = abs(float(lv[0]) - args.vx)
            in_settle = step < settle
            status = "SETTLE" if in_settle else ("RUNNING" if not runner.done else runner.termination_reason)
            print(f"{step:>6}  {h:>7.4f}  {float(lv[0]):>+7.3f}  {vx_err:>7.3f}  {status}")

        # Real-time pacing
        if args.realtime:
            elapsed   = time.time() - t0
            wait      = STEP_DT - elapsed
            if wait > 0:
                time.sleep(wait)

        step += 1

    print()
    metrics = runner.get_episode_metrics()
    print("=== EPISODE SUMMARY ===")
    print(f"  Result          : {'SURVIVED' if metrics['survived'] else 'FAILED'}")
    print(f"  Termination     : {metrics['termination_reason']}")
    print(f"  Survival time   : {metrics['survival_time_s']:.2f}s")
    print(f"  Mean height     : {metrics['mean_height']:.4f}m")
    print(f"  Mean vx error   : {metrics['mean_vx_error']:.4f} m/s")
    print(f"  Mean vy error   : {metrics['mean_vy_error']:.4f} m/s")
    print(f"  Mean wz error   : {metrics['mean_wz_error']:.4f} rad/s")
    print(f"  Contact acc     : {metrics['mean_contact_acc']:.3f}")
    print(f"  Torque norm     : {metrics['mean_torque_norm']:.2f} N·m")
    print(f"  XY drift        : {metrics['xy_drift_m']:.4f}m")
    print()

    viewer.close()


if __name__ == "__main__":
    main()