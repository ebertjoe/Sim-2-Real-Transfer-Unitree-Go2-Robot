#!/usr/bin/env python3
"""
run_sweep.py — Sweep all gaits × velocity commands × seeds.

Usage:
    python run_sweep.py \
        --xml   /path/to/scene_flat.xml \
        --policy /path/to/policy.pt \
        --out   results/results_raw.csv \
        [--n-runs 5] \
        [--max-steps 1000] \
        [--save-failures]

Output:
    results/results_raw.csv   — one row per episode
    results/failures/         — optional .npz timeseries for failed episodes
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np

# ── Allow running from any directory ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from sim_runner import SimRunner, GAIT_TABLE, MAX_EPISODE_STEPS

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[warn] tqdm not found — install it for a progress bar: pip install tqdm")


# ── Sweep grid ────────────────────────────────────────────────────────────────

# Sampled uniformly across training distribution boundaries.
# lin_vel_x: forward only (0 → 1.2 m/s)
# lin_vel_y: symmetric lateral  (−0.4 → +0.4 m/s)
# ang_vel_z: symmetric yaw rate (−0.5 → +0.5 rad/s)
VX_CMDS  = [0.0, 0.3, 0.6, 0.9, 1.2]
VY_CMDS  = [-0.4, -0.2, 0.0, 0.2, 0.4]
WZ_CMDS  = [-0.5, -0.25, 0.0, 0.25, 0.5]
ALL_GAITS = list(GAIT_TABLE.keys())   # 0–7

# Stand gait (6) only ever runs with zero command — no point sweeping it
# over the full velocity grid.
STAND_GAIT_ID = 6


# ── CSV schema ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "run_id", "seed",
    # identity
    "gait_id", "gait_name", "vx_cmd", "vy_cmd", "wz_cmd",
    # termination
    "survived", "survival_steps", "survival_time_s", "termination_reason",
    # velocity tracking
    "mean_vx_actual", "mean_vy_actual", "mean_wz_actual",
    "mean_vx_error",  "mean_vy_error",  "mean_wz_error",
    # stability
    "mean_height", "std_height", "mean_roll", "mean_pitch",
    # contact
    "mean_contact_frac", "mean_contact_acc",
    # effort
    "mean_torque_norm",
    # drift
    "xy_drift_m",
    # meta
    "wall_time_s",
]


# ── Job builder ───────────────────────────────────────────────────────────────

def build_jobs(n_runs: int) -> list[dict]:
    """
    Return a shuffled list of job dicts, one per episode.

    Stand gait: n_runs episodes with zero command only.
    All other gaits: full VX × VY × WZ grid, n_runs each.
    """
    jobs = []

    for gait_id in ALL_GAITS:
        if gait_id == STAND_GAIT_ID:
            for seed in range(n_runs):
                jobs.append({
                    "gait_id": gait_id,
                    "vel_cmd": [0.0, 0.0, 0.0],
                    "seed":    seed,
                })
        else:
            for vx, vy, wz in product(VX_CMDS, VY_CMDS, WZ_CMDS):
                for seed in range(n_runs):
                    jobs.append({
                        "gait_id": gait_id,
                        "vel_cmd": [vx, vy, wz],
                        "seed":    seed,
                    })

    # Shuffle so early ETA estimates reflect the full distribution,
    # not just the first gait at the easiest velocity.
    random.shuffle(jobs)
    return jobs


# ── Timeseries recorder ───────────────────────────────────────────────────────

class TimeseriesRecorder:
    """Accumulates per-step arrays during a single episode."""

    def __init__(self):
        self.height:   list = []
        self.vx:       list = []
        self.vy:       list = []
        self.wz:       list = []
        self.roll:     list = []
        self.pitch:    list = []
        self.contact:  list = []   # shape (T, 4)
        self.tau_norm: list = []

    def record(self, runner: "SimRunner"):
        """Read current state directly from a SimRunner's last-step buffers."""
        # Buffers may be empty during settle window — guard with len check
        if runner._buf_height:
            self.height.append(runner._buf_height[-1])
            self.vx.append(runner._buf_vx_actual[-1])
            self.vy.append(runner._buf_vy_actual[-1])
            self.wz.append(runner._buf_wz_actual[-1])
            self.roll.append(runner._buf_roll[-1])
            self.pitch.append(runner._buf_pitch[-1])
            self.contact.append(
                [runner._buf_contact_frac[-1]] * 4  # placeholder per-foot
            )
            self.tau_norm.append(runner._buf_torque_norm[-1])

    def to_npz(self, path: str, meta: dict):
        np.savez_compressed(
            path,
            height   = np.array(self.height,   dtype=np.float32),
            vx       = np.array(self.vx,        dtype=np.float32),
            vy       = np.array(self.vy,        dtype=np.float32),
            wz       = np.array(self.wz,        dtype=np.float32),
            roll     = np.array(self.roll,      dtype=np.float32),
            pitch    = np.array(self.pitch,     dtype=np.float32),
            tau_norm = np.array(self.tau_norm,  dtype=np.float32),
            meta     = json.dumps(meta),
        )


# ── Progress tracking ─────────────────────────────────────────────────────────

class SweepStats:
    """Running counters printed periodically."""

    def __init__(self, total: int):
        self.total      = total
        self.done       = 0
        self.survived   = 0
        self.failed     = 0
        self.errors     = 0
        self._t_start   = time.time()
        # per-gait counters
        self.gait_done    = {g: 0 for g in ALL_GAITS}
        self.gait_survived= {g: 0 for g in ALL_GAITS}

    def update(self, gait_id: int, survived: bool, error: bool = False):
        self.done += 1
        self.gait_done[gait_id] += 1
        if error:
            self.errors += 1
        elif survived:
            self.survived += 1
            self.gait_survived[gait_id] += 1
        else:
            self.failed += 1

    def eta_str(self) -> str:
        elapsed = time.time() - self._t_start
        if self.done == 0:
            return "?"
        rate     = self.done / elapsed            # episodes / s
        remaining = (self.total - self.done) / rate
        m, s     = divmod(int(remaining), 60)
        h, m     = divmod(m, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"

    def summary_line(self) -> str:
        pct = 100 * self.survived / max(self.done - self.errors, 1)
        elapsed = time.time() - self._t_start
        rate    = self.done / max(elapsed, 1e-6)
        return (
            f"[{self.done}/{self.total}] "
            f"survived={self.survived} ({pct:.1f}%)  "
            f"failed={self.failed}  errors={self.errors}  "
            f"rate={rate:.1f} ep/s  ETA={self.eta_str()}"
        )

    def gait_summary(self) -> str:
        lines = ["  Gait survival rates so far:"]
        for g in ALL_GAITS:
            n = self.gait_done[g]
            s = self.gait_survived[g]
            pct = 100 * s / n if n > 0 else 0.0
            lines.append(
                f"    {GAIT_TABLE[g]['name']:8s}  {s:4d}/{n:<4d}  ({pct:.0f}%)"
            )
        return "\n".join(lines)


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    xml_path:      str,
    policy_path:   str,
    out_path:      str,
    n_runs:        int  = 5,
    max_steps:     int  = MAX_EPISODE_STEPS,
    save_failures: bool = False,
    print_every:   int  = 100,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    failures_dir = out_path.parent / "failures"
    if save_failures:
        failures_dir.mkdir(parents=True, exist_ok=True)

    # ── Build job list ────────────────────────────────────────────────────
    jobs  = build_jobs(n_runs)
    total = len(jobs)

    n_locomotion_gaits = len([g for g in ALL_GAITS if g != STAND_GAIT_ID])
    n_stand_episodes   = n_runs
    n_loco_episodes    = n_locomotion_gaits * len(VX_CMDS) * len(VY_CMDS) * len(WZ_CMDS) * n_runs

    print(f"\n{'='*60}")
    print(f"  Gait sweep — {total} episodes total")
    print(f"  Locomotion gaits : {n_locomotion_gaits} × "
          f"{len(VX_CMDS)}vx × {len(VY_CMDS)}vy × {len(WZ_CMDS)}wz × "
          f"{n_runs} runs = {n_loco_episodes}")
    print(f"  Stand gait       : {n_stand_episodes} runs (zero cmd only)")
    print(f"  Episode length   : {max_steps * 0.01:.0f} s max")
    print(f"  Output           : {out_path}")
    print(f"{'='*60}\n")

    # ── Initialise runner (model loaded once, reused across episodes) ─────
    print("Loading MuJoCo model and policy...")
    runner = SimRunner(xml_path=xml_path, policy_path=policy_path)
    print("Ready.\n")

    stats    = SweepStats(total)
    run_id   = 0
    existing = 0

    # ── CSV: append mode so a crashed sweep can be resumed ────────────────
    csv_exists = out_path.exists()
    csv_fh     = open(out_path, "a", newline="")
    writer     = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()
        csv_fh.flush()
    else:
        # Count existing rows so run_id stays unique across resume
        with open(out_path) as f:
            existing = sum(1 for _ in f) - 1   # subtract header
        print(f"[resume] Found {existing} existing rows — appending.")

    # ── Progress bar ──────────────────────────────────────────────────────
    pbar = tqdm(total=total, unit="ep") if HAS_TQDM else None

    try:
        for job_idx, job in enumerate(jobs):
            gait_id = job["gait_id"]
            vel_cmd = job["vel_cmd"]
            seed    = job["seed"]
            run_id  = existing + job_idx

            ts_recorder = TimeseriesRecorder() if save_failures else None
            t0 = time.time()

            try:
                runner.reset(gait_id=gait_id, vel_cmd=vel_cmd, seed=seed)

                while not runner.done:
                    runner.step()
                    if ts_recorder is not None:
                        ts_recorder.record(runner)

                metrics    = runner.get_episode_metrics()
                wall_time  = time.time() - t0
                survived   = metrics["survived"]

                row = {"run_id": run_id, "seed": seed, "wall_time_s": round(wall_time, 3)}
                row.update(metrics)
                writer.writerow(row)
                csv_fh.flush()

                stats.update(gait_id, survived=survived)

                # Save timeseries for failed episodes
                if save_failures and not survived and ts_recorder is not None:
                    fname = (
                        f"fail_g{gait_id}_vx{vel_cmd[0]:.2f}"
                        f"_vy{vel_cmd[1]:.2f}_wz{vel_cmd[2]:.2f}"
                        f"_s{seed}.npz"
                    )
                    ts_recorder.to_npz(
                        str(failures_dir / fname),
                        meta={**metrics, "run_id": run_id},
                    )

            except Exception as e:
                wall_time = time.time() - t0
                print(f"\n[ERROR] run_id={run_id} "
                      f"gait={GAIT_TABLE[gait_id]['name']} "
                      f"vel={vel_cmd} seed={seed}")
                traceback.print_exc()
                # Write error row so we know which episode failed
                error_row = {
                    "run_id":             run_id,
                    "seed":               seed,
                    "gait_id":            gait_id,
                    "gait_name":          GAIT_TABLE[gait_id]["name"],
                    "vx_cmd":             vel_cmd[0],
                    "vy_cmd":             vel_cmd[1],
                    "wz_cmd":             vel_cmd[2],
                    "survived":           False,
                    "survival_steps":     -1,
                    "survival_time_s":    -1,
                    "termination_reason": f"error: {type(e).__name__}",
                    "wall_time_s":        round(wall_time, 3),
                }
                writer.writerow(error_row)
                csv_fh.flush()
                stats.update(gait_id, survived=False, error=True)

            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"{GAIT_TABLE[gait_id]['name']} "
                    f"vx={vel_cmd[0]:.1f} "
                    f"{'OK' if stats.survived > 0 else '--'}"
                )

            # Periodic console summary
            if (job_idx + 1) % print_every == 0:
                print(f"\n{stats.summary_line()}")
                print(stats.gait_summary())
                print()

    finally:
        if pbar:
            pbar.close()
        csv_fh.close()

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SWEEP COMPLETE")
    print(f"  {stats.summary_line()}")
    print(stats.gait_summary())
    print(f"  Results → {out_path}")
    if save_failures:
        n_saved = len(list(failures_dir.glob("*.npz")))
        print(f"  Failure timeseries ({n_saved} files) → {failures_dir}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gait robustness sweep")
    p.add_argument("--xml",    required=True,  help="Path to scene_flat.xml")
    p.add_argument("--policy", required=True,  help="Path to policy.pt")
    p.add_argument("--out",    default="results/results_raw.csv",
                   help="Output CSV path (default: results/results_raw.csv)")
    p.add_argument("--n-runs", type=int, default=5,
                   help="Episodes per (gait, vel_cmd) combination (default: 5)")
    p.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS,
                   help=f"Max policy steps per episode (default: {MAX_EPISODE_STEPS})")
    p.add_argument("--save-failures", action="store_true",
                   help="Save per-step timeseries .npz for every failed episode")
    p.add_argument("--print-every", type=int, default=100,
                   help="Print gait summary every N episodes (default: 100)")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed for job shuffle (default: 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    run_sweep(
        xml_path      = args.xml,
        policy_path   = args.policy,
        out_path      = args.out,
        n_runs        = args.n_runs,
        max_steps     = args.max_steps,
        save_failures = args.save_failures,
        print_every   = args.print_every,
    )