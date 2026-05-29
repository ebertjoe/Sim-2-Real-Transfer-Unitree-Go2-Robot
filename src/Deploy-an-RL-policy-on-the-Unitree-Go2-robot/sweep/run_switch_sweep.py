#!/usr/bin/env python3
"""
run_switch_sweep.py — MuJoCo gait-switch robustness sweep.

Tests all 56 gait-from × gait-to pairs across 4 velocity commands × 5 seeds.
Each episode:
  Steps 0   → settle:        stand gait warm-up (gait-aware settle)
  Steps 0   → 300:           Phase 1 — run gait_from (establish rhythm)
  Step  300:                 SWITCH to gait_to
  Steps 300 → 1000:          Phase 2 — run gait_to (measure recovery)

Failure criterion (same as run_sweep.py):
  base_height < 0.15m for 10 consecutive steps → fall

Usage:
    python3 sweep/run_switch_sweep.py \
        --xml    resources/go2/scene_flat.xml \
        --policy resources/go2/policy.pt \
        --out    sweep/results/switch_results_raw.csv \
        [--n-runs 5] \
        [--save-timeseries]
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sim_runner import (
    SimRunner, GAIT_TABLE, STEP_DT, PHYSICS_DT, DECIMATION,
    DEFAULT_SETTLE_STEPS, GAIT_SETTLE_STEPS,
    FALL_HEIGHT, FALL_HEIGHT_GRACE,
    MUJOCO_TO_INTERNAL, INTERNAL_TO_MUJOCO,
    DEFAULT_JOINT_POS_INTERNAL, ACTION_SCALE, OBS_CLIP,
    GRAVITY_W, HIP_POS_B, NUM_JOINTS,
    quat_rotate_inverse, quat_to_rotmat, quat_to_rpy, read_foot_contact,
)
import mujoco
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Switch sweep constants ────────────────────────────────────────────────────

ALL_GAITS     = list(GAIT_TABLE.keys())   # 0–7
STAND_GAIT_ID = 6

# 4 representative velocity commands — fixed through entire episode
SWITCH_VEL_CMDS = [
    (0.0, 0.0, 0.0),   # zero  — pure gait switch, no locomotion
    (0.6, 0.0, 0.0),   # moderate forward
    (1.2, 0.0, 0.0),   # high forward
    (0.6, 0.2, 0.0),   # forward + lateral
]

PHASE1_STEPS   = 300    # steps running gait_from before switch (3s)
PHASE2_STEPS   = 700    # steps running gait_to after switch (7s)
MAX_EP_STEPS   = PHASE1_STEPS + PHASE2_STEPS   # 1000 total

# Metric windows (relative to switch point at step PHASE1_STEPS)
PRE_WINDOW     = 100    # last N steps of phase1 for baseline
TRANS_WINDOW   = 100    # first N steps after switch for transition metrics
POST_WINDOW    = 200    # last N steps of phase2 for steady-state metrics

RECOVERY_HEIGHT_TOL = 0.02   # m   — height within this of pre-mean = recovered
RECOVERY_VEL_TOL    = 0.10   # m/s — vx_error below this = recovered

# ── CSV schema ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "run_id", "seed",
    "gait_from", "gait_from_name", "gait_to", "gait_to_name",
    "vx_cmd", "vy_cmd", "wz_cmd",
    # termination
    "survived", "survival_steps", "survival_time_s", "termination_reason",
    # pre-switch baseline (last PRE_WINDOW steps of phase1)
    "pre_mean_height", "pre_mean_vx_error", "pre_mean_contact_acc",
    # transition window (first TRANS_WINDOW steps after switch)
    "trans_min_height", "trans_max_vx_error", "trans_max_roll",
    # recovery
    "height_recovery_steps", "vel_recovery_steps",
    # steady-state after (last POST_WINDOW steps of phase2)
    "post_mean_height", "post_mean_vx_error",
    "post_mean_contact_acc", "post_mean_torque_norm",
    # meta
    "wall_time_s",
]


# ── Raibert gait helper (minimal, for obs construction) ──────────────────────

class _RaibertGait:
    """Minimal Raibert gait for obs construction inside the switch runner."""
    STEP_HEIGHT = 0.10
    BLEND_ALPHA = 0.1

    def __init__(self, gait_id):
        import math
        self._math = math
        g = GAIT_TABLE[gait_id]
        self._period    = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset    = np.array(g["offset"], dtype=np.float64)
        self._k         = float(g["k"])
        self._z_nom     = float(g["z_nom"])
        self._x_lim     = float(g["x_lim"])
        self._y_lim     = float(g["y_lim"])
        self._period_b  = self._period
        self._znom_b    = self._z_nom
        self._old_pb    = self._period
        self._switched  = False
        self._t_exec    = 0.0
        self._phase_comp= 0.0
        self._p_ref_B   = HIP_POS_B.copy().astype(np.float64)
        self._p_ref_B[:,2] = self._z_nom
        self._prev_c    = np.ones(4, dtype=np.float64)

    def switch(self, new_gait_id):
        self._old_pb = self._period_b
        g = GAIT_TABLE[new_gait_id]
        self._period    = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset    = np.array(g["offset"], dtype=np.float64)
        self._k         = float(g["k"])
        self._z_nom     = float(g["z_nom"])
        self._x_lim     = float(g["x_lim"])
        self._y_lim     = float(g["y_lim"])
        self._switched  = True

    def step(self, v_B, v_cmd, q_wxyz):
        import math
        a = self.BLEND_ALPHA
        self._period_b = a*self._period + (1-a)*self._period_b
        self._znom_b   = a*self._z_nom  + (1-a)*self._znom_b
        if self._switched:
            t = self._t_exec
            self._phase_comp = t - (t-self._phase_comp)*(self._period_b/max(self._old_pb,1e-6))
            self._switched = False
        T=self._period_b; thr=self._threshold; k=self._k
        z=self._znom_b; Tst=thr*T
        gp=((self._t_exec-self._phase_comp)%T)/T
        lp=(gp+self._offset)%1.0
        c_ref=(lp<thr).astype(np.float64)
        dx=0.5*Tst*float(v_B[0])+k*(float(v_B[0])-float(v_cmd[0]))
        dy=0.5*Tst*float(v_B[1])+k*(float(v_B[1])-float(v_cmd[1]))
        np_ = HIP_POS_B.copy().astype(np.float64)
        np_[:,0]+=dx; np_[:,1]+=dy; np_[:,2]=z
        np_[:,0]=np.clip(np_[:,0],HIP_POS_B[:,0]-self._x_lim,HIP_POS_B[:,0]+self._x_lim)
        np_[:,1]=np.clip(np_[:,1],HIP_POS_B[:,1]-self._y_lim,HIP_POS_B[:,1]+self._y_lim)
        liftoff=(self._prev_c>0.5)&(c_ref<0.5)
        for l in range(4):
            if liftoff[l]: self._p_ref_B[l]=np_[l]
        sw=(c_ref<0.5).astype(np.float64)
        xsw=np.clip((lp-thr)/max(1-thr,1e-6),0,1)
        zsw=0.5*self.STEP_HEIGHT*(1-np.cos(2*math.pi*xsw))*sw
        pf=self._p_ref_B.copy(); pf[:,2]=z+zsw
        R=quat_to_rotmat(q_wxyz); pw=(R@pf.T).T
        self._prev_c=c_ref.copy(); self._t_exec+=STEP_DT
        return {
            "desFeetContact": c_ref.astype(np.float32),
            "refFootZ": pw[:,2].astype(np.float32),
            "refFootX": pw[:,0].astype(np.float32),
            "refFootY": pw[:,1].astype(np.float32),
        }


# ── Switch episode runner ─────────────────────────────────────────────────────

class SwitchEpisodeRunner:
    """
    Runs a single gait-switch episode using an existing SimRunner's
    MuJoCo model and policy. Handles obs construction, gait switching,
    metric accumulation, and per-step timeseries recording.
    """

    def __init__(self, sim: SimRunner):
        self.sim    = sim   # provides .m, .d, .policy
        self.m      = sim.m
        self.d      = sim.d
        self.policy = sim.policy

    def run(self, gait_from: int, gait_to: int, vel_cmd, seed: int,
            save_timeseries: bool = False):
        """
        Run one switch episode. Returns (metrics_dict, timeseries_dict|None).
        vel_cmd: (vx, vy, wz)
        """
        vx, vy, wz = vel_cmd
        settle = GAIT_SETTLE_STEPS.get(gait_from, DEFAULT_SETTLE_STEPS)

        # ── Reset via SimRunner's two-phase reset ─────────────────────────
        self.sim.reset(gait_id=gait_from, vel_cmd=list(vel_cmd), seed=seed)
        # sim.reset() already ran PD+policy warmup and set up RaibertGait
        # We need our own Raibert instance that we can switch mid-episode
        raibert = _RaibertGait(gait_from)
        raibert._t_exec = raibert._period * 0.5

        # ── Per-step state ────────────────────────────────────────────────
        current_gait   = gait_from
        switch_done    = False
        low_h_count    = 0
        step           = 0
        terminated     = False
        term_reason    = "running"
        survival_steps = 0

        # Metric buffers
        buf_h    = []; buf_vx   = []; buf_vy   = []
        buf_roll = []; buf_cacc = []; buf_tnorm= []
        buf_vxe  = []

        # Window buffers
        pre_h=[]; pre_vxe=[]; pre_cacc=[]
        trans_h=[]; trans_vxe=[]; trans_roll=[]
        post_h=[]; post_vxe=[]; post_cacc=[]; post_tnorm=[]

        # Timeseries (steps around switch: -PRE_WINDOW to +TRANS_WINDOW)
        ts_h=[]; ts_vxe=[]; ts_step=[]

        tau = self.sim._tau.copy()

        for step in range(MAX_EP_STEPS + settle):
            policy_step = step - settle   # <0 during settle

            # ── Switch at PHASE1_STEPS ────────────────────────────────────
            if policy_step == PHASE1_STEPS and not switch_done:
                current_gait = gait_to
                raibert.switch(gait_to)
                switch_done = True

            # ── Read state ────────────────────────────────────────────────
            jp  = self.d.qpos[7:19].astype(np.float32)[MUJOCO_TO_INTERNAL]
            jv  = self.d.qvel[6:18].astype(np.float32)[MUJOCO_TO_INTERNAL]
            tq  = np.clip(self.d.sensordata[24:36].astype(np.float32),-23.5,23.5)[MUJOCO_TO_INTERNAL]
            q_wxyz  = self.d.qpos[3:7].astype(np.float32)
            pg  = quat_rotate_inverse(q_wxyz, GRAVITY_W)
            av  = self.d.sensordata[40:43].astype(np.float32)
            lv  = self.d.sensordata[52:55].astype(np.float32)
            bh  = float(self.d.qpos[2])
            fc  = read_foot_contact(self.d, self.m, self.sim._calf_body_ids)
            roll, pitch, _ = quat_to_rpy(q_wxyz)

            # ── Gait obs ──────────────────────────────────────────────────
            vc_obs = np.zeros(3, dtype=np.float32) if current_gait == STAND_GAIT_ID \
                     else np.array([vx,vy,wz], dtype=np.float32)
            gait_obs = raibert.step(v_B=lv, v_cmd=np.array([vx,vy,wz]), q_wxyz=q_wxyz)

            # ── Build obs ─────────────────────────────────────────────────
            obs = np.concatenate([
                pg, jp, av, jv, lv, vc_obs, tq, fc,
                np.array([bh], dtype=np.float32),
                gait_obs["desFeetContact"],
                gait_obs["refFootZ"],
                gait_obs["refFootX"],
                gait_obs["refFootY"],
            ], dtype=np.float32)
            obs = np.clip(obs, -OBS_CLIP, OBS_CLIP)

            # ── Policy inference ──────────────────────────────────────────
            with torch.no_grad():
                act = self.policy(
                    torch.from_numpy(obs).unsqueeze(0)
                ).squeeze(0).numpy().astype(np.float32)

            tgt_i = DEFAULT_JOINT_POS_INTERNAL + act * ACTION_SCALE
            tgt_m = tgt_i[INTERNAL_TO_MUJOCO]
            for i in range(NUM_JOINTS):
                q  = float(self.d.qpos[7+i])
                dq = float(self.d.qvel[6+i])
                tau[i] = float(np.clip((tgt_m[i]-q)*25.-dq*0.5,-23.5,23.5))
            for _ in range(DECIMATION):
                self.d.ctrl[:] = tau
                mujoco.mj_step(self.m, self.d)

            # ── Termination (only after settle) ───────────────────────────
            if policy_step >= 0:
                if bh < FALL_HEIGHT:
                    low_h_count += 1
                else:
                    low_h_count = 0
                if low_h_count >= FALL_HEIGHT_GRACE:
                    survival_steps = policy_step
                    if policy_step < PHASE1_STEPS:
                        term_reason = "fall_phase1"
                    elif policy_step < PHASE1_STEPS + TRANS_WINDOW:
                        term_reason = "fall_transition"
                    else:
                        term_reason = "fall_phase2"
                    terminated = True
                    break

            # ── Accumulate metrics (only after settle) ────────────────────
            if policy_step >= 0:
                vxe = abs(float(lv[0]) - vx)
                des = gait_obs["desFeetContact"]
                cacc = float(np.mean((fc > 0.5) == (des > 0.5)))
                tnorm = float(np.linalg.norm(self.d.sensordata[24:36]))

                buf_h.append(bh); buf_vx.append(float(lv[0]))
                buf_vy.append(float(lv[1])); buf_roll.append(abs(roll))
                buf_cacc.append(cacc); buf_tnorm.append(tnorm)
                buf_vxe.append(vxe)

                # Pre-switch baseline
                if PHASE1_STEPS - PRE_WINDOW <= policy_step < PHASE1_STEPS:
                    pre_h.append(bh); pre_vxe.append(vxe); pre_cacc.append(cacc)

                # Transition window
                if PHASE1_STEPS <= policy_step < PHASE1_STEPS + TRANS_WINDOW:
                    trans_h.append(bh); trans_vxe.append(vxe); trans_roll.append(abs(roll))

                # Post steady-state
                post_start = MAX_EP_STEPS - POST_WINDOW
                if policy_step >= post_start:
                    post_h.append(bh); post_vxe.append(vxe)
                    post_cacc.append(cacc); post_tnorm.append(tnorm)

                # Timeseries around switch
                if PHASE1_STEPS - PRE_WINDOW <= policy_step < PHASE1_STEPS + TRANS_WINDOW:
                    ts_h.append(bh)
                    ts_vxe.append(vxe)
                    ts_step.append(policy_step - PHASE1_STEPS)  # relative to switch

        if not terminated:
            survival_steps = MAX_EP_STEPS
            term_reason    = "timeout"

        # ── Recovery metrics ──────────────────────────────────────────────
        pre_h_mean = float(np.mean(pre_h))  if pre_h  else float("nan")
        pre_vxe_mean = float(np.mean(pre_vxe)) if pre_vxe else float("nan")

        # Height recovery: steps after switch until height within tol of pre_mean
        h_rec = MAX_EP_STEPS  # default = never recovered
        for i, h in enumerate(trans_h + post_h[: MAX_EP_STEPS]):
            if abs(h - pre_h_mean) <= RECOVERY_HEIGHT_TOL:
                h_rec = i
                break

        # Velocity recovery: steps after switch until vxe < tol
        v_rec = MAX_EP_STEPS
        all_post_vxe = trans_vxe + buf_vxe[PHASE1_STEPS + TRANS_WINDOW:]
        for i, e in enumerate(all_post_vxe):
            if e < RECOVERY_VEL_TOL:
                v_rec = i
                break

        def _m(b): return float(np.mean(b)) if b else float("nan")

        metrics = {
            "survived":             term_reason == "timeout",
            "survival_steps":       survival_steps,
            "survival_time_s":      round(survival_steps * STEP_DT, 3),
            "termination_reason":   term_reason,
            # pre
            "pre_mean_height":      round(_m(pre_h),    4),
            "pre_mean_vx_error":    round(pre_vxe_mean, 4),
            "pre_mean_contact_acc": round(_m(pre_cacc), 4),
            # transition
            "trans_min_height":     round(float(min(trans_h))  if trans_h  else float("nan"), 4),
            "trans_max_vx_error":   round(float(max(trans_vxe)) if trans_vxe else float("nan"), 4),
            "trans_max_roll":       round(float(max(trans_roll)) if trans_roll else float("nan"), 4),
            # recovery
            "height_recovery_steps": h_rec,
            "vel_recovery_steps":    v_rec,
            # post
            "post_mean_height":      round(_m(post_h),     4),
            "post_mean_vx_error":    round(_m(post_vxe),   4),
            "post_mean_contact_acc": round(_m(post_cacc),  4),
            "post_mean_torque_norm": round(_m(post_tnorm), 4),
        }

        ts = None
        if save_timeseries:
            ts = {
                "step":    np.array(ts_step,  dtype=np.int16),
                "height":  np.array(ts_h,     dtype=np.float32),
                "vx_error":np.array(ts_vxe,   dtype=np.float32),
            }

        return metrics, ts


# ── Job builder ───────────────────────────────────────────────────────────────

def build_jobs(n_runs: int) -> list:
    jobs = []
    for gf in ALL_GAITS:
        for gt in ALL_GAITS:
            if gf == gt:
                continue
            for vel in SWITCH_VEL_CMDS:
                for seed in range(n_runs):
                    jobs.append({
                        "gait_from": gf,
                        "gait_to":   gt,
                        "vel_cmd":   vel,
                        "seed":      seed,
                    })
    return jobs


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(args):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts_dir = out_path.parent / "switch_timeseries"
    if args.save_timeseries:
        ts_dir.mkdir(parents=True, exist_ok=True)

    jobs  = build_jobs(args.n_runs)
    total = len(jobs)

    print(f"\n{'='*60}")
    print(f"  Gait-switch sweep — {total} episodes")
    print(f"  56 pairs × {len(SWITCH_VEL_CMDS)} vel cmds × {args.n_runs} seeds")
    print(f"  Phase1={PHASE1_STEPS} steps, Phase2={PHASE2_STEPS} steps")
    print(f"  Failure: height < {FALL_HEIGHT}m for {FALL_HEIGHT_GRACE} consecutive steps")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    print("Loading SimRunner (MuJoCo + policy)...")
    sim    = SimRunner(xml_path=args.xml, policy_path=args.policy)
    runner = SwitchEpisodeRunner(sim)
    print("Ready.\n")

    csv_exists = out_path.exists()
    csv_fh     = open(out_path, "a", newline="")
    writer     = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader(); csv_fh.flush()
    existing = 0
    if csv_exists:
        with open(out_path) as f:
            existing = sum(1 for _ in f) - 1
        print(f"[resume] {existing} existing rows.")

    pbar       = tqdm(total=total, unit="ep") if HAS_TQDM else None
    run_id     = existing
    n_survived = 0
    n_done     = 0

    for job in jobs:
        gf  = job["gait_from"]
        gt  = job["gait_to"]
        vel = job["vel_cmd"]
        sd  = job["seed"]
        vx, vy, wz = vel

        t0 = time.time()
        try:
            metrics, ts = runner.run(
                gait_from=gf, gait_to=gt,
                vel_cmd=vel, seed=sd,
                save_timeseries=args.save_timeseries,
            )
            wall_time = time.time() - t0

            if args.save_timeseries and ts is not None:
                fname = (f"ts_gf{gf}_gt{gt}_vx{vx:.1f}"
                         f"_vy{vy:.1f}_wz{wz:.1f}_s{sd}.npz")
                np.savez_compressed(
                    ts_dir / fname,
                    step=ts["step"], height=ts["height"],
                    vx_error=ts["vx_error"],
                    meta=json.dumps({
                        "gait_from": gf, "gait_to": gt,
                        "vx_cmd": vx, "vy_cmd": vy, "wz_cmd": wz,
                        "seed": sd, **metrics,
                    }),
                )

            row = {
                "run_id": run_id, "seed": sd,
                "gait_from": gf,
                "gait_from_name": GAIT_TABLE[gf]["name"],
                "gait_to":   gt,
                "gait_to_name":   GAIT_TABLE[gt]["name"],
                "vx_cmd": round(vx,2), "vy_cmd": round(vy,2), "wz_cmd": round(wz,2),
                "wall_time_s": round(wall_time, 3),
                **metrics,
            }
            writer.writerow(row); csv_fh.flush()

            n_survived += int(metrics["survived"])
            n_done     += 1
            run_id     += 1

        except Exception as e:
            wall_time = time.time() - t0
            print(f"\n[ERROR] gf={GAIT_TABLE[gf]['name']} gt={GAIT_TABLE[gt]['name']} "
                  f"vel={vel} seed={sd}")
            traceback.print_exc()
            writer.writerow({
                "run_id": run_id, "seed": sd,
                "gait_from": gf, "gait_from_name": GAIT_TABLE[gf]["name"],
                "gait_to":   gt, "gait_to_name":   GAIT_TABLE[gt]["name"],
                "vx_cmd": round(vx,2), "vy_cmd": round(vy,2), "wz_cmd": round(wz,2),
                "survived": False, "survival_steps": -1, "survival_time_s": -1,
                "termination_reason": f"error: {type(e).__name__}",
                "wall_time_s": round(wall_time, 3),
            })
            csv_fh.flush()
            run_id += 1
            n_done += 1

        if pbar:
            pbar.update(1)
            pbar.set_postfix_str(
                f"{GAIT_TABLE[gf]['name']}→{GAIT_TABLE[gt]['name']} "
                f"sr={100*n_survived/max(n_done,1):.0f}%"
            )

        if n_done % 200 == 0:
            print(f"\n[{n_done}/{total}] survival={100*n_survived/max(n_done,1):.1f}%")

    if pbar: pbar.close()
    csv_fh.close()

    print(f"\n{'='*60}")
    print("  SWITCH SWEEP COMPLETE")
    print(f"  Total rows : {run_id - existing}")
    print(f"  Survived   : {n_survived}  ({100*n_survived/max(n_done,1):.1f}%)")
    print(f"  Output     : {out_path}")
    print(f"{'='*60}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Gait switch sweep (MuJoCo)")
    p.add_argument("--xml",    required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--out",    default="sweep/results/switch_results_raw.csv")
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--save-timeseries", action="store_true",
                   help="Save height/vx_error timeseries around switch point")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)