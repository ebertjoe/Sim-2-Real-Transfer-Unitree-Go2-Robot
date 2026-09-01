#!/usr/bin/env python3
"""
compare_raibert.py
─────────────────
Standalone MuJoCo script (no ROS2) that runs the 53-dim policy and
logs Raibert foot targets side by side for two planners:
  - PLANNER A: uses v_cmd  (naive — what the 53-dim policy deployment does)
  - PLANNER B: uses v_gt   (MuJoCo ground truth — best case for hardware)

Both planners share the same phase state each step; only dx computation
differs. This gives a clean apples-to-apples comparison of foot targets.

Gait: run (gait 7), velocity ramps from 0.2 → 1.2 m/s over ~15s.
Output: CSV + console print every 10 steps.

Usage:
    python3 compare_raibert.py
"""

import math
import csv
from pathlib import Path

import mujoco
import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
POLICY_PATH = "/home/ubuntu/ros2_ws/src/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/go2/policy53.pt"
XML_PATH    = "/home/ubuntu/ros2_ws/src/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/go2/scene_flat.xml"
CSV_OUTPUT  = "/home/ubuntu/compare_raibert.csv"

# ── Sim constants ─────────────────────────────────────────────────────────────
ACTION_SCALE  = 0.25
STEP_DT       = 0.010
PHYSICS_DT    = 0.002
DECIMATION    = 5
OBS_CLIP      = 100.0
N_STEPS       = 1500   # 15 seconds of policy execution
SETTLE_STEPS  = 500    # physics steps before policy starts (1s at 500Hz)

# Velocity ramp: 0.2 → 1.2 m/s over N_STEPS
V_START = 0.2
V_END   = 1.2
GAIT_ID = 7  # run

# ── Joint mappings ────────────────────────────────────────────────────────────
MUJOCO_TO_INTERNAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
INTERNAL_TO_MUJOCO = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

DEFAULT_JOINT_POS_INTERNAL = np.array(
    [0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  1.0,  1.0,
    -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)

GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float64)

HIP_POS_B = np.array(
    [[ 0.183, -0.122, 0.0],
     [ 0.183,  0.122, 0.0],
     [-0.183, -0.122, 0.0],
     [-0.183,  0.122, 0.0]],
    dtype=np.float32,
)

GAIT_PARAMS = {
    "period":    0.3,
    "threshold": 0.4,
    "offset":    [0.0, 0.5, 0.5, 0.0],
    "k":         0.03,
    "z_nom":    -0.32,
    "x_lim":     0.12,
    "y_lim":     0.10,
}


# ── Utilities ─────────────────────────────────────────────────────────────────

def quat_rotate_inverse(q_wxyz, v):
    w     = float(q_wxyz[0])
    q_xyz = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3]], dtype=np.float64)
    v64   = np.array(v, dtype=np.float64)
    t     = 2.0 * np.cross(q_xyz, v64)
    return (v64 - w * t + np.cross(q_xyz, t)).astype(np.float32)


def quat_to_rotmat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float32)


def read_foot_contact(d, m, calf_body_ids):
    foot_contact = np.zeros(4, dtype=np.float32)
    for con in range(int(d.ncon)):
        c  = d.contact[con]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        for i, bid in enumerate(calf_body_ids):
            if b1 == bid or b2 == bid:
                foot_contact[i] = 1.0
    return foot_contact


# ── Raibert planner ───────────────────────────────────────────────────────────

class RaibertPlanner:
    """
    Single Raibert planner that computes foot targets using a provided velocity.
    Each step() call advances the shared phase and returns:
      - the full gait obs dict (for policy input)
      - dx_cmd: foot displacement using v_cmd
      - dx_gt:  foot displacement using v_gt
    so both comparisons share identical phase state.
    """
    STEP_HEIGHT = 0.10
    BLEND_ALPHA = 0.1

    def __init__(self):
        g = GAIT_PARAMS
        self._period    = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset    = np.array(g["offset"], dtype=np.float64)
        self._k         = float(g["k"])
        self._z_nom     = float(g["z_nom"])
        self._x_lim     = float(g["x_lim"])
        self._y_lim     = float(g["y_lim"])

        self._period_blended = self._period
        self._znom_blended   = self._z_nom
        self._t_exec         = 0.0
        self._phase_comp     = 0.0

        # Two independent p_ref_B: one for v_cmd targets, one for v_gt targets
        self._p_ref_cmd       = HIP_POS_B.copy().astype(np.float64)
        self._p_ref_cmd[:, 2] = self._z_nom
        self._p_ref_gt        = HIP_POS_B.copy().astype(np.float64)
        self._p_ref_gt[:, 2]  = self._z_nom
        self._prev_c          = np.ones(4, dtype=np.float64)

    def step(self, v_cmd_x, v_gt_x, q_wxyz):
        """
        Advance shared phase, compute foot targets for both v_cmd and v_gt.
        Returns gait_obs (built from v_cmd, for policy input) and comparison values.
        """
        self._period_blended = (self.BLEND_ALPHA * self._period
                                + (1 - self.BLEND_ALPHA) * self._period_blended)
        self._znom_blended   = (self.BLEND_ALPHA * self._z_nom
                                + (1 - self.BLEND_ALPHA) * self._znom_blended)

        T     = self._period_blended
        thr   = self._threshold
        k     = self._k
        z_nom = self._znom_blended
        Tst   = thr * T

        global_phase = ((self._t_exec - self._phase_comp) % T) / T
        leg_phase    = (global_phase + self._offset) % 1.0
        c_ref        = (leg_phase < thr).astype(np.float64)

        # ── dx for both velocity sources ──────────────────────────────────
        dx_cmd = 0.5 * Tst * float(v_cmd_x) + k * (float(v_cmd_x) - float(v_cmd_x))
        dx_gt  = 0.5 * Tst * float(v_gt_x)  + k * (float(v_gt_x)  - float(v_cmd_x))

        # ── Update p_ref for both on liftoff ──────────────────────────────
        liftoff = (self._prev_c > 0.5) & (c_ref < 0.5)

        for p_ref, dx in [(self._p_ref_cmd, dx_cmd), (self._p_ref_gt, dx_gt)]:
            new_p = HIP_POS_B.copy().astype(np.float64)
            new_p[:, 0] += dx
            new_p[:, 0]  = np.clip(new_p[:, 0],
                                   HIP_POS_B[:, 0] - self._x_lim,
                                   HIP_POS_B[:, 0] + self._x_lim)
            new_p[:, 2]  = z_nom
            for leg in range(4):
                if liftoff[leg]:
                    p_ref[leg] = new_p[leg]

        # ── Swing arc ─────────────────────────────────────────────────────
        swing_mask = (c_ref < 0.5).astype(np.float64)
        x_sw = np.clip((leg_phase - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
        z_sw = (0.5 * self.STEP_HEIGHT
                * (1.0 - np.cos(2.0 * math.pi * x_sw)) * swing_mask)

        R = quat_to_rotmat(q_wxyz)

        # cmd targets (for policy obs)
        pf_cmd = self._p_ref_cmd.copy()
        pf_cmd[:, 2] = z_nom + z_sw
        pw_cmd = (R @ pf_cmd.T).T

        # gt targets (comparison only)
        pf_gt = self._p_ref_gt.copy()
        pf_gt[:, 2] = z_nom + z_sw
        pw_gt = (R @ pf_gt.T).T

        self._prev_c  = c_ref.copy()
        self._t_exec += STEP_DT

        gait_obs = {
            "desFeetContact": c_ref.astype(np.float32),
            "refFootZ":       pw_cmd[:, 2].astype(np.float32),
            "refFootX":       pw_cmd[:, 0].astype(np.float32),
            "refFootY":       pw_cmd[:, 1].astype(np.float32),
        }

        return gait_obs, dx_cmd, dx_gt, pw_cmd[:, 0], pw_gt[:, 0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {XML_PATH}")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.timestep = PHYSICS_DT

    calf_body_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
    ]

    print(f"Loading policy: {POLICY_PATH}")
    policy = torch.jit.load(POLICY_PATH, map_location="cpu")
    policy.eval()

    raibert = RaibertPlanner()
    # Start Raibert mid-phase so stand doesn't begin at liftoff
    raibert._t_exec = raibert._period * 0.5

    tau = np.zeros(12, dtype=np.float32)

    # ── Settle: hold default pose for 1s before policy activates ─────────
    print(f"Settling for {SETTLE_STEPS} physics steps...")
    default_mujoco = DEFAULT_JOINT_POS_INTERNAL[INTERNAL_TO_MUJOCO]
    for _ in range(SETTLE_STEPS):
        for i in range(12):
            q  = d.qpos[7 + i]
            dq = d.qvel[6 + i]
            tau[i] = float(np.clip((default_mujoco[i] - q) * 25.0 - dq * 0.5,
                                   -23.5, 23.5))
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

    print(f"Starting policy. height={d.qpos[2]:.3f}m")

    # ── CSV setup ─────────────────────────────────────────────────────────
    csv_file = open(CSV_OUTPUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "step", "ep_t",
        "v_cmd", "v_gt",
        "dx_cmd", "dx_gt", "dx_diff",
        "mean_refFootX_cmd", "mean_refFootX_gt", "refFootX_diff",
        "height", "n_contact"
    ])

    print(f"\n{'step':>5} {'t':>5} {'v_cmd':>7} {'v_gt':>7} "
          f"{'dx_cmd':>8} {'dx_gt':>8} {'dx_diff':>9} "
          f"{'fX_cmd':>8} {'fX_gt':>8} {'fX_diff':>9} "
          f"{'height':>7} {'ct':>3}")
    print("-" * 100)

    physics_count = 0
    step_count    = 0
    episode_t     = 0.0

    for _ in range(N_STEPS * DECIMATION):

        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        physics_count += 1

        if physics_count % DECIMATION != 0:
            continue

        # ── Read state ────────────────────────────────────────────────────
        joint_pos = d.qpos[7:19].astype(np.float32)[MUJOCO_TO_INTERNAL]
        joint_vel = d.qvel[6:18].astype(np.float32)[MUJOCO_TO_INTERNAL]

        q_wxyz      = d.qpos[3:7].astype(np.float32)
        proj_grav   = quat_rotate_inverse(q_wxyz, GRAVITY_W)
        ang_vel_b   = d.sensordata[40:43].astype(np.float32)
        lin_vel_b   = d.sensordata[52:55].astype(np.float32)
        base_height = float(d.qpos[2])
        foot_contact = read_foot_contact(d, m, calf_body_ids)
        n_contact   = int(foot_contact.sum())

        # ── Velocity command: linear ramp ─────────────────────────────────
        alpha   = min(step_count / N_STEPS, 1.0)
        v_cmd_x = V_START + alpha * (V_END - V_START)
        v_gt_x  = float(lin_vel_b[0])
        vel_cmd = np.array([v_cmd_x, 0.0, 0.0], dtype=np.float32)

        # ── Raibert: shared phase, dual dx computation ────────────────────
        gait_obs, dx_cmd, dx_gt, fx_cmd, fx_gt = raibert.step(
            v_cmd_x=v_cmd_x, v_gt_x=v_gt_x, q_wxyz=q_wxyz)

        dx_diff = dx_gt - dx_cmd
        mean_fx_cmd = float(fx_cmd.mean())
        mean_fx_gt  = float(fx_gt.mean())
        fx_diff     = mean_fx_gt - mean_fx_cmd

        # ── Assemble 53-dim obs (policy always uses v_cmd Raibert) ────────
        obs = np.concatenate([
            proj_grav,
            joint_pos,
            ang_vel_b,
            joint_vel,
            vel_cmd,
            foot_contact,
            gait_obs["desFeetContact"],
            gait_obs["refFootZ"],
            gait_obs["refFootX"],
            gait_obs["refFootY"],
        ], dtype=np.float32)

        obs_clipped = np.clip(obs, -OBS_CLIP, OBS_CLIP)

        with torch.no_grad():
            action_raw = policy(
                torch.from_numpy(obs_clipped).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)

        target_internal = DEFAULT_JOINT_POS_INTERNAL + action_raw * ACTION_SCALE
        target_mujoco   = target_internal[INTERNAL_TO_MUJOCO]

        for i in range(12):
            q  = d.qpos[7 + i]
            dq = d.qvel[6 + i]
            tau[i] = float(np.clip(
                (target_mujoco[i] - q) * 25.0 - dq * 0.5, -23.5, 23.5))

        # ── Log ───────────────────────────────────────────────────────────
        writer.writerow([
            step_count, f"{episode_t:.3f}",
            f"{v_cmd_x:.4f}", f"{v_gt_x:.4f}",
            f"{dx_cmd:.5f}", f"{dx_gt:.5f}", f"{dx_diff:+.5f}",
            f"{mean_fx_cmd:.5f}", f"{mean_fx_gt:.5f}", f"{fx_diff:+.5f}",
            f"{base_height:.4f}", n_contact,
        ])

        if step_count % 10 == 0:
            print(f"{step_count:5d} {episode_t:5.2f} "
                  f"{v_cmd_x:7.3f} {v_gt_x:7.3f} "
                  f"{dx_cmd:8.4f} {dx_gt:8.4f} {dx_diff:+9.4f} "
                  f"{mean_fx_cmd:8.4f} {mean_fx_gt:8.4f} {fx_diff:+9.4f} "
                  f"{base_height:7.3f} {n_contact:3d}")

        episode_t  += STEP_DT
        step_count += 1

    csv_file.close()
    print(f"\nDone. CSV saved to: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()