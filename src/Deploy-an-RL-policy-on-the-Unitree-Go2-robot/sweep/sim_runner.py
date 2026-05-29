#!/usr/bin/env python3
"""
sim_runner.py — Headless MuJoCo + policy runner for gait sweep.

No ROS, no viewer, no threading. Takes (gait_id, vel_cmd, seed),
runs one episode, returns a metrics dict.

Usage:
    runner = SimRunner(xml_path, policy_path)
    runner.reset(gait_id=1, vel_cmd=[0.6, 0.0, 0.0], seed=42)
    while not runner.done:
        runner.step()
    metrics = runner.get_episode_metrics()
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import torch

# ── Constants (must match simulator exactly) ─────────────────────────────────

NUM_JOINTS   = 12
ACTION_SCALE = 0.25
STEP_DT      = 0.010   # policy dt  = 10 ms = 100 Hz
PHYSICS_DT   = 0.002   # physics dt =  2 ms = 500 Hz
DECIMATION   = 5       # physics steps per policy step
OBS_CLIP     = 100.0

# Joint reordering (see simulator for full derivation)
# MUJOCO_TO_INTERNAL: identity — obs joint arrays fed in MuJoCo order
MUJOCO_TO_INTERNAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# INTERNAL_TO_MUJOCO: action reindex from Isaac internal → MuJoCo order
INTERNAL_TO_MUJOCO = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

DEFAULT_JOINT_POS_INTERNAL = np.array(
    [ 0.1, -0.1,  0.1, -0.1,   # hip:   FL, FR, RL, RR
      0.8,  0.8,  1.0,  1.0,   # thigh: FL, FR, RL, RR
     -1.5, -1.5, -1.5, -1.5],  # calf:  FL, FR, RL, RR
    dtype=np.float32,
)

DEFAULT_JOINT_POS_TRAINING = np.array(
    [+0.1,  0.8, -1.5,   # FR
     -0.1,  0.8, -1.5,   # FL
     +0.1,  1.0, -1.5,   # RR
     -0.1,  1.0, -1.5],  # RL
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

GAIT_TABLE = {
    0: {"name": "bound",  "period": 0.4, "threshold": 0.4,   "offset": [0.5, 0.5, 0.0,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    1: {"name": "trot",   "period": 0.4, "threshold": 0.5,   "offset": [0.0, 0.5, 0.5,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    2: {"name": "hop",    "period": 0.3, "threshold": 0.5,   "offset": [0.0, 0.0, 0.0,  0.0 ], "k": 0.03, "z_nom": -0.30, "x_lim": 0.10, "y_lim": 0.10},
    3: {"name": "amble",  "period": 0.5, "threshold": 0.625, "offset": [0.0, 0.5, 0.25, 0.75], "k": 0.02, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    4: {"name": "pronk",  "period": 0.5, "threshold": 0.5,   "offset": [0.0, 0.0, 0.0,  0.0 ], "k": 0.01, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    5: {"name": "limp",   "period": 0.4, "threshold": 0.5,   "offset": [0.5, 0.5, 0.5,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    6: {"name": "stand",  "period": 1.0, "threshold": 1.0,   "offset": [0.0, 0.0, 0.0,  0.0 ], "k": 0.01, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    7: {"name": "run",    "period": 0.3, "threshold": 0.4,   "offset": [0.0, 0.5, 0.5,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
}

# ── Termination thresholds ────────────────────────────────────────────────────

FALL_HEIGHT        = 0.15   # m    — base below this → fallen
                             # 0.15 chosen from pronk data: normal bounce dips to 0.189m,
                             # genuine collapse goes to 0.05-0.10m. 0.20 caused false positives.
FALL_HEIGHT_GRACE  = 10     # consecutive steps below FALL_HEIGHT before fall declared
                             # prevents false positives during pronk/hop flight phases
MAX_EPISODE_STEPS  = 1000   # policy steps = 10 s

# Gait-aware settle window — steps at episode start excluded from metrics AND
# termination. Most gaits need ~1s to establish rhythm from standing pose.
# Pronk (all-four flight) and hop need longer — data shows startup failures
# clustering at 0.85s for pronk, so 2s settle eliminates false fall detection.
DEFAULT_SETTLE_STEPS = 100   # 1s — trot, run, amble, limp, bound, stand
GAIT_SETTLE_STEPS = {
    2: 150,   # hop   — bilateral flight phase needs ~1.5s to establish
    4: 200,   # pronk — all-four flight needs ~2s to establish bounce rhythm
}

# PD gains for standup
STANDUP_KP            = 25.0
STANDUP_KD            = 0.5
STANDUP_PHYSICS_STEPS = 500    # 1 s at 500 Hz
STANDUP_POSE_NOISE    = 0.01   # rad, per-joint std


# ── Geometry helpers ─────────────────────────────────────────────────────────

def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world frame into body frame."""
    w     = float(q_wxyz[0])
    q_xyz = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3]], dtype=np.float64)
    v64   = np.array(v, dtype=np.float64)
    t     = 2.0 * np.cross(q_xyz, v64)
    return (v64 - w * t + np.cross(q_xyz, t)).astype(np.float32)


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float32)


def quat_to_rpy(q_wxyz: np.ndarray):
    """Return (roll, pitch, yaw) in radians from wxyz quaternion."""
    w, x, y, z = q_wxyz.astype(np.float64)
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


# ── Raibert gait (identical to simulator) ────────────────────────────────────

class RaibertGait:
    STEP_HEIGHT = 0.10
    BLEND_ALPHA = 0.1

    def __init__(self, gait_id: int = 6):
        g = GAIT_TABLE[gait_id]
        self._gait_id   = gait_id
        self._period    = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset    = np.array(g["offset"], dtype=np.float64)
        self._k         = float(g["k"])
        self._z_nom     = float(g["z_nom"])
        self._x_lim     = float(g["x_lim"])
        self._y_lim     = float(g["y_lim"])

        self._period_blended      = self._period
        self._znom_blended        = self._z_nom
        self._old_period_blended  = self._period_blended
        self._gait_just_switched  = False

        self._t_exec             = 0.0
        self._phase_compensation = 0.0

        self._p_ref_B       = HIP_POS_B.copy().astype(np.float64)
        self._p_ref_B[:, 2] = self._z_nom
        self._prev_c        = np.ones(4, dtype=np.float64)

    def reset(self):
        self._t_exec             = 0.0
        self._phase_compensation = 0.0
        self._period_blended     = self._period
        self._znom_blended       = self._z_nom
        self._prev_c             = np.ones(4, dtype=np.float64)
        self._p_ref_B            = HIP_POS_B.copy().astype(np.float64)
        self._p_ref_B[:, 2]      = self._z_nom

    def switch_gait(self, new_gait_id: int):
        self._old_period_blended = self._period_blended
        g = GAIT_TABLE[new_gait_id]
        self._gait_id   = new_gait_id
        self._period    = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset    = np.array(g["offset"], dtype=np.float64)
        self._k         = float(g["k"])
        self._z_nom     = float(g["z_nom"])
        self._x_lim     = float(g["x_lim"])
        self._y_lim     = float(g["y_lim"])
        self._gait_just_switched = True

    def step(self, v_B, v_cmd, q_wxyz):
        a = self.BLEND_ALPHA
        self._period_blended = a * self._period + (1.0 - a) * self._period_blended
        self._znom_blended   = a * self._z_nom  + (1.0 - a) * self._znom_blended

        if self._gait_just_switched:
            t = self._t_exec
            self._phase_compensation = (
                t - (t - self._phase_compensation) *
                (self._period_blended / max(self._old_period_blended, 1e-6))
            )
            self._gait_just_switched = False

        T     = self._period_blended
        thr   = self._threshold
        k     = self._k
        z_nom = self._znom_blended
        Tst   = thr * T

        global_phase = ((self._t_exec - self._phase_compensation) % T) / T
        leg_phase    = (global_phase + self._offset) % 1.0
        c_ref        = (leg_phase < thr).astype(np.float64)

        dx = 0.5 * Tst * float(v_B[0]) + k * (float(v_B[0]) - float(v_cmd[0]))
        dy = 0.5 * Tst * float(v_B[1]) + k * (float(v_B[1]) - float(v_cmd[1]))

        new_p = HIP_POS_B.copy().astype(np.float64)
        new_p[:, 0] += dx
        new_p[:, 1] += dy
        new_p[:, 2]  = z_nom
        new_p[:, 0]  = np.clip(new_p[:, 0],
                                HIP_POS_B[:, 0] - self._x_lim,
                                HIP_POS_B[:, 0] + self._x_lim)
        new_p[:, 1]  = np.clip(new_p[:, 1],
                                HIP_POS_B[:, 1] - self._y_lim,
                                HIP_POS_B[:, 1] + self._y_lim)

        liftoff = (self._prev_c > 0.5) & (c_ref < 0.5)
        for leg in range(4):
            if liftoff[leg]:
                self._p_ref_B[leg] = new_p[leg]

        swing_mask = (c_ref < 0.5).astype(np.float64)
        x_sw = np.clip((leg_phase - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
        z_sw = 0.5 * self.STEP_HEIGHT * (1.0 - np.cos(2.0 * math.pi * x_sw)) * swing_mask

        pf       = self._p_ref_B.copy()
        pf[:, 2] = z_nom + z_sw
        R        = quat_to_rotmat(q_wxyz)
        pw       = (R @ pf.T).T

        self._prev_c  = c_ref.copy()
        self._t_exec += STEP_DT

        return {
            "desFeetContact": c_ref.astype(np.float32),
            "refFootZ":       pw[:, 2].astype(np.float32),
            "refFootX":       pw[:, 0].astype(np.float32),
            "refFootY":       pw[:, 1].astype(np.float32),
        }


# ── Contact helper ────────────────────────────────────────────────────────────

def read_foot_contact(d, m, body_ids) -> np.ndarray:
    contact = np.zeros(4, dtype=np.float32)
    for con in range(int(d.ncon)):
        c  = d.contact[con]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        for i, bid in enumerate(body_ids):
            if b1 == bid or b2 == bid:
                contact[i] = 1.0
    return contact


# ── SimRunner ─────────────────────────────────────────────────────────────────

class SimRunner:
    """
    Headless single-episode runner.

    Typical usage:
        runner = SimRunner(xml_path, policy_path)
        runner.reset(gait_id=1, vel_cmd=[0.6, 0.0, 0.0], seed=0)
        while not runner.done:
            runner.step()
        metrics = runner.get_episode_metrics()
    """

    def __init__(self, xml_path: str, policy_path: str):
        # ── MuJoCo ────────────────────────────────────────────────────────
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = PHYSICS_DT

        self._calf_body_ids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        ]

        # ── Policy ────────────────────────────────────────────────────────
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        # ── Episode state (initialised by reset()) ────────────────────────
        self.done               = True
        self.termination_reason = "not_started"
        self._gait_id           = 6
        self._vel_cmd           = np.zeros(3, dtype=np.float32)
        self._step_count        = 0
        self._episode_t         = 0.0
        self._raibert           = RaibertGait(gait_id=6)
        self._tau               = np.zeros(NUM_JOINTS, dtype=np.float32)

        # ── Per-step metric accumulators ──────────────────────────────────
        self._buf_vx_actual:    list = []
        self._buf_vy_actual:    list = []
        self._buf_wz_actual:    list = []
        self._buf_vx_error:     list = []
        self._buf_vy_error:     list = []
        self._buf_wz_error:     list = []
        self._buf_height:       list = []
        self._buf_roll:         list = []
        self._buf_pitch:        list = []
        self._buf_contact_frac: list = []
        self._buf_contact_acc:  list = []
        self._buf_torque_norm:  list = []
        self._start_qpos_xy     = np.zeros(2)

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _reset_to_policy_stand(self, seed: int):
        """
        Two-phase reset that replicates the live simulator experience:

        Phase 1 — PD standup (2000 physics steps = 4s):
            Drive joints toward DEFAULT_JOINT_POS_TRAINING from the URDF
            default (all-zeros). There are no keyframes in scene_flat.xml so
            mj_resetDataKeyframe is not used. 2000 steps are needed because
            500 steps leave calves ~0.2 rad away from target.

        Phase 2 — Policy warm-up (300 policy steps = 3s, stand gait):
            Run the actual policy with stand gait and zero velocity command.
            This drives the robot to the policy's own steady-state standing
            pose (~height 0.315m, rear calves -1.37, front calves -1.82)
            which differs from DEFAULT_JOINT_POS_TRAINING. This exactly
            matches the live simulator where the robot runs in stand mode
            before the operator switches gaits.

        Seeds produce distinct episodes via small joint noise added before
        phase 1, so the policy warm-up converges to slightly different poses.
        """
        # ── Phase 1: PD standup from URDF default ─────────────────────────
        mujoco.mj_resetData(self.m, self.d)

        rng   = np.random.default_rng(seed)
        noise = rng.normal(0.0, STANDUP_POSE_NOISE, NUM_JOINTS).astype(np.float32)
        self.d.qpos[7:19] += noise
        mujoco.mj_forward(self.m, self.d)

        target = DEFAULT_JOINT_POS_TRAINING.copy()
        for _ in range(2000):   # 4s at 500 Hz — fully converges calves to -1.5
            for i in range(NUM_JOINTS):
                q   = float(self.d.qpos[7 + i])
                dq  = float(self.d.qvel[6 + i])
                self.d.ctrl[i] = float(np.clip(
                    (target[i] - q) * STANDUP_KP - dq * STANDUP_KD,
                    -23.5, 23.5,
                ))
            mujoco.mj_step(self.m, self.d)

        # ── Phase 2: policy warm-up with stand gait ───────────────────────
        # Stand gait obs: all feet in contact, refs at hip positions, zero vel
        STAND_GAIT_ID = 6
        z_nom  = GAIT_TABLE[STAND_GAIT_ID]["z_nom"]
        des_c  = np.ones(4,  dtype=np.float32)
        ref_z  = np.full(4,  z_nom, dtype=np.float32)
        ref_x  = HIP_POS_B[:, 0].astype(np.float32)
        ref_y  = HIP_POS_B[:, 1].astype(np.float32)
        tau    = np.zeros(NUM_JOINTS, dtype=np.float32)

        for _ in range(300):   # 3s — fully converges to policy stand pose
            jp  = self.d.qpos[7:19].astype(np.float32)[MUJOCO_TO_INTERNAL]
            jv  = self.d.qvel[6:18].astype(np.float32)[MUJOCO_TO_INTERNAL]
            tq  = np.clip(self.d.sensordata[24:36].astype(np.float32),
                          -23.5, 23.5)[MUJOCO_TO_INTERNAL]
            q_wxyz = self.d.qpos[3:7].astype(np.float32)
            pg  = quat_rotate_inverse(q_wxyz, GRAVITY_W)
            av  = self.d.sensordata[40:43].astype(np.float32)
            lv  = self.d.sensordata[52:55].astype(np.float32)
            bh  = np.array([float(self.d.qpos[2])], dtype=np.float32)
            fc  = np.ones(4, dtype=np.float32)

            obs = np.concatenate([
                pg, jp, av, jv, lv,
                np.zeros(3, dtype=np.float32),   # vel_cmd = 0
                tq, fc, bh, des_c, ref_z, ref_x, ref_y,
            ]).astype(np.float32)
            obs = np.clip(obs, -OBS_CLIP, OBS_CLIP)

            with torch.no_grad():
                act = self.policy(
                    torch.from_numpy(obs).unsqueeze(0)
                ).squeeze(0).numpy().astype(np.float32)

            tgt_i = DEFAULT_JOINT_POS_INTERNAL + act * ACTION_SCALE
            tgt_m = tgt_i[INTERNAL_TO_MUJOCO]

            for i in range(NUM_JOINTS):
                q  = float(self.d.qpos[7 + i])
                dq = float(self.d.qvel[6 + i])
                tau[i] = float(np.clip(
                    (tgt_m[i] - q) * 25.0 - dq * 0.5, -23.5, 23.5
                ))

            for _ in range(DECIMATION):
                self.d.ctrl[:] = tau
                mujoco.mj_step(self.m, self.d)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, gait_id: int, vel_cmd, seed: int = 0):
        """
        Prepare a new episode.

        Args:
            gait_id:  integer 0–7
            vel_cmd:  [vx, vy, wz] float array-like
            seed:     RNG seed for initial pose noise
        """
        self._gait_id = int(gait_id)
        self._vel_cmd = np.array(vel_cmd, dtype=np.float32)

        # Stand gait always gets zero command regardless of what was passed
        if self._gait_id == 6:
            self._vel_cmd = np.zeros(3, dtype=np.float32)

        self._reset_to_policy_stand(seed)

        # Raibert: start half-period in, matching simulator convention
        self._raibert         = RaibertGait(gait_id=self._gait_id)
        self._raibert._t_exec = self._raibert._period * 0.5

        self._step_count = 0
        self._episode_t  = 0.0
        self._tau        = np.zeros(NUM_JOINTS, dtype=np.float32)

        # Clear accumulators
        self._buf_vx_actual    = []
        self._buf_vy_actual    = []
        self._buf_wz_actual    = []
        self._buf_vx_error     = []
        self._buf_vy_error     = []
        self._buf_wz_error     = []
        self._buf_height       = []
        self._buf_roll         = []
        self._buf_pitch        = []
        self._buf_contact_frac = []
        self._buf_contact_acc  = []
        self._buf_torque_norm  = []

        self._start_qpos_xy    = self.d.qpos[0:2].copy()
        self._low_height_count = 0
        self._settle_steps     = GAIT_SETTLE_STEPS.get(self._gait_id, DEFAULT_SETTLE_STEPS)

        self.done               = False
        self.termination_reason = "running"

    def step(self):
        """
        Run one policy step (= DECIMATION physics steps).
        Updates self.done and self.termination_reason on termination.
        """
        if self.done:
            return

        # ── Apply current tau for DECIMATION physics steps ────────────────
        for _ in range(DECIMATION):
            self.d.ctrl[:] = self._tau
            mujoco.mj_step(self.m, self.d)

        # ── Read state ────────────────────────────────────────────────────
        joint_pos_mj = self.d.qpos[7:19].astype(np.float32)
        joint_vel_mj = self.d.qvel[6:18].astype(np.float32)
        torques_mj   = np.clip(
            self.d.sensordata[24:36].astype(np.float32), -23.5, 23.5
        )

        # Reorder MuJoCo → Isaac internal (identity here, but kept explicit)
        joint_pos = joint_pos_mj[MUJOCO_TO_INTERNAL]
        joint_vel = joint_vel_mj[MUJOCO_TO_INTERNAL]
        torques   = torques_mj[MUJOCO_TO_INTERNAL]

        base_height  = float(self.d.qpos[2])
        q_wxyz       = self.d.qpos[3:7].astype(np.float32)
        proj_grav    = quat_rotate_inverse(q_wxyz, GRAVITY_W)
        ang_vel_b    = self.d.sensordata[40:43].astype(np.float32)
        lin_vel_b    = self.d.sensordata[52:55].astype(np.float32)
        foot_contact = read_foot_contact(self.d, self.m, self._calf_body_ids)
        roll, pitch, _ = quat_to_rpy(q_wxyz)

        # ── Termination checks ────────────────────────────────────────────
        # Only terminate on physical falls (sustained low height).
        # Orientation, velocity divergence, and position error are recorded
        # as metrics but never cause termination.
        #
        # During the settle window: reset the low-height counter so startup
        # transients (robot transitioning from standing to gait rhythm) never
        # trigger fall detection. Pronk/hop have longer settle windows.
        if self._step_count < self._settle_steps:
            self._low_height_count = 0
        elif base_height < FALL_HEIGHT:
            self._low_height_count += 1
        else:
            self._low_height_count = 0

        if self._low_height_count >= FALL_HEIGHT_GRACE:
            self.done               = True
            self.termination_reason = "fall_height"
        elif self._step_count >= MAX_EPISODE_STEPS - 1:
            self.done               = True
            self.termination_reason = "timeout"

        # ── Gait obs ──────────────────────────────────────────────────────
        # Stand gait zeros the command in the obs; other gaits pass it through.
        vel_cmd_obs = (
            np.zeros(3, dtype=np.float32)
            if self._gait_id == 6
            else self._vel_cmd
        )
        gait_obs = self._raibert.step(
            v_B=lin_vel_b, v_cmd=self._vel_cmd, q_wxyz=q_wxyz
        )

        # ── Accumulate metrics (skip gait-specific settle window) ───────────
        if self._step_count >= self._settle_steps:
            self._buf_height.append(base_height)
            self._buf_roll.append(abs(roll))
            self._buf_pitch.append(abs(pitch))

            self._buf_vx_actual.append(float(lin_vel_b[0]))
            self._buf_vy_actual.append(float(lin_vel_b[1]))
            self._buf_wz_actual.append(float(ang_vel_b[2]))

            self._buf_vx_error.append(abs(float(lin_vel_b[0]) - float(self._vel_cmd[0])))
            self._buf_vy_error.append(abs(float(lin_vel_b[1]) - float(self._vel_cmd[1])))
            self._buf_wz_error.append(abs(float(ang_vel_b[2]) - float(self._vel_cmd[2])))

            self._buf_contact_frac.append(float(foot_contact.mean()))

            # Contact accuracy: fraction of feet matching desired contact state
            des = gait_obs["desFeetContact"]
            acc = float(np.mean((foot_contact > 0.5) == (des > 0.5)))
            self._buf_contact_acc.append(acc)

            self._buf_torque_norm.append(float(np.linalg.norm(torques_mj)))

        # ── Build observation ─────────────────────────────────────────────
        obs = np.concatenate([
            proj_grav,                                      # 3
            joint_pos,                                      # 12
            ang_vel_b,                                      # 3
            joint_vel,                                      # 12
            lin_vel_b,                                      # 3
            vel_cmd_obs,                                    # 3
            torques,                                        # 12
            foot_contact,                                   # 4
            np.array([base_height], dtype=np.float32),     # 1
            gait_obs["desFeetContact"],                     # 4
            gait_obs["refFootZ"],                           # 4
            gait_obs["refFootX"],                           # 4
            gait_obs["refFootY"],                           # 4
        ], dtype=np.float32)                                # total = 69

        # ── Policy inference ──────────────────────────────────────────────
        obs_clipped = np.clip(obs, -OBS_CLIP, OBS_CLIP)
        with torch.no_grad():
            action_raw = self.policy(
                torch.from_numpy(obs_clipped).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)

        # ── Compute tau: internal order → MuJoCo order ────────────────────
        target_internal = DEFAULT_JOINT_POS_INTERNAL + action_raw * ACTION_SCALE
        target_mujoco   = target_internal[INTERNAL_TO_MUJOCO]

        for i in range(NUM_JOINTS):
            q  = float(self.d.qpos[7 + i])
            dq = float(self.d.qvel[6 + i])
            self._tau[i] = np.clip(
                (target_mujoco[i] - q) * 25.0 - dq * 0.5,
                -23.5, 23.5,
            )

        self._episode_t  += STEP_DT
        self._step_count += 1

    def get_episode_metrics(self) -> dict:
        """
        Return a flat dict of episode-level metrics.
        Safe to call mid-episode (partial metrics) or after done.
        """
        survived = self.termination_reason == "timeout"

        def _mean(buf): return float(np.mean(buf)) if buf else float("nan")
        def _std(buf):  return float(np.std(buf))  if buf else float("nan")

        xy_drift = float(np.linalg.norm(self.d.qpos[0:2] - self._start_qpos_xy))

        return {
            # ── Identity ──────────────────────────────────────────────────
            "gait_id":            self._gait_id,
            "gait_name":          GAIT_TABLE[self._gait_id]["name"],
            "vx_cmd":             float(self._vel_cmd[0]),
            "vy_cmd":             float(self._vel_cmd[1]),
            "wz_cmd":             float(self._vel_cmd[2]),

            # ── Termination ───────────────────────────────────────────────
            "survived":           survived,
            "survival_steps":     self._step_count,
            "survival_time_s":    self._step_count * STEP_DT,
            "termination_reason": self.termination_reason,

            # ── Velocity tracking ─────────────────────────────────────────
            "mean_vx_actual":     _mean(self._buf_vx_actual),
            "mean_vy_actual":     _mean(self._buf_vy_actual),
            "mean_wz_actual":     _mean(self._buf_wz_actual),
            "mean_vx_error":      _mean(self._buf_vx_error),
            "mean_vy_error":      _mean(self._buf_vy_error),
            "mean_wz_error":      _mean(self._buf_wz_error),

            # ── Stability ─────────────────────────────────────────────────
            "mean_height":        _mean(self._buf_height),
            "std_height":         _std(self._buf_height),
            "mean_roll":          _mean(self._buf_roll),
            "mean_pitch":         _mean(self._buf_pitch),

            # ── Contact quality ───────────────────────────────────────────
            "mean_contact_frac":  _mean(self._buf_contact_frac),
            "mean_contact_acc":   _mean(self._buf_contact_acc),

            # ── Effort ────────────────────────────────────────────────────
            "mean_torque_norm":   _mean(self._buf_torque_norm),

            # ── Positional drift (most meaningful for zero-cmd) ───────────
            "xy_drift_m":         xy_drift,
        }