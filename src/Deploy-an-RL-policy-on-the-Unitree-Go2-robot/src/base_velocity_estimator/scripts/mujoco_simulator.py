#!/usr/bin/env python3

import math
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowCmd, LowState

project_root = Path(__file__).parents[4]

# ── Policy constants ─────────────────────────────────────────────────────────
POLICY_PATH    = str(project_root / "resources" / "go2" / "policy.pt")
NUM_JOINTS     = 12
ACTION_SCALE   = 0.25
STEP_DT        = 0.010   # policy dt = 10ms = 100Hz
PHYSICS_DT     = 0.002   # physics timestep = 2ms = 500Hz (matches training sim.dt)
DECIMATION     = 5       # call policy every 5 physics steps = 100Hz
OBS_CLIP       = 100.0

GAIT_SCHEDULE = [
    (100, 6, [0.0, 0.0, 0.0]),  # trot at 0.5 m/s
    (99999, 1, [1.2, 0.0, 0.0]),
]

# ── Joint order mapping ───────────────────────────────────────────────────────
# MuJoCo XML joint order (FR/FL/RR/RL):
#   0=FR_hip  1=FR_thigh  2=FR_calf
#   3=FL_hip  4=FL_thigh  5=FL_calf
#   6=RR_hip  7=RR_thigh  8=RR_calf
#   9=RL_hip 10=RL_thigh 11=RL_calf
#
# Isaac internal order (FL/FR/RL/RR, alphabetical asset load order):
#   0=FL_hip  1=FR_hip  2=RL_hip  3=RR_hip
#   4=FL_thigh 5=FR_thigh 6=RL_thigh 7=RR_thigh
#   8=FL_calf  9=FR_calf 10=RL_calf 11=RR_calf
#
# asset_cfg.joint_ids = [1,5,9,0,4,8,3,7,11,2,6,10] was used during training
# to reorder Isaac internal → logical FR/FL/RR/RL for the obs, AND to apply
# actions back via set_joint_position_target(..., joint_ids=...).
# The policy therefore learned: action[i] controls Isaac_internal[joint_ids[i]].
#
# MUJOCO_TO_INTERNAL[i] = Isaac internal index for MuJoCo joint i
# (used to reorder obs joint arrays: MuJoCo order → Isaac internal order)
MUJOCO_TO_INTERNAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# INTERNAL_TO_MUJOCO[i] = MuJoCo index for Isaac internal joint i
# (used to reorder action output: Isaac internal order → MuJoCo order)
INTERNAL_TO_MUJOCO = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

# Default joint positions in Isaac INTERNAL order
# (what the policy offset was trained with, via use_default_offset=True)
# Isaac internal order: FL_hip, FR_hip, RL_hip, RR_hip,
#                       FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#                       FL_calf, FR_calf, RL_calf, RR_calf
DEFAULT_JOINT_POS_INTERNAL = np.array(
    [0.1,  -0.1, 0.1,  -0.1,   # hip:   FL, FR, RL, RR
      0.8,  0.8,  1.0,  1.0,   # thigh: FL, FR, RL, RR
     -1.5, -1.5, -1.5, -1.5],  # calf:  FL, FR, RL, RR
    dtype=np.float32,
)

# Default joint positions in MuJoCo order (for PD standup / debug comparisons)
DEFAULT_JOINT_POS_TRAINING = np.array(
    [+0.1,  0.8, -1.5,   # FR: hip, thigh, calf  (note: FR hip = +0.1? see below)
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

MIN_HEIGHT_FOR_ACTIVATION = 0.25
MAX_ANGVEL_FOR_ACTIVATION = 0.3


# ── Gait utilities ───────────────────────────────────────────────────────────

def quat_rotate_inverse(q_wxyz, v):
    """Rotate vector v from world frame into body frame (inverse rotation)."""
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


class RaibertGait:
    """
    Isaac-matching Raibert gait with:
      - Phase continuity on gait switch (no phase jump)
      - Smooth period and z_nom blending across gait transitions (alpha=0.1)
      - Foot reference updated only at liftoff (same as Isaac)
      - Swing height via (1 - cos) profile (same as Isaac)
    """
    STEP_HEIGHT = 0.10
    BLEND_ALPHA = 0.1   # matches Isaac's blend_alpha

    def __init__(self, gait_id: int = 6):
        g = GAIT_TABLE[gait_id]
        self._gait_id = gait_id

        # Target gait params (updated instantly on switch)
        self._period    = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset    = np.array(g["offset"], dtype=np.float64)
        self._k         = float(g["k"])
        self._z_nom     = float(g["z_nom"])
        self._x_lim     = float(g["x_lim"])
        self._y_lim     = float(g["y_lim"])

        # Blended params (smoothly follow target, never hard-reset on switch)
        self._period_blended = self._period
        self._znom_blended   = self._z_nom
        self._old_period_blended  = self._period_blended
        self._gait_just_switched  = False

        # Phase state.
        self._t_exec             = 0.0
        self._phase_compensation = 0.0

        # Foot placement state
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
        # Save old blended period BEFORE updating anything
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

    def get_p_ref_B(self):
        return self._p_ref_B.copy()

    def step(self, v_B, v_cmd, q_wxyz):
        a = self.BLEND_ALPHA
        # 1) blend FIRST
        self._period_blended = a * self._period + (1.0 - a) * self._period_blended
        self._znom_blended   = a * self._z_nom  + (1.0 - a) * self._znom_blended

        # 2) phase compensation AFTER blend, matching Isaac's order exactly
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
        new_p[:, 0] = np.clip(new_p[:, 0],
                            HIP_POS_B[:, 0] - self._x_lim,
                            HIP_POS_B[:, 0] + self._x_lim)
        new_p[:, 1] = np.clip(new_p[:, 1],
                            HIP_POS_B[:, 1] - self._y_lim,
                            HIP_POS_B[:, 1] + self._y_lim)

        liftoff = (self._prev_c > 0.5) & (c_ref < 0.5)
        for leg in range(4):
            if liftoff[leg]:
                self._p_ref_B[leg] = new_p[leg]

        swing_mask = (c_ref < 0.5).astype(np.float64)
        x_sw = np.clip((leg_phase - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
        z_sw = 0.5 * self.STEP_HEIGHT * (1.0 - np.cos(2.0 * math.pi * x_sw)) * swing_mask

        pf = self._p_ref_B.copy()
        pf[:, 2] = z_nom + z_sw
        R  = quat_to_rotmat(q_wxyz)
        pw = (R @ pf.T).T

        self._prev_c  = c_ref.copy()
        self._t_exec += STEP_DT

        return {
            "desFeetContact": c_ref.astype(np.float32),
            "refFootZ":       pw[:, 2].astype(np.float32),
            "refFootX":       pw[:, 0].astype(np.float32),
            "refFootY":       pw[:, 1].astype(np.float32),
        }


def get_schedule_entry(episode_t):
    elapsed = 0.0
    for i, (duration, gait_id, vel_cmd) in enumerate(GAIT_SCHEDULE):
        dur_s = duration * STEP_DT
        elapsed += dur_s
        if episode_t < elapsed or i == len(GAIT_SCHEDULE) - 1:
            return gait_id, np.array(vel_cmd, dtype=np.float32)
    last = GAIT_SCHEDULE[-1]
    return last[1], np.array(last[2], dtype=np.float32)


def _read_foot_contact(d, m, foot_body_ids):
    """Read foot contact from MuJoCo collision data. Call with lock held."""
    foot_contact = np.zeros(4, dtype=np.float32)
    ncon = int(d.ncon)
    for con in range(ncon):
        c  = d.contact[con]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        for i, bid in enumerate(foot_body_ids):
            if b1 == bid or b2 == bid:
                foot_contact[i] = 1.0
    return foot_contact


# ── Main simulator node ──────────────────────────────────────────────────────

class MujocoSimulator(Node):
    def __init__(self):
        super().__init__("mujoco_simulator")

        # ── Publishers ─────────────────────────────────────────────────────
        self.low_state_puber  = self.create_publisher(LowState,          "/mujoco/lowstate",      10)
        self.pos_pub          = self.create_publisher(Float32MultiArray, "/mujoco/pos",           10)
        self.force_pub        = self.create_publisher(Float32MultiArray, "/mujoco/force",         10)
        self.torque_pub       = self.create_publisher(Float32MultiArray, "/mujoco/torque",        10)
        self.base_lin_vel_pub = self.create_publisher(Float32MultiArray, "/mujoco/base_lin_vel_b",10)
        self.base_height_pub  = self.create_publisher(Float32MultiArray, "/mujoco/base_height",   10)
        self.foot_contact_pub = self.create_publisher(Float32MultiArray, "/mujoco/foot_contact",  10)

        # ── Subscriptions ──────────────────────────────────────────────────
        self.lowcmd_sub = self.create_subscription(
            LowCmd, "/mujoco/lowcmd", self.lowcmd_callback, 10)
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)

        # ── MuJoCo setup ───────────────────────────────────────────────────
        self.xml_path = project_root / "resources" / "go2" / "scene_flat.xml"
        self.foot_body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.foot_body_ids = []
        self.calf_body_ids = []
        self.init_mujoco()

        # ── Low-level control state ────────────────────────────────────────
        self.target_dof_pos = [0.0] * 12
        self.tau            = np.zeros(12, dtype=np.float32)
        self.kps            = np.array([25.0] * 12, dtype=np.float32)
        self.kds            = np.array([0.5]  * 12, dtype=np.float32)
        self.received_data  = False

        # ── Policy setup ───────────────────────────────────────────────────
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.get_logger().info(f"Loading policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH, map_location="cpu")
        self.policy.eval()
        self.get_logger().info("Policy ready.")

        # ── Policy state ───────────────────────────────────────────────────
        self._policy_active   = False
        self._step_count      = 0
        self._episode_t       = 0.0
        self._physics_count   = 0
        self._current_gait_id = GAIT_SCHEDULE[0][1]
        self.raibert          = RaibertGait(gait_id=GAIT_SCHEDULE[0][1])

        # ── Threading ──────────────────────────────────────────────────────
        self._mujoco_lock = threading.Lock()

        self.running = True
        self.timer_sensor = self.create_timer(0.005, self.publish_sensor_data)
        self.timer_tau    = self.create_timer(0.001, self.update_tau)
        self.sim_thread   = threading.Thread(target=self.step_simulation, daemon=True)
        self.sim_thread.start()

        self.debug_count = 0
        self.get_logger().info("MujocoSimulator ready. Stand up robot then activate with LB+RB.")

    def init_mujoco(self):
        self.m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = PHYSICS_DT
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

        for name in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            self.foot_body_ids.append(
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name))

        for name in ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]:
            self.calf_body_ids.append(
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name))

        self.get_logger().info("MuJoCo initialized.")

    def lowcmd_callback(self, msg: LowCmd):
        if not self._policy_active:
            self.received_data = True
            for i in range(12):
                self.target_dof_pos[i] = float(msg.motor_cmd[i].q)
                self.kps[i] = float(msg.motor_cmd[i].kp)
                self.kds[i] = float(msg.motor_cmd[i].kd)

    def _joy_cb(self, msg):
        lb = len(msg.buttons) > 4 and msg.buttons[4]
        rb = len(msg.buttons) > 5 and msg.buttons[5]
        want_active = bool(lb and rb)
        was_active  = self._policy_active

        if want_active and not was_active:
            height       = float(self.d.qpos[2])
            ang_vel_norm = float(np.linalg.norm(self.d.sensordata[40:43]))

            if height < MIN_HEIGHT_FOR_ACTIVATION:
                self.get_logger().warn(f"Activation REJECTED: height={height:.3f}m")
                return
            if ang_vel_norm > MAX_ANGVEL_FOR_ACTIVATION:
                self.get_logger().warn(f"Activation REJECTED: |ang_vel|={ang_vel_norm:.3f}")
                return

            print(f"\n=== POSE AT ACTIVATION ===")
            for i in range(12):
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i+1)
                print(f"  {name}: {self.d.qpos[7+i]:.4f}")
            print(f"  height: {self.d.qpos[2]:.4f}")
            print("=== END ===\n")

            self._policy_active   = True
            self._step_count      = 0
            self._episode_t       = 0.0
            self._physics_count   = 0
            self._current_gait_id = GAIT_SCHEDULE[0][1]
            self.raibert          = RaibertGait(gait_id=GAIT_SCHEDULE[0][1])
            self.raibert._t_exec  = self.raibert._period * 0.5

            self.get_logger().info(
                f"Policy ACTIVATED — height={height:.3f}m  "
                f"starting with {GAIT_TABLE[GAIT_SCHEDULE[0][1]]['name']} gait  "
                f"t0={self.raibert._t_exec:.3f}s")

        elif not want_active and was_active:
            self._policy_active = False
            self.get_logger().info("Policy DEACTIVATED.")

    def update_tau(self):
        if not self.received_data or self._policy_active:
            return
        for i in range(12):
            q  = self.d.qpos[7 + i]
            dq = self.d.qvel[6 + i]
            self.tau[i] = np.clip(
                self.pd_control(self.target_dof_pos[i], q, self.kps[i], dq, self.kds[i]),
                -23.5, 23.5)

    def _run_policy(self):
        # Called from step_simulation which already holds _mujoco_lock.

        # ── Raw MuJoCo state (MuJoCo order: FR/FL/RR/RL) ───────────────────
        joint_pos_mujoco = self.d.qpos[7:19].astype(np.float32)
        joint_vel_mujoco = self.d.qvel[6:18].astype(np.float32)
        torques_mujoco   = np.clip(self.d.sensordata[24:36].astype(np.float32), -23.5, 23.5)

        # ── Reorder joint arrays from MuJoCo order → Isaac internal order ──
        # This matches what the policy saw during training via asset_cfg.joint_ids
        joint_pos = joint_pos_mujoco[MUJOCO_TO_INTERNAL]
        joint_vel = joint_vel_mujoco[MUJOCO_TO_INTERNAL]
        torques   = torques_mujoco[MUJOCO_TO_INTERNAL]

        # Warm-start: use Isaac steady-state torques for first 5 steps
        # (these should also be in Isaac internal order)
        ISAAC_WARMSTART_TORQUES = np.array([
             2.5872,  7.6188, -1.6298,  0.2407,   # FL_thigh→FR... reordered below
            -1.9906,  7.5874, -3.6953,  2.3493,
             2.5872,  7.5874, -1.6298,  2.7453
        ], dtype=np.float32)
        # Warmstart torques from Isaac in MuJoCo order, then reorder:
        ISAAC_WARMSTART_TORQUES_MUJOCO = np.array([
             7.6188,  2.5872,  7.5874,
            -7.4888, -1.9906,  6.4493,
            -0.5166, -3.6953,  2.3493,
             0.2407, -1.6298,  2.7453
        ], dtype=np.float32)
        if self._step_count < 5:
            torques = ISAAC_WARMSTART_TORQUES_MUJOCO[MUJOCO_TO_INTERNAL]

        base_height = np.array([float(self.d.qpos[2])], dtype=np.float32)

        # ── Orientation ─────────────────────────────────────────────────────
        q_wxyz = self.d.qpos[3:7].astype(np.float32)

        # ── Observations that depend on orientation ──────────────────────────
        proj_grav = quat_rotate_inverse(q_wxyz, GRAVITY_W)
        ang_vel_b = self.d.sensordata[40:43].astype(np.float32)
        lin_vel_b = self.d.sensordata[52:55].astype(np.float32)

        # ── Foot contact — MuJoCo collision detection ────────────────────────
        foot_contact = _read_foot_contact(self.d, self.m, self.calf_body_ids)

        # ── Gait obs ────────────────────────────────────────────────────────
        gait_id, vel_cmd = get_schedule_entry(self._episode_t)
        if gait_id != self._current_gait_id:
            self.get_logger().info(
                f"Gait switch: {GAIT_TABLE[self._current_gait_id]['name']} → "
                f"{GAIT_TABLE[gait_id]['name']}  (ep_t={self._episode_t:.2f}s)")
            self.raibert.switch_gait(gait_id)
            self._current_gait_id = gait_id

        vel_cmd_obs = np.zeros(3, dtype=np.float32) if gait_id == 6 else vel_cmd

        gait_obs = self.raibert.step(v_B=lin_vel_b, v_cmd=vel_cmd, q_wxyz=q_wxyz)

        # ── Debug prints ────────────────────────────────────────────────────
        if self._step_count == 0:
            print("\n=== OBS AT ACTIVATION (step 0) ===")
            labels = [
                ("proj_grav",      proj_grav),
                ("joint_pos",      joint_pos),
                ("ang_vel_b",      ang_vel_b),
                ("joint_vel",      joint_vel),
                ("lin_vel_b",      lin_vel_b),
                ("vel_cmd_obs",    vel_cmd_obs),
                ("torques",        torques),
                ("foot_contact",   foot_contact),
                ("base_height",    base_height),
                ("desFeetContact", gait_obs["desFeetContact"]),
                ("refFootZ",       gait_obs["refFootZ"]),
                ("refFootX",       gait_obs["refFootX"]),
                ("refFootY",       gait_obs["refFootY"]),
            ]
            for name, val in labels:
                print(f"  {name:16s} min={val.min():+.3f} max={val.max():+.3f} "
                      f"vals={np.round(val, 3)}")
            print(f"  q_wxyz: {np.round(q_wxyz, 4)}")
            print(f"  t_exec={self.raibert._t_exec:.4f}s  "
                  f"phase_comp={self.raibert._phase_compensation:.4f}s  "
                  f"period_blended={self.raibert._period_blended:.4f}  "
                  f"znom_blended={self.raibert._znom_blended:.4f}")
            print(f"  p_ref_B: {self.raibert.get_p_ref_B().round(4).tolist()}")
            print("===================================\n")

        if 0 <= self._step_count < 20:
            print(
                f"[s{self._step_count:02d}] "
                f"h={base_height[0]:.3f} "
                f"grav=[{proj_grav[0]:+.3f},{proj_grav[1]:+.3f},{proj_grav[2]:+.3f}] "
                f"w=[{ang_vel_b[0]:+.3f},{ang_vel_b[1]:+.3f},{ang_vel_b[2]:+.3f}] "
                f"desC={gait_obs['desFeetContact'].astype(int).tolist()} "
                f"refZ=[{' '.join(f'{v:+.3f}' for v in gait_obs['refFootZ'])}]"
            )

        if 300 <= self._step_count < 350:
            print(
                f"refX=[{' '.join(f'{v:+.3f}' for v in gait_obs['refFootX'])}] "
                f"refY=[{' '.join(f'{v:+.3f}' for v in gait_obs['refFootY'])}] "
                f"refZ=[{' '.join(f'{v:+.3f}' for v in gait_obs['refFootZ'])}]"
            )

        if self._step_count == 100:
            print("\n=== SETTLED POSE (step 100) ===")
            print(f"joint_pos (internal): {joint_pos.round(3).tolist()}")
            print(f"joint_pos (mujoco):   {joint_pos_mujoco.round(3).tolist()}")
            print(f"height:    {base_height[0]:.3f}")
            print(f"grav:      {proj_grav.round(3).tolist()}")
            print("=== END ===\n")

        if self._step_count % 50 == 0:
            print(f"[step {self._step_count}] gait={GAIT_TABLE[gait_id]['name']} vel_cmd={vel_cmd}")

        # ── Assemble observation (joint arrays are now in Isaac internal order) ─
        obs = np.concatenate([
            proj_grav,                       # 3
            joint_pos,                       # 12  (Isaac internal order)
            ang_vel_b,                       # 3
            joint_vel,                       # 12  (Isaac internal order)
            lin_vel_b,                       # 3
            vel_cmd_obs,                     # 3
            torques,                         # 12  (Isaac internal order)
            foot_contact,                    # 4
            base_height,                     # 1
            gait_obs["desFeetContact"],      # 4
            gait_obs["refFootZ"],            # 4
            gait_obs["refFootX"],            # 4
            gait_obs["refFootY"],            # 4
        ], dtype=np.float32)                 # total = 69

        if self._step_count % 100 == 0:
            print(f"step={self._step_count} gait={GAIT_TABLE[gait_id]['name']} "
                  f"h={base_height[0]:.3f} grav={proj_grav.round(3)} "
                  f"period_b={self.raibert._period_blended:.3f} "
                  f"znom_b={self.raibert._znom_blended:.3f} "
                  f"desC={gait_obs['desFeetContact'].astype(int).tolist()}")
            print(f"  joint_pos (mujoco)   : {joint_pos_mujoco.round(3).tolist()}")
            print(f"  joint_pos (internal) : {joint_pos.round(3).tolist()}")
            print(f"  joint_vel (internal) : {joint_vel.round(3).tolist()}")
            print(f"  default   (internal) : {DEFAULT_JOINT_POS_INTERNAL.round(3).tolist()}")
            print(f"  jp_error  (internal) : {(joint_pos - DEFAULT_JOINT_POS_INTERNAL).round(3).tolist()}")
            print(f"  foot_z_w: {np.array([self.d.xpos[bid][2] for bid in self.foot_body_ids]).round(4)}")
            print(f"  ncon: {int(self.d.ncon)}")
            for con in range(int(self.d.ncon)):
                c = self.d.contact[con]
                b1 = self.m.geom_bodyid[c.geom1]
                b2 = self.m.geom_bodyid[c.geom2]
                n1 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b1)
                n2 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b2)
                print(f"    contact {con}: {n1} vs {n2}")

        obs_clipped = np.clip(obs, -OBS_CLIP, OBS_CLIP)

        with torch.no_grad():
            action_raw = self.policy(
                torch.from_numpy(obs_clipped).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)

        # ── Step 0 comparison (Isaac internal order now) ─────────────────────
        if self._step_count == 0:
            obs_np    = obs_clipped
            action_np = action_raw

            names_sizes = [
                ("proj_grav",3),("joint_pos",12),("ang_vel",3),
                ("joint_vel",12),("lin_vel",3),("vel_cmd",3),
                ("torques",12),("foot_contact",4),("base_height",1),
                ("desFeetContact",4),("refFootZ",4),("refFootX",4),("refFootY",4)
            ]
            # Isaac reference obs — joint slots are in Isaac internal order
            isaac = {
                "proj_grav":      [0.1846, -0.0203, -0.9826],
                "joint_pos":      [-0.1791, 0.6807, -1.8039, 0.1286, 0.7553, -1.7987, -0.1203, 0.6998, -1.3073, 0.0647, 0.7118, -1.2939],
                "ang_vel":        [0.0047, 0.0024, -0.003],
                "joint_vel":      [0.001, 0.0013, -0.0117, 0.0027, -0.0041, 0.0026, -0.0043, -0.0036, -0.0019, -0.003, -0.0018, 0.0007],
                "lin_vel":        [-0.0003, -0.0013, -0.0007],
                "vel_cmd":        [0.0, 0.0, 0.0],
                "torques":        [7.6188, 2.5872, 7.5874, -7.4888, -1.9906, 6.4493, -0.5166, -3.6953, 2.3493, 0.2407, -1.6298, 2.7453],
                "foot_contact":   [1.0, 1.0, 1.0, 1.0],
                "base_height":    [0.3062],
                "desFeetContact": [1.0, 1.0, 1.0, 1.0],
                "refFootZ":       [-0.3507, -0.3457, -0.2831, -0.2782],
                "refFootX":       [0.111, 0.131, -0.2477, -0.2277],
                "refFootY":       [-0.1244, 0.1187, -0.0963, 0.1468],
            }
            isaac_action = [-1.0837, 0.9025, -0.1029, -0.1647, -0.4976, -0.0629, -1.4143, -1.7925, -0.1626, -0.0024, 1.2642, 1.1465]

            print("\n" + "="*70)
            print("POLICY STEP 0 — OBS COMPARISON (MuJoCo vs Isaac steady state)")
            print("="*70)
            idx = 0
            for name, size in names_sizes:
                mj_vals  = obs_np[idx:idx+size]
                ref_vals = np.array(isaac[name])
                diff     = mj_vals - ref_vals
                max_diff = float(np.abs(diff).max())
                flag     = " <<<" if max_diff > 0.05 else ""
                print(f"\n  {name} (max_diff={max_diff:.4f}){flag}")
                print(f"    MuJoCo : {np.round(mj_vals, 4).tolist()}")
                print(f"    Isaac  : {np.round(ref_vals, 4).tolist()}")
                print(f"    diff   : {np.round(diff, 4).tolist()}")
                idx += size

            # Action is in Isaac internal order
            int_names = ['FL_hip','FR_hip','RL_hip','RR_hip',
                         'FL_thigh','FR_thigh','RL_thigh','RR_thigh',
                         'FL_calf','FR_calf','RL_calf','RR_calf']
            print("\n" + "="*70)
            print("ACTION COMPARISON (Isaac internal order)")
            print("="*70)
            print(f"  {'Joint':<12} {'Isaac':>8} {'MuJoCo':>8} {'Diff':>8}")
            print("  " + "-"*40)
            for n, a, e in zip(int_names, action_np, isaac_action):
                flag = " <<<" if abs(a-e) > 0.1 else ""
                print(f"  {n:<12} {e:>8.4f} {a:>8.4f} {a-e:>8.4f}{flag}")
            print("="*70 + "\n")

        if 0 <= self._step_count < 20:
            print(f"       act=[{' '.join(f'{v:+.3f}' for v in action_raw)}]")

        # ── Compute target positions and apply to MuJoCo ─────────────────────
        # action_raw is in Isaac internal order.
        # DEFAULT_JOINT_POS_INTERNAL is also in Isaac internal order.
        # Compute target in internal order, then reindex to MuJoCo order.
        target_internal = DEFAULT_JOINT_POS_INTERNAL + action_raw * ACTION_SCALE
        target_mujoco   = target_internal[INTERNAL_TO_MUJOCO]

        if self._step_count < 5:
            if self._step_count > 0 and hasattr(self, '_prev_jpos_debug'):
                delta = joint_pos_mujoco - self._prev_jpos_debug
                print(f"\n=== MUJOCO STEP {self._step_count} JOINT MOTION ===")
                print(f"prev_target (mujoco): {np.round(self._prev_target_debug, 3).tolist()}")
                print(f"prev_jpos   (mujoco): {np.round(self._prev_jpos_debug, 3).tolist()}")
                print(f"curr_jpos   (mujoco): {np.round(joint_pos_mujoco, 3).tolist()}")
                print(f"delta:                {np.round(delta, 3).tolist()}")
                print(f"max_delta:            {np.abs(delta).max():.4f} rad")
                print("=== END ===\n")
            self._prev_jpos_debug   = joint_pos_mujoco.copy()
            self._prev_target_debug = target_mujoco.copy()

        # Apply PD control in MuJoCo order
        for i in range(12):
            q  = self.d.qpos[7 + i]
            dq = self.d.qvel[6 + i]
            self.tau[i] = np.clip(
                self.pd_control(float(target_mujoco[i]), q, 25.0, dq, 0.5),
                -23.5, 23.5)

        self._episode_t  += STEP_DT
        self._step_count += 1

    def step_simulation(self):
        while self.viewer.is_running() and self.running:
            if not self.received_data and not self._policy_active:
                time.sleep(0.001)
                continue

            step_start = time.time()

            with self._mujoco_lock:
                self.d.ctrl[:] = self.tau
                mujoco.mj_step(self.m, self.d)
                self._physics_count += 1

                if self._policy_active and (self._physics_count % DECIMATION == 0):
                    self._run_policy()

            self.viewer.sync()

            time_until_next = PHYSICS_DT - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    @staticmethod
    def pd_control(target_q, q, kp, dq, kd):
        return (target_q - q) * kp - dq * kd

    def publish_sensor_data(self):
        with self._mujoco_lock:
            joint_pos   = self.d.qpos[7:19].copy().astype(np.float32)
            joint_vel   = self.d.qvel[6:18].copy().astype(np.float32)
            quat        = self.d.qpos[3:7].copy().astype(np.float32)
            gyro        = self.d.sensordata[40:43].copy().astype(np.float32)
            torque_data = self.d.sensordata[24:36].copy()
            lin_vel_b   = self.d.sensordata[52:55].copy().astype(np.float32)
            base_height = float(self.d.qpos[2])
            qpos_full   = self.d.qpos[:19].copy()
            f1 = self.d.sensordata[55:58].copy().astype(np.float32)
            f2 = self.d.sensordata[58:61].copy().astype(np.float32)
            f3 = self.d.sensordata[61:64].copy().astype(np.float32)
            f4 = self.d.sensordata[64:67].copy().astype(np.float32)
            foot_contact = _read_foot_contact(self.d, self.m, self.calf_body_ids)

        low_state_msg = LowState()
        for i in range(12):
            low_state_msg.motor_state[i].q  = float(joint_pos[i])
            low_state_msg.motor_state[i].dq = float(joint_vel[i])
            if hasattr(low_state_msg.motor_state[i], "tau_est"):
                low_state_msg.motor_state[i].tau_est = float(self.tau[i])

        torque_msg = Float32MultiArray()
        torque_msg.data = list(map(float, torque_data))
        self.torque_pub.publish(torque_msg)

        low_state_msg.imu_state.quaternion = quat
        low_state_msg.imu_state.gyroscope  = gyro
        self.low_state_puber.publish(low_state_msg)

        pos_msg = Float32MultiArray()
        pos_msg.data = qpos_full.tolist()
        self.pos_pub.publish(pos_msg)

        force_msg = Float32MultiArray()
        force_msg.data = np.concatenate([f1, f2, f3, f4]).tolist()
        self.force_pub.publish(force_msg)

        lin_msg = Float32MultiArray()
        lin_msg.data = lin_vel_b.tolist()
        self.base_lin_vel_pub.publish(lin_msg)

        height_msg = Float32MultiArray()
        height_msg.data = [base_height]
        self.base_height_pub.publish(height_msg)

        contact_msg = Float32MultiArray()
        contact_msg.data = foot_contact.tolist()
        self.foot_contact_pub.publish(contact_msg)

        self.debug_count += 1
        if self.debug_count % 200 == 0:
            torque_sensor = self.d.sensordata[24:36].copy()
            torque_norm = float(np.linalg.norm(torque_sensor))
            print(f"height={base_height:.3f}  "
                f"lin_vel={np.round(lin_vel_b, 3).tolist()}  "
                f"contact={foot_contact.tolist()}  "
                f"torque_norm={torque_norm:.2f}  "
                f"torques={np.round(torque_sensor, 2).tolist()}")
            base_xpos_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            base_pos = self.d.xpos[base_xpos_id].copy()
            base_quat = self.d.xquat[base_xpos_id].copy()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, base_quat)
            R = R.reshape(3, 3)
            foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
            print(f"[MUJOCO Body-Frame Foot Debug]")
            print(f"{'Leg':<6} | {'Real_Foot_B (x,y,z)':<30}")
            print("-" * 45)
            for name in foot_names:
                bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
                rel = self.d.xpos[bid] - base_pos
                foot_b = R.T @ rel
                print(f"  {name}: {np.round(foot_b, 3).tolist()}")

    def stop_simulation(self):
        self.running = False
        self.sim_thread.join()

    def destroy_node(self):
        try:
            self.stop_simulation()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_simulation()
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()