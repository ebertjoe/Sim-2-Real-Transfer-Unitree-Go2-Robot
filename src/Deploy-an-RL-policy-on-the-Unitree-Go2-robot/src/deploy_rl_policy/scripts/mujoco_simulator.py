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
POLICY_PATH    = str(project_root / "resources" / "go2" / "policy53.pt")
NUM_JOINTS     = 12
ACTION_SCALE   = 0.25
STEP_DT        = 0.010   # policy dt = 10ms = 100Hz
PHYSICS_DT     = 0.002   # physics timestep = 2ms = 500Hz
DECIMATION     = 5       # call policy every 5 physics steps = 100Hz
OBS_CLIP       = 100.0

GAIT_SCHEDULE = [
    (100,  6, [0.0, 0.0, 0.0]),
    (300,  0, [0.2, 0.0, 0.0]),
    (300,  1, [0.4, 0.1, 0.0]),
    (300,  2, [0.6, 0.2, 0.0]),
    (300,  3, [0.8, 0.3, 0.0]),
    (300,  4, [1.0, 0.3, 0.0]),
    (300,  5, [1.2, 0.4, 0.0]),
    (99999,7, [1.2, 0.4, 0.0]),
]

# ── Joint order mapping ───────────────────────────────────────────────────────
# MuJoCo XML joint order (FR/FL/RR/RL):
#   0=FR_hip  1=FR_thigh  2=FR_calf
#   3=FL_hip  4=FL_thigh  5=FL_calf
#   6=RR_hip  7=RR_thigh  8=RR_calf
#   9=RL_hip 10=RL_thigh 11=RL_calf
#
# Isaac internal order (FL/FR/RL/RR alphabetical):
#   0=FL_hip  1=FR_hip  2=RL_hip  3=RR_hip
#   4=FL_thigh 5=FR_thigh 6=RL_thigh 7=RR_thigh
#   8=FL_calf  9=FR_calf 10=RL_calf 11=RR_calf
MUJOCO_TO_INTERNAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
INTERNAL_TO_MUJOCO = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

DEFAULT_JOINT_POS_INTERNAL = np.array(
    [0.1,  -0.1, 0.1,  -0.1,
      0.8,  0.8,  1.0,  1.0,
     -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)

DEFAULT_JOINT_POS_TRAINING = np.array(
    [+0.1,  0.8, -1.5,
     -0.1,  0.8, -1.5,
     +0.1,  1.0, -1.5,
     -0.1,  1.0, -1.5],
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
    """Rotate vector v from world frame into body frame."""
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
    Raibert gait planner — adapted for real-robot deployment.

    Key change vs original: uses v_cmd instead of v_B for foot placement.
    The original formula was:
        dx = 0.5*Tst*v_B + k*(v_B - v_cmd)
    New formula:
        dx = 0.5*Tst*v_cmd
    This removes the dependency on base velocity measurement, making the
    planner fully deployable on the real Go2 which has no direct velocity sensor.
    The policy compensates for velocity errors through joint position/velocity
    feedback in the remaining observation terms.
    """
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

    def get_p_ref_B(self):
        return self._p_ref_B.copy()

    def step(self, v_cmd, q_wxyz):
        """
        Step the Raibert planner.

        Args:
            v_cmd:   commanded velocity [vx, vy, wz] — used for foot placement
            q_wxyz:  base orientation quaternion (w, x, y, z)

        Note: v_B (measured velocity) is intentionally NOT used here.
        Foot placement is based purely on v_cmd, matching the training change
        where v_cmd replaced v_B in observations.py beta_l_raibert().
        """
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
        z_nom = self._znom_blended
        Tst   = thr * T

        global_phase = ((self._t_exec - self._phase_compensation) % T) / T
        leg_phase    = (global_phase + self._offset) % 1.0
        c_ref        = (leg_phase < thr).astype(np.float64)

        # Foot placement: use v_cmd only (no measured velocity)
        dx = 0.5 * Tst * float(v_cmd[0])
        dy = 0.5 * Tst * float(v_cmd[1])

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
    """Read foot contact from MuJoCo collision data."""
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
        self.foot_contact_pub = self.create_publisher(Float32MultiArray, "/mujoco/foot_contact",  10)

        # ── Subscriptions ──────────────────────────────────────────────────
        self.lowcmd_sub = self.create_subscription(
            LowCmd, "/mujoco/lowcmd", self.lowcmd_callback, 10)
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)

        # ── MuJoCo setup ───────────────────────────────────────────────────
        self.xml_path = project_root / "resources" / "go2" / "scene_flat.xml"
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
        self.running      = True

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

            self._policy_active   = True
            self._step_count      = 0
            self._episode_t       = 0.0
            self._physics_count   = 0
            self._current_gait_id = GAIT_SCHEDULE[0][1]
            self.raibert          = RaibertGait(gait_id=GAIT_SCHEDULE[0][1])
            self.raibert._t_exec  = self.raibert._period * 0.5

            self.get_logger().info(
                f"Policy ACTIVATED — height={height:.3f}m  "
                f"starting with {GAIT_TABLE[GAIT_SCHEDULE[0][1]]['name']} gait")

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
        """
        Assemble 53-dim observation and run policy inference.

        Observation space (53 dims) — real-robot deployable:
            projected_gravity   3   IMU
            joint_pos          12   encoders (Isaac internal order)
            ang_vel             3   IMU
            joint_vel          12   encoders (Isaac internal order)
            vel_cmd             3   operator command
            foot_contact        4   foot contact sensors
            desFeetContact      4   Raibert schedule
            refFootZ            4   Raibert schedule
            refFootX            4   Raibert schedule
            refFootY            4   Raibert schedule
            ─────────────────────
            Total              53

        Removed vs original 69-dim policy:
            lin_vel      (3)  — no base velocity sensor on real robot
            joint_torques(12) — not directly available on Go2 hardware
            base_height  (1)  — no height sensor on real robot

        Raibert change: foot placement uses v_cmd instead of v_B,
        removing the velocity sensor dependency entirely.
        """
        # ── Raw MuJoCo state ───────────────────────────────────────────────
        joint_pos_mujoco = self.d.qpos[7:19].astype(np.float32)
        joint_vel_mujoco = self.d.qvel[6:18].astype(np.float32)

        # ── Reorder MuJoCo → Isaac internal order ──────────────────────────
        joint_pos = joint_pos_mujoco[MUJOCO_TO_INTERNAL]
        joint_vel = joint_vel_mujoco[MUJOCO_TO_INTERNAL]

        # ── Orientation ────────────────────────────────────────────────────
        q_wxyz    = self.d.qpos[3:7].astype(np.float32)
        proj_grav = quat_rotate_inverse(q_wxyz, GRAVITY_W)
        ang_vel_b = self.d.sensordata[40:43].astype(np.float32)

        # ── Foot contact ───────────────────────────────────────────────────
        foot_contact = _read_foot_contact(self.d, self.m, self.calf_body_ids)

        # ── Gait command ───────────────────────────────────────────────────
        gait_id, vel_cmd = get_schedule_entry(self._episode_t)
        if gait_id != self._current_gait_id:
            self.get_logger().info(
                f"Gait switch: {GAIT_TABLE[self._current_gait_id]['name']} → "
                f"{GAIT_TABLE[gait_id]['name']}  (ep_t={self._episode_t:.2f}s)")
            self.raibert.switch_gait(gait_id)
            self._current_gait_id = gait_id

        vel_cmd_obs = np.zeros(3, dtype=np.float32) if gait_id == 6 else vel_cmd

        # ── Raibert gait planner ───────────────────────────────────────────
        # Uses v_cmd only — no measured velocity needed.
        gait_obs = self.raibert.step(v_cmd=vel_cmd, q_wxyz=q_wxyz)

        # ── Debug prints ───────────────────────────────────────────────────
        if self._step_count == 0:
            print("\n=== OBS AT ACTIVATION (step 0) — 53-dim ===")
            labels = [
                ("proj_grav",      proj_grav),
                ("joint_pos",      joint_pos),
                ("ang_vel_b",      ang_vel_b),
                ("joint_vel",      joint_vel),
                ("vel_cmd_obs",    vel_cmd_obs),
                ("foot_contact",   foot_contact),
                ("desFeetContact", gait_obs["desFeetContact"]),
                ("refFootZ",       gait_obs["refFootZ"]),
                ("refFootX",       gait_obs["refFootX"]),
                ("refFootY",       gait_obs["refFootY"]),
            ]
            for name, val in labels:
                print(f"  {name:16s} min={val.min():+.3f} max={val.max():+.3f} "
                      f"vals={np.round(val, 3)}")
            print("===================================\n")

        if 0 <= self._step_count < 20:
            print(
                f"[s{self._step_count:02d}] "
                f"grav=[{proj_grav[0]:+.3f},{proj_grav[1]:+.3f},{proj_grav[2]:+.3f}] "
                f"w=[{ang_vel_b[0]:+.3f},{ang_vel_b[1]:+.3f},{ang_vel_b[2]:+.3f}] "
                f"desC={gait_obs['desFeetContact'].astype(int).tolist()} "
                f"refZ=[{' '.join(f'{v:+.3f}' for v in gait_obs['refFootZ'])}]"
            )

        if self._step_count % 50 == 0:
            print(f"[step {self._step_count}] gait={GAIT_TABLE[gait_id]['name']} "
                  f"vel_cmd={vel_cmd} contact={foot_contact.tolist()}")

        if self._step_count % 100 == 0:
            print(f"  joint_pos (mujoco)  : {joint_pos_mujoco.round(3).tolist()}")
            print(f"  joint_pos (internal): {joint_pos.round(3).tolist()}")
            print(f"  joint_vel (internal): {joint_vel.round(3).tolist()}")

        # ── Assemble 53-dim observation ────────────────────────────────────
        obs = np.concatenate([
            proj_grav,                       # 3   — IMU
            joint_pos,                       # 12  — encoders (Isaac internal order)
            ang_vel_b,                       # 3   — IMU
            joint_vel,                       # 12  — encoders (Isaac internal order)
            vel_cmd_obs,                     # 3   — operator command
            foot_contact,                    # 4   — foot contact sensors
            gait_obs["desFeetContact"],      # 4   — Raibert schedule
            gait_obs["refFootZ"],            # 4   — Raibert schedule
            gait_obs["refFootX"],            # 4   — Raibert schedule
            gait_obs["refFootY"],            # 4   — Raibert schedule
        ], dtype=np.float32)                 # total = 53

        obs_clipped = np.clip(obs, -OBS_CLIP, OBS_CLIP)

        with torch.no_grad():
            action_raw = self.policy(
                torch.from_numpy(obs_clipped).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)

        if 0 <= self._step_count < 20:
            print(f"       act=[{' '.join(f'{v:+.3f}' for v in action_raw)}]")

        # ── Compute target positions and apply to MuJoCo ───────────────────
        target_internal = DEFAULT_JOINT_POS_INTERNAL + action_raw * ACTION_SCALE
        target_mujoco   = target_internal[INTERNAL_TO_MUJOCO]

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
            joint_pos    = self.d.qpos[7:19].copy().astype(np.float32)
            joint_vel    = self.d.qvel[6:18].copy().astype(np.float32)
            quat         = self.d.qpos[3:7].copy().astype(np.float32)
            gyro         = self.d.sensordata[40:43].copy().astype(np.float32)
            qpos_full    = self.d.qpos[:19].copy()
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

        low_state_msg.imu_state.quaternion = quat
        low_state_msg.imu_state.gyroscope  = gyro
        self.low_state_puber.publish(low_state_msg)

        pos_msg = Float32MultiArray()
        pos_msg.data = qpos_full.tolist()
        self.pos_pub.publish(pos_msg)

        force_msg = Float32MultiArray()
        force_msg.data = np.concatenate([f1, f2, f3, f4]).tolist()
        self.force_pub.publish(force_msg)

        contact_msg = Float32MultiArray()
        contact_msg.data = foot_contact.tolist()
        self.foot_contact_pub.publish(contact_msg)

        self.debug_count += 1
        if self.debug_count % 200 == 0:
            base_height = float(self.d.qpos[2])
            lin_vel_b   = self.d.sensordata[52:55].copy().astype(np.float32)
            print(f"height={base_height:.3f}  "
                  f"lin_vel={np.round(lin_vel_b, 3).tolist()}  "
                  f"contact={foot_contact.tolist()}")

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