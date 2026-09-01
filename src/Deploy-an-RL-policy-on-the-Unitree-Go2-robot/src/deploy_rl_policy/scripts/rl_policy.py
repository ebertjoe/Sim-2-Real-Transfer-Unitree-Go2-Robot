#!/usr/bin/env python3
"""
rl_policy.py — RL locomotion policy node (53-dim obs, Raibert gait planner).

Identical code path in simulation and on the real robot. The only difference
is who publishes /lowstate:

    sim   : mujoco_simulator.py   (is_simulation:=true)
    real  : the Go2 firmware over DDS (rt/lowstate)  (is_simulation:=false)

Output: /rl/target_pos — 12 target joint positions in MUJOCO/SDK order.
low_level_ctrl consumes these and applies PD with policy_kp / policy_kd,
so set those to match the original inline values:

    -p policy_kp:=25.0 -p policy_kd:=0.5

Obs layout (53):
    projected_gravity   3   from imu_state.quaternion
    joint_pos          12   motor_state[i].q     (reordered, see below)
    ang_vel             3   imu_state.gyroscope
    joint_vel          12   motor_state[i].dq    (reordered, see below)
    vel_cmd             3   gait schedule
    foot_contact        4   foot_force > threshold
    desFeetContact      4   Raibert
    refFootZ            4   Raibert
    refFootX            4   Raibert
    refFootY            4   Raibert
"""

import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowState

project_root = Path(__file__).parents[4]

# ── Policy constants ─────────────────────────────────────────────────────────
POLICY_PATH  = str(project_root / "resources" / "go2" / "2026-08-17_20-51-38.pt")
NUM_JOINTS   = 12
ACTION_SCALE = 0.25
STEP_DT      = 0.010          # 100 Hz
OBS_CLIP     = 100.0
OBS_DIM      = 53

# foot_force -> binary contact. mujoco_simulator publishes 0 / 100, so 20 works
# in sim unchanged. CALIBRATE on the real robot: hang it (expect ~0), stand it,
# read foot_force, set to roughly half the standing value.
FOOT_FORCE_THRESHOLD = 20.0

# ── Joint order ──────────────────────────────────────────────────────────────
# MuJoCo / SDK order : FR(hip,thigh,calf), FL, RR, RL
# Isaac internal     : grouped by joint type, legs FL,FR,RL,RR
#
# NOTE: MUJOCO_TO_INTERNAL is left as identity, matching the original code.
# The inverse of INTERNAL_TO_MUJOCO would be [3,0,9,6,4,1,10,7,5,2,11,8];
# swap it in here to test the alternative — nothing else needs to change.
MUJOCO_TO_INTERNAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
INTERNAL_TO_MUJOCO = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

DEFAULT_JOINT_POS_INTERNAL = np.array(
    [ 0.1, -0.1,  0.1, -0.1,
      0.8,  0.8,  1.0,  1.0,
     -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)
DEFAULT_JOINT_POS_MUJOCO = DEFAULT_JOINT_POS_INTERNAL[INTERNAL_TO_MUJOCO]

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

# Schedule used in simulation — unchanged from the original file.
GAIT_SCHEDULE_SIM = [
    (500,   6, [0.0, 0.0, 0.0]),
    (2000,  0, [0.8, 0.0, 0.0]),
    (500,   1, [0.8, 0.0, 0.0]),
    (500,   2, [0.8, 0.0, 0.0]),
    (500,   3, [0.8, 0.0, 0.0]),
    (500,   4, [0.8, 0.0, 0.0]),
    (500,   5, [0.8, 0.0, 0.0]),
    (500,   7, [0.8, 0.0, 0.0]),
    (99999, 7, [0.8, 0.0, 0.0]),
]

# Conservative schedule for the real robot: stand -> trot in place -> slow trot.
GAIT_SCHEDULE_REAL = [
    (1000,  6, [0.0, 0.0, 0.0]),
    (1000,  1, [0.0, 0.0, 0.0]),
    (99999, 1, [0.3, 0.0, 0.0]),
]

# Activation gates. No ground-truth base height off-sim, so gate on attitude.
MAX_TILT_FOR_ACTIVATION     = 0.15   # |proj_grav_xy|
MAX_ANGVEL_FOR_ACTIVATION   = 0.3    # rad/s
MAX_POSE_ERR_FOR_ACTIVATION = 0.30   # rad
STATE_TIMEOUT               = 0.05   # s

# Safety clamp on targets (URDF limits minus margin), MuJoCo/SDK order.
_M = 0.05
JOINT_LIMIT_LO = np.array([
    -1.0472 + _M, -1.5708 + _M, -2.7227 + _M,   # FR
    -1.0472 + _M, -1.5708 + _M, -2.7227 + _M,   # FL
    -1.0472 + _M, -0.5236 + _M, -2.7227 + _M,   # RR
    -1.0472 + _M, -0.5236 + _M, -2.7227 + _M,   # RL
], dtype=np.float32)
JOINT_LIMIT_HI = np.array([
     1.0472 - _M,  3.4907 - _M, -0.83776 - _M,
     1.0472 - _M,  3.4907 - _M, -0.83776 - _M,
     1.0472 - _M,  4.5379 - _M, -0.83776 - _M,
     1.0472 - _M,  4.5379 - _M, -0.83776 - _M,
], dtype=np.float32)


# ── Gait utilities ───────────────────────────────────────────────────────────

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


class RaibertGait:
    """Raibert planner. Uses v_cmd for foot placement (lin_vel is not in obs)."""
    STEP_HEIGHT = 0.10
    BLEND_ALPHA = 0.1

    def __init__(self, gait_id: int = 6):
        self._apply(gait_id)
        self._period_blended     = self._period
        self._znom_blended       = self._z_nom
        self._old_period_blended = self._period_blended
        self._gait_just_switched = False
        self._t_exec             = 0.0
        self._phase_compensation = 0.0
        self._p_ref_B            = HIP_POS_B.copy().astype(np.float64)
        self._p_ref_B[:, 2]      = self._z_nom
        self._prev_c             = np.ones(4, dtype=np.float64)

    def _apply(self, gait_id):
        g = GAIT_TABLE[gait_id]
        self._gait_id   = gait_id
        self._period    = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset    = np.array(g["offset"], dtype=np.float64)
        self._k         = float(g["k"])
        self._z_nom     = float(g["z_nom"])
        self._x_lim     = float(g["x_lim"])
        self._y_lim     = float(g["y_lim"])

    def switch_gait(self, new_gait_id: int):
        self._old_period_blended = self._period_blended
        self._apply(new_gait_id)
        self._gait_just_switched = True

    def step(self, v_cmd, q_wxyz):
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

        dx = 0.5 * Tst * float(v_cmd[0])
        dy = 0.5 * Tst * float(v_cmd[1])

        new_p = HIP_POS_B.copy().astype(np.float64)
        new_p[:, 0] = np.clip(new_p[:, 0] + dx,
                              HIP_POS_B[:, 0] - self._x_lim,
                              HIP_POS_B[:, 0] + self._x_lim)
        new_p[:, 1] = np.clip(new_p[:, 1] + dy,
                              HIP_POS_B[:, 1] - self._y_lim,
                              HIP_POS_B[:, 1] + self._y_lim)
        new_p[:, 2] = z_nom

        liftoff = (self._prev_c > 0.5) & (c_ref < 0.5)
        for leg in range(4):
            if liftoff[leg]:
                self._p_ref_B[leg] = new_p[leg]

        swing_mask = (c_ref < 0.5).astype(np.float64)
        x_sw = np.clip((leg_phase - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
        z_sw = 0.5 * self.STEP_HEIGHT * (1.0 - np.cos(2.0 * math.pi * x_sw)) * swing_mask

        pf = self._p_ref_B.copy()
        pf[:, 2] = z_nom + z_sw
        pw = (quat_to_rotmat(q_wxyz) @ pf.T).T

        self._prev_c  = c_ref.copy()
        self._t_exec += STEP_DT

        return {
            "desFeetContact": c_ref.astype(np.float32),
            "refFootZ":       pw[:, 2].astype(np.float32),
            "refFootX":       pw[:, 0].astype(np.float32),
            "refFootY":       pw[:, 1].astype(np.float32),
        }


def get_schedule_entry(schedule, episode_t):
    elapsed = 0.0
    for i, (duration, gait_id, vel_cmd) in enumerate(schedule):
        elapsed += duration * STEP_DT
        if episode_t < elapsed or i == len(schedule) - 1:
            return gait_id, np.array(vel_cmd, dtype=np.float32)
    last = schedule[-1]
    return last[1], np.array(last[2], dtype=np.float32)


# ── Node ─────────────────────────────────────────────────────────────────────

class RLPolicy(Node):
    def __init__(self):
        super().__init__("rl_policy")

        self.declare_parameter("is_simulation", True)
        # Empty default = derive from is_simulation, mirroring low_level_ctrl.cpp.
        # Set it explicitly to override.
        self.declare_parameter("lowstate_topic", "")
        self.declare_parameter("policy_path", POLICY_PATH)

        self.is_sim    = bool(self.get_parameter("is_simulation").value)
        lowstate_topic = self.get_parameter("lowstate_topic").value
        if not lowstate_topic:
            lowstate_topic = "/mujoco/lowstate" if self.is_sim else "/lowstate"
        policy_path    = self.get_parameter("policy_path").value

        self.schedule = GAIT_SCHEDULE_SIM if self.is_sim else GAIT_SCHEDULE_REAL

        # ── Robot state ──────────────────────────────────────────────────
        self._lock           = threading.Lock()
        self._q              = np.zeros(12, dtype=np.float32)   # MuJoCo/SDK order
        self._dq             = np.zeros(12, dtype=np.float32)
        self._quat_wxyz      = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._gyro           = np.zeros(3, dtype=np.float32)
        self._foot_force     = np.zeros(4, dtype=np.float32)
        self._last_state_t   = 0.0
        self._received_state = False

        self.create_subscription(LowState, lowstate_topic, self._lowstate_cb, 10)
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)
        self.target_pub = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)

        # ── Policy ───────────────────────────────────────────────────────
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.get_logger().info(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        with torch.no_grad():                      # warm up the JIT
            for _ in range(3):
                self.policy(torch.zeros(1, OBS_DIM))
        self.get_logger().info("Policy ready.")

        self._policy_active   = False
        self._step_count      = 0
        self._episode_t       = 0.0
        self._current_gait_id = self.schedule[0][1]
        self.raibert          = RaibertGait(gait_id=self.schedule[0][1])
        self._overruns        = 0

        self.create_timer(STEP_DT, self._policy_tick)

        self.get_logger().info(
            f"rl_policy ready — {'SIM' if self.is_sim else 'REAL'}  "
            f"lowstate<-{lowstate_topic}  targets->/rl/target_pos  "
            f"(hold LB+RB to engage)")

    # ── Callbacks ────────────────────────────────────────────────────────
    def _lowstate_cb(self, msg: LowState):
        with self._lock:
            for i in range(12):
                self._q[i]  = msg.motor_state[i].q
                self._dq[i] = msg.motor_state[i].dq
            self._quat_wxyz  = np.array(msg.imu_state.quaternion, dtype=np.float32)
            self._gyro       = np.array(msg.imu_state.gyroscope,  dtype=np.float32)
            self._foot_force = np.array(msg.foot_force, dtype=np.float32)
            self._last_state_t   = time.monotonic()
            self._received_state = True

    def _joy_cb(self, msg: Joy):
        lb = len(msg.buttons) > 4 and msg.buttons[4]
        rb = len(msg.buttons) > 5 and msg.buttons[5]
        want = bool(lb and rb)

        if want and not self._policy_active:
            ok, why = self._activation_check()
            if not ok:
                self.get_logger().warn(f"Activation REJECTED: {why}")
                return
            self._policy_active   = True
            self._step_count      = 0
            self._episode_t       = 0.0
            self._current_gait_id = self.schedule[0][1]
            self.raibert          = RaibertGait(gait_id=self.schedule[0][1])
            self.raibert._t_exec  = self.raibert._period * 0.5
            self.get_logger().info(
                f"Policy ACTIVATED — {GAIT_TABLE[self._current_gait_id]['name']} gait")

        elif not want and self._policy_active:
            self._policy_active = False
            self.get_logger().info("Policy DEACTIVATED.")

    def _activation_check(self):
        with self._lock:
            if not self._received_state:
                return False, "no lowstate received yet"
            if time.monotonic() - self._last_state_t > STATE_TIMEOUT:
                return False, "lowstate is stale"
            proj_grav = quat_rotate_inverse(self._quat_wxyz, GRAVITY_W)
            tilt      = float(np.linalg.norm(proj_grav[:2]))
            angvel    = float(np.linalg.norm(self._gyro))
            pose_err  = float(np.max(np.abs(self._q - DEFAULT_JOINT_POS_MUJOCO)))
        if tilt > MAX_TILT_FOR_ACTIVATION:
            return False, f"tilt={tilt:.3f} (not upright)"
        if angvel > MAX_ANGVEL_FOR_ACTIVATION:
            return False, f"|ang_vel|={angvel:.3f}"
        if pose_err > MAX_POSE_ERR_FOR_ACTIVATION:
            return False, f"pose_err={pose_err:.3f} rad (not near standing pose)"
        return True, ""

    # ── 100 Hz policy loop ───────────────────────────────────────────────
    def _policy_tick(self):
        if not self._policy_active:
            return
        t0 = time.monotonic()

        with self._lock:
            if time.monotonic() - self._last_state_t > STATE_TIMEOUT:
                self._policy_active = False
                self.get_logger().error(
                    "lowstate STALE — deactivating. low_level_ctrl should damp.")
                return
            q_mj       = self._q.copy()
            dq_mj      = self._dq.copy()
            q_wxyz     = self._quat_wxyz.copy()
            gyro       = self._gyro.copy()
            foot_force = self._foot_force.copy()

        # ── Observation ──────────────────────────────────────────────────
        joint_pos = q_mj[MUJOCO_TO_INTERNAL]
        joint_vel = dq_mj[MUJOCO_TO_INTERNAL]

        proj_grav    = quat_rotate_inverse(q_wxyz, GRAVITY_W)
        foot_contact = (foot_force > FOOT_FORCE_THRESHOLD).astype(np.float32)

        gait_id, vel_cmd = get_schedule_entry(self.schedule, self._episode_t)
        if gait_id != self._current_gait_id:
            self.get_logger().info(
                f"Gait switch: {GAIT_TABLE[self._current_gait_id]['name']} -> "
                f"{GAIT_TABLE[gait_id]['name']}  (t={self._episode_t:.2f}s)")
            self.raibert.switch_gait(gait_id)
            self._current_gait_id = gait_id

        vel_cmd_obs = np.zeros(3, dtype=np.float32) if gait_id == 6 else vel_cmd
        gait_obs    = self.raibert.step(v_cmd=vel_cmd, q_wxyz=q_wxyz)

        obs = np.concatenate([
            proj_grav,                    # 3
            joint_pos,                    # 12
            gyro,                         # 3
            joint_vel,                    # 12
            vel_cmd_obs,                  # 3
            foot_contact,                 # 4
            gait_obs["desFeetContact"],   # 4
            gait_obs["refFootZ"],         # 4
            gait_obs["refFootX"],         # 4
            gait_obs["refFootY"],         # 4
        ], dtype=np.float32)

        assert obs.shape[0] == OBS_DIM, f"obs is {obs.shape[0]}, expected {OBS_DIM}"

        if self._step_count == 0:
            self._dump_obs(proj_grav, joint_pos, gyro, joint_vel,
                           vel_cmd_obs, foot_contact, gait_obs)

        with torch.no_grad():
            action = self.policy(
                torch.from_numpy(np.clip(obs, -OBS_CLIP, OBS_CLIP)).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)

        # ── Targets ──────────────────────────────────────────────────────
        target_internal = DEFAULT_JOINT_POS_INTERNAL + action * ACTION_SCALE
        target_mujoco   = target_internal[INTERNAL_TO_MUJOCO]
        target_mujoco   = np.clip(target_mujoco, JOINT_LIMIT_LO, JOINT_LIMIT_HI)

        msg = Float32MultiArray()
        msg.data = target_mujoco.tolist()
        self.target_pub.publish(msg)

        if self._step_count % 50 == 0:
            self.get_logger().info(
                f"[step {self._step_count}] gait={GAIT_TABLE[gait_id]['name']} "
                f"vel_cmd={vel_cmd.tolist()} contact={foot_contact.tolist()}")

        self._episode_t  += STEP_DT
        self._step_count += 1

        dt = time.monotonic() - t0
        if dt > STEP_DT:
            self._overruns += 1
            self.get_logger().warn(
                f"Policy tick overran: {dt*1000:.1f} ms (total {self._overruns})")

    def _dump_obs(self, proj_grav, joint_pos, gyro, joint_vel,
                  vel_cmd_obs, foot_contact, gait_obs):
        print(f"\n=== OBS AT ACTIVATION (step 0) — {OBS_DIM}-dim "
              f"[{'SIM' if self.is_sim else 'REAL'}] ===")
        for name, val in [
            ("proj_grav",      proj_grav),
            ("joint_pos",      joint_pos),
            ("ang_vel_b",      gyro),
            ("joint_vel",      joint_vel),
            ("vel_cmd_obs",    vel_cmd_obs),
            ("foot_contact",   foot_contact),
            ("desFeetContact", gait_obs["desFeetContact"]),
            ("refFootZ",       gait_obs["refFootZ"]),
            ("refFootX",       gait_obs["refFootX"]),
            ("refFootY",       gait_obs["refFootY"]),
        ]:
            print(f"  {name:16s} min={val.min():+.3f} max={val.max():+.3f} "
                  f"vals={np.round(val, 3)}")
        print("=" * 55 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = RLPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()