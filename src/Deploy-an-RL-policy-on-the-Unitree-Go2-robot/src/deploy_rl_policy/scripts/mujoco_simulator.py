#!/usr/bin/env python3
"""
mujoco_simulator.py — pure MuJoCo physics + sensor server for the Go2.

This node carries NO policy logic. It is the simulation stand-in for the robot
hardware, and nothing more:

    in   : /mujoco/lowcmd   (LowCmd)   — joint targets + PD gains, from low_level_ctrl
    out  : /mujoco/lowstate (LowState) — the same fields the Go2 firmware publishes

The control chain is identical in sim and on the real robot:

    rl_policy  --/rl/target_pos-->  low_level_ctrl  --lowcmd-->  robot / this node
         ^                                                            |
         +---------------------- lowstate ----------------------------+

On hardware the last hop is the Go2 firmware over DDS; here it is MuJoCo.
That means the policy path exercised in simulation is the one that is deployed.
"""

import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowCmd, LowState

project_root = Path(__file__).parents[4]

# ── Simulation constants ─────────────────────────────────────────────────────
NUM_JOINTS   = 12
PHYSICS_DT   = 0.002   # physics timestep = 2ms = 500Hz
TORQUE_LIMIT = 23.5    # Nm, per joint

# Reported in LowState.foot_force for a foot in contact. rl_policy thresholds
# this at FOOT_FORCE_THRESHOLD (20.0) to get a binary contact flag.
FOOT_FORCE_CONTACT = 100

# Fallback PD gains, used only until the first /mujoco/lowcmd arrives.
# low_level_ctrl sends its own kp/kd with every command.
DEFAULT_KP = 25.0
DEFAULT_KD = 0.5


def _read_foot_contact(d, m, foot_body_ids):
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

        self.low_state_puber  = self.create_publisher(LowState,          "/mujoco/lowstate",      10)
        self.pos_pub          = self.create_publisher(Float32MultiArray, "/mujoco/pos",           10)
        self.force_pub        = self.create_publisher(Float32MultiArray, "/mujoco/force",         10)
        self.torque_pub       = self.create_publisher(Float32MultiArray, "/mujoco/torque",        10)
        self.foot_contact_pub = self.create_publisher(Float32MultiArray, "/mujoco/foot_contact",  10)

        self.lowcmd_sub = self.create_subscription(
            LowCmd, "/mujoco/lowcmd", self.lowcmd_callback, 10)

        self.xml_path = project_root / "resources" / "go2" / "scene_flat.xml"
        self.foot_body_ids = []
        self.calf_body_ids = []
        self.init_mujoco()

        # ── Low-level control state ────────────────────────────────────────
        self.target_dof_pos = [0.0] * NUM_JOINTS
        self.tau            = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.kps            = np.array([DEFAULT_KP] * NUM_JOINTS, dtype=np.float32)
        self.kds            = np.array([DEFAULT_KD] * NUM_JOINTS, dtype=np.float32)
        self.received_data  = False

        self._mujoco_lock = threading.Lock()
        self.running      = True

        self.timer_sensor = self.create_timer(0.005, self.publish_sensor_data)
        self.timer_tau    = self.create_timer(0.001, self.update_tau)
        self.sim_thread   = threading.Thread(target=self.step_simulation, daemon=True)
        self.sim_thread.start()

        self.debug_count = 0
        self.get_logger().info(
            "MujocoSimulator ready — physics only. "
            "Waiting for /mujoco/lowcmd from low_level_ctrl.")

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
        self.received_data = True
        for i in range(NUM_JOINTS):
            self.target_dof_pos[i] = float(msg.motor_cmd[i].q)
            self.kps[i] = float(msg.motor_cmd[i].kp)
            self.kds[i] = float(msg.motor_cmd[i].kd)

    def update_tau(self):
        if not self.received_data:
            return
        for i in range(NUM_JOINTS):
            q  = self.d.qpos[7 + i]
            dq = self.d.qvel[6 + i]
            self.tau[i] = np.clip(
                self.pd_control(self.target_dof_pos[i], q, self.kps[i], dq, self.kds[i]),
                -TORQUE_LIMIT, TORQUE_LIMIT)

    def step_simulation(self):
        while self.viewer.is_running() and self.running:
            # Hold the reset pose until low_level_ctrl sends the first command.
            # Stepping with tau=0 here would just drop the robot on the floor.
            if not self.received_data:
                time.sleep(0.001)
                continue

            step_start = time.time()

            with self._mujoco_lock:
                self.d.ctrl[:] = self.tau
                mujoco.mj_step(self.m, self.d)

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
        for i in range(NUM_JOINTS):
            low_state_msg.motor_state[i].q  = float(joint_pos[i])
            low_state_msg.motor_state[i].dq = float(joint_vel[i])
            if hasattr(low_state_msg.motor_state[i], "tau_est"):
                low_state_msg.motor_state[i].tau_est = float(self.tau[i])

        low_state_msg.imu_state.quaternion = quat
        low_state_msg.imu_state.gyroscope  = gyro
        # Contact flags in FR/FL/RR/RL order, as int16 like the Go2 firmware.
        low_state_msg.foot_force = [
            int(c * FOOT_FORCE_CONTACT) for c in foot_contact]
        self.low_state_puber.publish(low_state_msg)

        pos_msg = Float32MultiArray()
        pos_msg.data = qpos_full.tolist()
        self.pos_pub.publish(pos_msg)

        force_msg = Float32MultiArray()
        force_msg.data = np.concatenate([f1, f2, f3, f4]).tolist()
        self.force_pub.publish(force_msg)

        torque_msg = Float32MultiArray()
        torque_msg.data = self.tau.tolist()
        self.torque_pub.publish(torque_msg)

        contact_msg = Float32MultiArray()
        contact_msg.data = foot_contact.tolist()
        self.foot_contact_pub.publish(contact_msg)

        self.debug_count += 1
        if self.debug_count % 200 == 0:
            base_height = float(self.d.qpos[2])
            print(f"height={base_height:.3f}  contact={foot_contact.tolist()}")

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
