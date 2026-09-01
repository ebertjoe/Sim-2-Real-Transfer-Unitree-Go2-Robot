"""Single-episode MuJoCo runner for the robustness sweep.

Reproduces the deployed controller exactly: the 53-dim observation of
``src/deploy_rl_policy/scripts/mujoco_simulator.py`` (no lin_vel, no joint
torques, no base height), the same Raibert planner driven by the commanded
velocity, the same joint reordering, PD gains and torque limits.

Per episode it records everything section 4.1 asks for: survival and
termination reason, the mean tracking errors and achieved values in vx / vy /
wz, mean and standard deviation of the base height, mean roll and pitch derived
from projected gravity, the per-foot contact fraction, the gait contact
accuracy against the Raibert schedule, and the planar drift at the end of the
episode.

    runner = EpisodeRunner(xml, policy_path, ProtocolCfg())
    runner.reset(gait_id=1, vel_cmd=[0.6, 0.0, 0.0], seed=0)
    while not runner.done:
        runner.step()
    metrics = runner.get_episode_metrics()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import grid as G

# ── Controller constants (must match the deployed simulator) ─────────────────

NUM_JOINTS = 12
OBS_DIM = 53            # the deployed observation: see _build_obs below
ACTION_SCALE = 0.25
STEP_DT = G.STEP_DT     # policy dt, 100 Hz
PHYSICS_DT = 0.002      # physics dt, 500 Hz
DECIMATION = 5          # physics steps per policy step
OBS_CLIP = 100.0
TORQUE_LIMIT = 23.5
KP, KD = 25.0, 0.5

MUJOCO_TO_INTERNAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
INTERNAL_TO_MUJOCO = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

DEFAULT_JOINT_POS_INTERNAL = np.array(
    [ 0.1, -0.1,  0.1, -0.1,    # hip:   FL, FR, RL, RR
      0.8,  0.8,  1.0,  1.0,    # thigh: FL, FR, RL, RR
     -1.5, -1.5, -1.5, -1.5],   # calf:  FL, FR, RL, RR
    dtype=np.float32,
)

DEFAULT_JOINT_POS_MUJOCO = np.array(
    [+0.1, 0.8, -1.5,   # FR
     -0.1, 0.8, -1.5,   # FL
     +0.1, 1.0, -1.5,   # RR
     -0.1, 1.0, -1.5],  # RL
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

CALF_BODIES = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
TRUNK_BODY = "base_link"

# Reset: PD stand-up followed by a policy warm-up in the stand gait, which is
# how the robot reaches the policy's own steady standing pose in the live
# simulator before the operator selects a gait.
STANDUP_PHYSICS_STEPS = 2000    # 4 s at 500 Hz
STANDUP_KP, STANDUP_KD = 25.0, 0.5
STANDUP_POSE_NOISE = 0.01       # rad, per joint
WARMUP_POLICY_STEPS = 300       # 3 s in the stand gait

# Fall detection: 0.15 m with a 10-step grace window. Chosen from pronk data,
# where the normal bounce dips to 0.189 m while a genuine collapse reaches
# 0.05-0.10 m; the grace window keeps flight phases from tripping it.
FALL_HEIGHT = 0.15
FALL_HEIGHT_GRACE = 10
ORIENTATION_LIMIT = -0.5        # projected gravity z above this = tilted past 60 deg


# ── Protocol configuration (mirrors the Isaac --protocol flags) ──────────────

@dataclass
class ProtocolCfg:
    episode_steps: int = G.EPISODE_STEPS
    settle_steps: int | None = None          # None -> per-gait default

    # start-up randomisation of the robot, on by default like the Isaac sweep
    domain_rand: bool = True
    mass_range: tuple[float, float] = (-1.0, 3.0)      # kg added to the trunk
    friction_range: tuple[float, float] = (0.6, 1.2)   # sliding friction of the floor

    # interval push disturbance, off by default
    push_robot: bool = False
    push_interval_s: float = 5.0
    push_vel: float = 0.5                    # m/s, uniform in [-v, v] on x and y

    # observation corruption, off by default
    obs_noise: bool = False
    noise_proj_gravity: float = 0.05
    noise_joint_pos: float = 0.01
    noise_ang_vel: float = 0.2
    noise_joint_vel: float = 1.5

    deterministic_reset: bool = False        # zero the reset pose/velocity noise

    terminate_on_base_contact: bool = True
    terminate_on_fall_height: bool = True
    terminate_on_orientation: bool = False


def check_policy(policy_path: str | Path, obs_dim: int = OBS_DIM):
    """Fail early if the export does not take the deployed observation.

    ``resources/go2`` also holds older 69-dim exports (with lin_vel, joint
    torques and base height); feeding those the 53-dim vector fails deep inside
    TorchScript, so check it up front the way the Isaac sweep checks the gait
    table of the task.
    """
    policy = torch.jit.load(str(policy_path), map_location="cpu")
    policy.eval()
    try:
        with torch.no_grad():
            action = policy(torch.zeros(1, obs_dim))
    except Exception as exc:                      # noqa: BLE001 - re-raised as a message
        raise SystemExit(
            f"policy {policy_path} does not accept the {obs_dim}-dim observation "
            f"of the deployed controller ({type(exc).__name__}). It is most likely "
            f"an older 69-dim export.") from exc
    if action.shape[-1] != NUM_JOINTS:
        raise SystemExit(
            f"policy {policy_path} returns {action.shape[-1]} actions, expected {NUM_JOINTS}")


# ── Geometry helpers ─────────────────────────────────────────────────────────

def quat_rotate_inverse(q_wxyz, v):
    """Rotate ``v`` from world frame into body frame."""
    w = float(q_wxyz[0])
    q_xyz = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3]], dtype=np.float64)
    v64 = np.array(v, dtype=np.float64)
    t = 2.0 * np.cross(q_xyz, v64)
    return (v64 - w * t + np.cross(q_xyz, t)).astype(np.float32)


def quat_to_rotmat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float32)


def rpy_from_projected_gravity(g_b) -> tuple[float, float]:
    """Roll and pitch (rad) from the projected gravity vector.

    ``g_b`` is gravity expressed in the body frame, i.e. (0, 0, -1) when level,
    which is exactly the first three observation entries the policy sees.
    """
    gx, gy, gz = (float(g_b[0]), float(g_b[1]), float(g_b[2]))
    roll = math.atan2(-gy, -gz)
    pitch = math.atan2(gx, math.sqrt(gy * gy + gz * gz))
    return roll, pitch


# ── Raibert planner (identical to the deployed 53-dim simulator) ─────────────

class RaibertGait:
    """Contact schedule and foot references; driven by the commanded velocity.

    ``lin_vel`` is not part of the 53-dim observation, so the foot placement
    term collapses to ``dx = 0.5 * Tst * v_cmd`` exactly as in
    ``mujoco_simulator.py``.
    """

    STEP_HEIGHT = 0.10
    BLEND_ALPHA = 0.1

    def __init__(self, gait_id: int = G.STAND_GAIT_ID):
        self._apply(gait_id)
        self._period_blended = self._period
        self._znom_blended = self._z_nom
        self._old_period_blended = self._period
        self._gait_just_switched = False
        self._t_exec = 0.0
        self._phase_compensation = 0.0
        self._p_ref_B = HIP_POS_B.copy().astype(np.float64)
        self._p_ref_B[:, 2] = self._z_nom
        self._prev_c = np.ones(4, dtype=np.float64)

    def _apply(self, gait_id: int):
        g = G.GAIT_TABLE[gait_id]
        self._gait_id = gait_id
        self._period = float(g["period"])
        self._threshold = float(g["threshold"])
        self._offset = np.array(g["offset"], dtype=np.float64)
        self._k = float(g["k"])
        self._z_nom = float(g["z_nom"])
        self._x_lim = float(g["x_lim"])
        self._y_lim = float(g["y_lim"])

    def step(self, v_cmd, q_wxyz) -> dict:
        a = self.BLEND_ALPHA
        self._period_blended = a * self._period + (1.0 - a) * self._period_blended
        self._znom_blended = a * self._z_nom + (1.0 - a) * self._znom_blended

        if self._gait_just_switched:
            t = self._t_exec
            self._phase_compensation = (
                t - (t - self._phase_compensation)
                * (self._period_blended / max(self._old_period_blended, 1e-6))
            )
            self._gait_just_switched = False

        T = self._period_blended
        thr = self._threshold
        z_nom = self._znom_blended
        Tst = thr * T

        global_phase = ((self._t_exec - self._phase_compensation) % T) / T
        leg_phase = (global_phase + self._offset) % 1.0
        c_ref = (leg_phase < thr).astype(np.float64)

        dx = 0.5 * Tst * float(v_cmd[0])
        dy = 0.5 * Tst * float(v_cmd[1])

        new_p = HIP_POS_B.copy().astype(np.float64)
        new_p[:, 0] += dx
        new_p[:, 1] += dy
        new_p[:, 2] = z_nom
        new_p[:, 0] = np.clip(new_p[:, 0], HIP_POS_B[:, 0] - self._x_lim,
                              HIP_POS_B[:, 0] + self._x_lim)
        new_p[:, 1] = np.clip(new_p[:, 1], HIP_POS_B[:, 1] - self._y_lim,
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
        pw = (quat_to_rotmat(q_wxyz) @ pf.T).T

        self._prev_c = c_ref.copy()
        self._t_exec += STEP_DT

        return {
            "desFeetContact": c_ref.astype(np.float32),
            "refFootZ": pw[:, 2].astype(np.float32),
            "refFootX": pw[:, 0].astype(np.float32),
            "refFootY": pw[:, 1].astype(np.float32),
        }


# ── Episode runner ───────────────────────────────────────────────────────────

class EpisodeRunner:
    """Headless MuJoCo + policy runner; one instance runs many episodes."""

    def __init__(self, xml_path: str | Path, policy_path: str | Path,
                 cfg: ProtocolCfg | None = None):
        self.cfg = cfg or ProtocolCfg()

        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = PHYSICS_DT

        self._calf_body_ids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in CALF_BODIES
        ]
        self._trunk_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY)
        self._floor_geom_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # nominal dynamics, restored before every episode's randomisation draw
        self._nominal_trunk_mass = float(self.m.body_mass[self._trunk_body_id])
        self._nominal_friction = self.m.geom_friction.copy()

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.policy = torch.jit.load(str(policy_path), map_location="cpu")
        self.policy.eval()

        # Optional ``sync_hook(sim_dt)``, called while the reset is simulated so
        # a live viewer can draw the stand-up as well as the episode itself.
        self.sync_hook = None

        self.done = True
        self.termination_reason = "not_started"
        self._gait_id = G.STAND_GAIT_ID
        self._vel_cmd = np.zeros(3, dtype=np.float32)
        self._tau = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._step_count = 0

    # ── policy plumbing ──────────────────────────────────────────────────────

    def _build_obs(self, proj_grav, joint_pos, ang_vel, joint_vel,
                   vel_cmd_obs, foot_contact, gait_obs) -> np.ndarray:
        """The 53-dim observation of the deployed controller."""
        return np.concatenate([
            proj_grav,                   # 3   IMU
            joint_pos,                   # 12  encoders, Isaac internal order
            ang_vel,                     # 3   IMU
            joint_vel,                   # 12  encoders, Isaac internal order
            vel_cmd_obs,                 # 3   operator command
            foot_contact,                # 4   contact sensors
            gait_obs["desFeetContact"],  # 4   Raibert schedule
            gait_obs["refFootZ"],        # 4
            gait_obs["refFootX"],        # 4
            gait_obs["refFootY"],        # 4
        ], dtype=np.float32)

    def _corrupt(self, obs: np.ndarray) -> np.ndarray:
        """Uniform observation noise on the sensed terms only."""
        c = self.cfg
        noisy = obs.copy()
        rng = self._rng
        noisy[0:3] += rng.uniform(-c.noise_proj_gravity, c.noise_proj_gravity, 3)
        noisy[3:15] += rng.uniform(-c.noise_joint_pos, c.noise_joint_pos, 12)
        noisy[15:18] += rng.uniform(-c.noise_ang_vel, c.noise_ang_vel, 3)
        noisy[18:30] += rng.uniform(-c.noise_joint_vel, c.noise_joint_vel, 12)
        return noisy.astype(np.float32)

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        if self.cfg.obs_noise:
            obs = self._corrupt(obs)
        obs = np.clip(obs, -OBS_CLIP, OBS_CLIP)
        with torch.no_grad():
            return self.policy(
                torch.from_numpy(obs).unsqueeze(0)
            ).squeeze(0).numpy().astype(np.float32)

    def _pd_torque(self, target_mujoco: np.ndarray) -> np.ndarray:
        q = self.d.qpos[7:19]
        dq = self.d.qvel[6:18]
        return np.clip((target_mujoco - q) * KP - dq * KD,
                       -TORQUE_LIMIT, TORQUE_LIMIT).astype(np.float32)

    def _foot_contact(self) -> np.ndarray:
        contact = np.zeros(4, dtype=np.float32)
        for con in range(int(self.d.ncon)):
            c = self.d.contact[con]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            for i, bid in enumerate(self._calf_body_ids):
                if b1 == bid or b2 == bid:
                    contact[i] = 1.0
        return contact

    def _trunk_contact(self) -> bool:
        for con in range(int(self.d.ncon)):
            c = self.d.contact[con]
            if (self.m.geom_bodyid[c.geom1] == self._trunk_body_id
                    or self.m.geom_bodyid[c.geom2] == self._trunk_body_id):
                return True
        return False

    # ── reset ────────────────────────────────────────────────────────────────

    def _randomise_dynamics(self):
        """Start-up mass / friction randomisation (the Isaac ``domain_rand``)."""
        self.m.body_mass[self._trunk_body_id] = self._nominal_trunk_mass
        self.m.geom_friction[:] = self._nominal_friction
        self._base_mass = self._nominal_trunk_mass
        self._floor_friction = float(self._nominal_friction[max(self._floor_geom_id, 0), 0])

        if not self.cfg.domain_rand:
            return

        added = float(self._rng.uniform(*self.cfg.mass_range))
        self._base_mass = self._nominal_trunk_mass + added
        self.m.body_mass[self._trunk_body_id] = self._base_mass

        friction = float(self._rng.uniform(*self.cfg.friction_range))
        self._floor_friction = friction
        if self._floor_geom_id >= 0:
            self.m.geom_friction[self._floor_geom_id, 0] = friction
        else:                       # no named floor geom: scale every geom
            self.m.geom_friction[:, 0] = friction

    def _stand_up(self):
        """PD stand-up followed by a policy warm-up in the stand gait."""
        mujoco.mj_resetData(self.m, self.d)

        if not self.cfg.deterministic_reset:
            self.d.qpos[7:19] += self._rng.normal(0.0, STANDUP_POSE_NOISE, NUM_JOINTS)
        mujoco.mj_forward(self.m, self.d)

        for step in range(STANDUP_PHYSICS_STEPS):
            q = self.d.qpos[7:19]
            dq = self.d.qvel[6:18]
            self.d.ctrl[:] = np.clip(
                (DEFAULT_JOINT_POS_MUJOCO - q) * STANDUP_KP - dq * STANDUP_KD,
                -TORQUE_LIMIT, TORQUE_LIMIT)
            mujoco.mj_step(self.m, self.d)
            if self.sync_hook is not None and step % DECIMATION == DECIMATION - 1:
                self.sync_hook(STEP_DT)

        stand = RaibertGait(G.STAND_GAIT_ID)
        stand._t_exec = stand._period * 0.5
        zero_cmd = np.zeros(3, dtype=np.float32)
        for _ in range(WARMUP_POLICY_STEPS):
            q_wxyz = self.d.qpos[3:7].astype(np.float32)
            gait_obs = stand.step(v_cmd=zero_cmd, q_wxyz=q_wxyz)
            obs = self._build_obs(
                proj_grav=quat_rotate_inverse(q_wxyz, GRAVITY_W),
                joint_pos=self.d.qpos[7:19].astype(np.float32)[MUJOCO_TO_INTERNAL],
                ang_vel=self.d.sensordata[40:43].astype(np.float32),
                joint_vel=self.d.qvel[6:18].astype(np.float32)[MUJOCO_TO_INTERNAL],
                vel_cmd_obs=zero_cmd,
                foot_contact=self._foot_contact(),
                gait_obs=gait_obs,
            )
            action = self._infer(obs)
            target = (DEFAULT_JOINT_POS_INTERNAL + action * ACTION_SCALE)[INTERNAL_TO_MUJOCO]
            tau = self._pd_torque(target)
            for _ in range(DECIMATION):
                self.d.ctrl[:] = tau
                mujoco.mj_step(self.m, self.d)
            if self.sync_hook is not None:
                self.sync_hook(STEP_DT)

        if self.cfg.deterministic_reset:
            self.d.qvel[:] = 0.0
            mujoco.mj_forward(self.m, self.d)

    def reset(self, gait_id: int, vel_cmd, seed: int = 0):
        self._rng = np.random.default_rng(int(seed))
        self._gait_id = int(gait_id)
        self._vel_cmd = np.array(vel_cmd, dtype=np.float32)
        if self._gait_id == G.STAND_GAIT_ID:      # stand is a zero-command gait
            self._vel_cmd = np.zeros(3, dtype=np.float32)

        self._randomise_dynamics()
        self._stand_up()

        # Raibert starts half a period in, matching the live simulator.
        self._raibert = RaibertGait(self._gait_id)
        self._raibert._t_exec = self._raibert._period * 0.5

        self._settle_steps = (self.cfg.settle_steps if self.cfg.settle_steps is not None
                              else G.settle_steps(self._gait_id))
        self._step_count = 0
        self._low_height_count = 0
        self._n_pushes = 0
        self._next_push_t = self.cfg.push_interval_s
        self._tau = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._start_xy = self.d.qpos[0:2].copy()
        self._settle_xy = self._start_xy.copy()

        self._buf = {k: [] for k in (
            "vx", "vy", "wz", "vx_err", "vy_err", "wz_err",
            "height", "roll", "pitch", "torque_norm", "contact_acc")}
        self._contact_sum = np.zeros(4, dtype=np.float64)
        self._contact_n = 0

        self.done = False
        self.termination_reason = "running"

    # ── step ─────────────────────────────────────────────────────────────────

    def _maybe_push(self):
        if not self.cfg.push_robot:
            return
        t = self._step_count * STEP_DT
        if t < self._next_push_t:
            return
        v = self.cfg.push_vel
        self.d.qvel[0:2] += self._rng.uniform(-v, v, 2)
        self._n_pushes += 1
        self._next_push_t += self.cfg.push_interval_s

    def step(self):
        """One control step: ``DECIMATION`` physics steps, then a policy call."""
        if self.done:
            return

        for _ in range(DECIMATION):
            self.d.ctrl[:] = self._tau
            mujoco.mj_step(self.m, self.d)

        # ── state ────────────────────────────────────────────────────────────
        joint_pos = self.d.qpos[7:19].astype(np.float32)[MUJOCO_TO_INTERNAL]
        joint_vel = self.d.qvel[6:18].astype(np.float32)[MUJOCO_TO_INTERNAL]
        torques = np.clip(self.d.sensordata[24:36].astype(np.float32),
                          -TORQUE_LIMIT, TORQUE_LIMIT)
        base_height = float(self.d.qpos[2])
        q_wxyz = self.d.qpos[3:7].astype(np.float32)
        proj_grav = quat_rotate_inverse(q_wxyz, GRAVITY_W)
        ang_vel_b = self.d.sensordata[40:43].astype(np.float32)
        lin_vel_b = self.d.sensordata[52:55].astype(np.float32)
        foot_contact = self._foot_contact()
        roll, pitch = rpy_from_projected_gravity(proj_grav)

        settled = self._step_count >= self._settle_steps

        # ── termination ──────────────────────────────────────────────────────
        # Only physical failures terminate; tracking error never does. The
        # settle window suppresses the checks while the gait is still starting.
        if not settled:
            self._low_height_count = 0
        elif base_height < FALL_HEIGHT:
            self._low_height_count += 1
        else:
            self._low_height_count = 0

        cfg = self.cfg
        if settled and cfg.terminate_on_base_contact and self._trunk_contact():
            self.done, self.termination_reason = True, "base_contact"
        elif cfg.terminate_on_fall_height and self._low_height_count >= FALL_HEIGHT_GRACE:
            self.done, self.termination_reason = True, "fall_height"
        elif (settled and cfg.terminate_on_orientation
                and float(proj_grav[2]) > ORIENTATION_LIMIT):
            self.done, self.termination_reason = True, "orientation"
        elif self._step_count >= cfg.episode_steps - 1:
            self.done, self.termination_reason = True, "timeout"

        # ── Raibert schedule ─────────────────────────────────────────────────
        gait_obs = self._raibert.step(v_cmd=self._vel_cmd, q_wxyz=q_wxyz)
        des_contact = gait_obs["desFeetContact"]

        # ── metrics (settle window excluded) ─────────────────────────────────
        if settled:
            b = self._buf
            b["vx"].append(float(lin_vel_b[0]))
            b["vy"].append(float(lin_vel_b[1]))
            b["wz"].append(float(ang_vel_b[2]))
            b["vx_err"].append(abs(float(lin_vel_b[0]) - float(self._vel_cmd[0])))
            b["vy_err"].append(abs(float(lin_vel_b[1]) - float(self._vel_cmd[1])))
            b["wz_err"].append(abs(float(ang_vel_b[2]) - float(self._vel_cmd[2])))
            b["height"].append(base_height)
            b["roll"].append(roll)
            b["pitch"].append(pitch)
            b["torque_norm"].append(float(np.linalg.norm(torques)))
            # gait contact accuracy: a control step counts only if all four feet
            # match the state scheduled by the planner
            b["contact_acc"].append(
                float(np.all((foot_contact > 0.5) == (des_contact > 0.5))))
            self._contact_sum += (foot_contact > 0.5)
            self._contact_n += 1
            if self._step_count == self._settle_steps:
                self._settle_xy = self.d.qpos[0:2].copy()

        # ── act ──────────────────────────────────────────────────────────────
        vel_cmd_obs = (np.zeros(3, dtype=np.float32)
                       if self._gait_id == G.STAND_GAIT_ID else self._vel_cmd)
        obs = self._build_obs(proj_grav, joint_pos, ang_vel_b, joint_vel,
                              vel_cmd_obs, foot_contact, gait_obs)
        action = self._infer(obs)
        target = (DEFAULT_JOINT_POS_INTERNAL + action * ACTION_SCALE)[INTERNAL_TO_MUJOCO]
        self._tau = self._pd_torque(target)

        self._step_count += 1
        self._maybe_push()

    # ── metrics ──────────────────────────────────────────────────────────────

    def get_episode_metrics(self) -> dict:
        def mean(key):
            buf = self._buf[key]
            return float(np.mean(buf)) if buf else float("nan")

        def std(key):
            buf = self._buf[key]
            return float(np.std(buf)) if buf else float("nan")

        contact_frac = (self._contact_sum / self._contact_n if self._contact_n
                        else np.full(4, np.nan))
        # Drift is reported from the start of the episode; the settle-relative
        # value is kept as well so it can be read on the same footing as the
        # other metrics, which all exclude the settle window.
        drift = self.d.qpos[0:2] - self._start_xy
        drift_settled = self.d.qpos[0:2] - self._settle_xy
        vel_cmd = self._vel_cmd
        zero_cmd = bool(np.allclose(vel_cmd, 0.0))

        metrics = {
            "gait_id": self._gait_id,
            "gait_name": G.GAIT_NAMES[self._gait_id],
            "vx_cmd": float(vel_cmd[0]),
            "vy_cmd": float(vel_cmd[1]),
            "wz_cmd": float(vel_cmd[2]),
            "zero_cmd": zero_cmd,

            "survived": self.termination_reason == "timeout",
            "termination_reason": self.termination_reason,
            "survival_steps": self._step_count,
            "survival_time_s": self._step_count * STEP_DT,

            "mean_vx_error": mean("vx_err"),
            "mean_vy_error": mean("vy_err"),
            "mean_wz_error": mean("wz_err"),
            "mean_vx": mean("vx"),
            "mean_vy": mean("vy"),
            "mean_wz": mean("wz"),

            "mean_base_height": mean("height"),
            "std_base_height": std("height"),

            "mean_roll": mean("roll"),
            "mean_pitch": mean("pitch"),
            "mean_abs_roll": float(np.mean(np.abs(self._buf["roll"])))
                             if self._buf["roll"] else float("nan"),
            "mean_abs_pitch": float(np.mean(np.abs(self._buf["pitch"])))
                              if self._buf["pitch"] else float("nan"),

            "mean_contact_frac": float(np.mean(contact_frac)),
            "gait_contact_accuracy": mean("contact_acc"),

            "planar_drift_m": float(np.linalg.norm(drift)),
            "drift_x_m": float(drift[0]),
            "drift_y_m": float(drift[1]),
            "planar_drift_settled_m": float(np.linalg.norm(drift_settled)),

            "mean_torque_norm": mean("torque_norm"),

            "base_mass_kg": float(self._base_mass),
            "floor_friction": float(self._floor_friction),
            "n_pushes": int(self._n_pushes),
            "settle_steps": int(self._settle_steps),
        }
        for i, foot in enumerate(G.FOOT_NAMES):
            metrics[f"contact_frac_{foot}"] = float(contact_frac[i])
        return metrics

    def run_episode(self, gait_id: int, vel_cmd, seed: int = 0) -> dict:
        self.reset(gait_id=gait_id, vel_cmd=vel_cmd, seed=seed)
        while not self.done:
            self.step()
        return self.get_episode_metrics()
