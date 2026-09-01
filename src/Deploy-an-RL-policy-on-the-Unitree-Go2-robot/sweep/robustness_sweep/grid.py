"""Command grid, gait table and CSV schema of the robustness sweep (thesis Table 4.1).

This module is the single source of truth shared by the MuJoCo sweep and the
Isaac Lab sweep: both backends enumerate the same jobs and write the same CSV
columns, so ``episodes.csv`` from either simulator can be concatenated and
compared row for row.

Grid (Table 4.1)
----------------
    vx  [m/s]    0.0, 0.3, 0.6, 0.9, 1.2
    vy  [m/s]   -0.4, -0.2, 0.0, 0.2, 0.4
    wz  [rad/s] -0.5, -0.25, 0.0, 0.25, 0.5
    gaits        bound, trot, hop, amble, pronk, limp, stand, run
    seeds        5

``stand`` is a special case: it is evaluated at zero command only, so the sweep
is 7 x 125 x 5 + 5 = 4380 episodes rather than the nominal 8 x 125 x 5 = 5000.
"""

from __future__ import annotations

import hashlib
from itertools import product

# ── Table 4.1 ────────────────────────────────────────────────────────────────

VX_CMDS = [0.0, 0.3, 0.6, 0.9, 1.2]
VY_CMDS = [-0.4, -0.2, 0.0, 0.2, 0.4]
WZ_CMDS = [-0.5, -0.25, 0.0, 0.25, 0.5]

N_SEEDS = 5

# ── Gaits (identical to the Raibert table in the deployed simulator) ─────────

GAIT_TABLE = {
    0: {"name": "bound", "period": 0.4, "threshold": 0.4,   "offset": [0.5, 0.5, 0.0,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    1: {"name": "trot",  "period": 0.4, "threshold": 0.5,   "offset": [0.0, 0.5, 0.5,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    2: {"name": "hop",   "period": 0.3, "threshold": 0.5,   "offset": [0.0, 0.0, 0.0,  0.0 ], "k": 0.03, "z_nom": -0.30, "x_lim": 0.10, "y_lim": 0.10},
    3: {"name": "amble", "period": 0.5, "threshold": 0.625, "offset": [0.0, 0.5, 0.25, 0.75], "k": 0.02, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    4: {"name": "pronk", "period": 0.5, "threshold": 0.5,   "offset": [0.0, 0.0, 0.0,  0.0 ], "k": 0.01, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    5: {"name": "limp",  "period": 0.4, "threshold": 0.5,   "offset": [0.5, 0.5, 0.5,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    6: {"name": "stand", "period": 1.0, "threshold": 1.0,   "offset": [0.0, 0.0, 0.0,  0.0 ], "k": 0.01, "z_nom": -0.32, "x_lim": 0.10, "y_lim": 0.10},
    7: {"name": "run",   "period": 0.3, "threshold": 0.4,   "offset": [0.0, 0.5, 0.5,  0.0 ], "k": 0.03, "z_nom": -0.32, "x_lim": 0.12, "y_lim": 0.10},
}

GAIT_NAMES = {gid: cfg["name"] for gid, cfg in GAIT_TABLE.items()}
NAME_TO_GAIT_ID = {name: gid for gid, name in GAIT_NAMES.items()}
ALL_GAIT_IDS = sorted(GAIT_TABLE)

STAND_GAIT_ID = NAME_TO_GAIT_ID["stand"]

# Feet in the order used everywhere in this project (MuJoCo body order).
FOOT_NAMES = ["FR", "FL", "RR", "RL"]

# ── Episode protocol ─────────────────────────────────────────────────────────

STEP_DT = 0.010             # policy dt, 100 Hz
EPISODE_STEPS = 1000        # 10 s per episode

# Steps at the start of an episode that are excluded from the metrics and from
# the termination check: most gaits need ~1 s to establish their rhythm from
# the standing pose, pronk and hop need longer (all-four / bilateral flight).
DEFAULT_SETTLE_STEPS = 100
GAIT_SETTLE_STEPS = {
    NAME_TO_GAIT_ID["hop"]: 150,
    NAME_TO_GAIT_ID["pronk"]: 200,
}


def settle_steps(gait_id: int) -> int:
    return GAIT_SETTLE_STEPS.get(int(gait_id), DEFAULT_SETTLE_STEPS)


# ── Job enumeration ──────────────────────────────────────────────────────────

def episode_seed(gait_id: int, vx: float, vy: float, wz: float, seed: int) -> int:
    """Deterministic per-episode RNG seed.

    Derived from the full job key so that replicate ``seed`` reproduces the same
    episode regardless of the order the sweep runs in, while two different grid
    points never share an initial state.
    """
    key = f"{int(gait_id)}|{vx:.4f}|{vy:.4f}|{wz:.4f}|{int(seed)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def build_jobs(seeds: int = N_SEEDS, base_seed: int = 0,
               gaits: list[int] | None = None) -> list[dict]:
    """Enumerate every episode of the sweep.

    ``stand`` is emitted at zero command only; every other gait gets the full
    vx x vy x wz grid. Order is gait-major and deterministic, so ``--resume``
    picks up exactly where a crashed sweep stopped.
    """
    gait_ids = ALL_GAIT_IDS if gaits is None else [int(g) for g in gaits]
    seed_values = [base_seed + i for i in range(seeds)]

    jobs: list[dict] = []
    for gait_id in gait_ids:
        if gait_id == STAND_GAIT_ID:
            commands = [(0.0, 0.0, 0.0)]
        else:
            commands = list(product(VX_CMDS, VY_CMDS, WZ_CMDS))
        for vx, vy, wz in commands:
            for seed in seed_values:
                jobs.append({
                    "gait_id": gait_id,
                    "gait_name": GAIT_NAMES[gait_id],
                    "vx_cmd": float(vx),
                    "vy_cmd": float(vy),
                    "wz_cmd": float(wz),
                    "seed": int(seed),
                    "episode_seed": episode_seed(gait_id, vx, vy, wz, seed),
                })
    return jobs


def job_key(row: dict) -> tuple:
    """Identity of an episode, used to skip already-recorded rows on resume."""
    return (
        int(row["gait_id"]),
        round(float(row["vx_cmd"]), 4),
        round(float(row["vy_cmd"]), 4),
        round(float(row["wz_cmd"]), 4),
        int(row["seed"]),
    )


# ── CSV schema (shared with the Isaac backend) ───────────────────────────────

CSV_FIELDS = [
    # identity
    "run_id", "sim", "policy", "gait_id", "gait_name",
    "vx_cmd", "vy_cmd", "wz_cmd", "zero_cmd", "seed", "episode_seed",
    # survival and termination
    "survived", "termination_reason", "survival_steps", "survival_time_s",
    # velocity tracking: mean absolute error and achieved value
    "mean_vx_error", "mean_vy_error", "mean_wz_error",
    "mean_vx", "mean_vy", "mean_wz",
    # base height
    "mean_base_height", "std_base_height",
    # orientation from projected gravity
    "mean_roll", "mean_pitch", "mean_abs_roll", "mean_abs_pitch",
    # contacts
    "contact_frac_FR", "contact_frac_FL", "contact_frac_RR", "contact_frac_RL",
    "mean_contact_frac", "gait_contact_accuracy",
    # zero-command stability: planar drift over the whole episode, and over
    # the measured window only (settle window excluded, like every mean above)
    "planar_drift_m", "drift_x_m", "drift_y_m", "planar_drift_settled_m",
    # effort
    "mean_torque_norm",
    # per-episode protocol record
    "base_mass_kg", "floor_friction", "n_pushes", "settle_steps", "wall_time_s",
]
