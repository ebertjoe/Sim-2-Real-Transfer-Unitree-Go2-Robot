#!/usr/bin/env python3
"""Robustness sweep of a Go2 locomotion policy in MuJoCo (thesis §4.1, Table 4.1).

MuJoCo counterpart of ``run_robustness_sweep_isaac.py``: same command grid, same
gaits, same seeds, same per-episode record, same CSV schema, so the two
``episodes.csv`` files can be compared row for row.

Evaluates a policy over the full velocity grid for every gait, five seeds, and
records per episode: survival and termination reason, mean tracking errors and
achieved values in vx / vy / wz, mean and standard deviation of the base height,
mean roll and pitch derived from projected gravity, the per-foot contact
fraction, the gait contact accuracy against the Raibert schedule, and - for
zero-command episodes - the planar drift at the end of the episode. A single
annotated mp4 of the sweep plus a seekable manifest are written alongside the
CSV.

``stand`` is evaluated at zero command only, so the sweep is
7 x 125 x 5 + 5 = 4380 episodes.

Examples
--------
    # latest export in resources/go2, all gaits, all seeds (~90 min)
    python3 sweep/run_robustness_sweep_mujoco.py

    # a specific checkpoint, with the sweep video
    python3 sweep/run_robustness_sweep_mujoco.py --video \
        --policy policy53Final.pt --video_stride 40

    # quick shakedown: one seed, two gaits
    python3 sweep/run_robustness_sweep_mujoco.py --seeds 1 --gaits trot,stand

    # watch it run in the MuJoCo viewer: every 20th episode, at 2x real time
    python3 sweep/run_robustness_sweep_mujoco.py --viewer --viewer_stride 20 \
        --viewer_speed 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robustness_sweep import grid as G
from robustness_sweep.checkpoints import (
    DEFAULT_POLICY_DIR,
    DEFAULT_XML,
    default_out_dir,
    resolve_policy,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MuJoCo robustness sweep (thesis section 4.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--xml", default=str(DEFAULT_XML), help="MuJoCo scene to evaluate in.")

    pol = p.add_argument_group("policy")
    pol.add_argument("--policy", default="latest",
                     help="'latest', a file name inside the policy dir, or a path.")
    pol.add_argument("--policy_dir", default=str(DEFAULT_POLICY_DIR),
                     help="Directory the TorchScript exports live in.")

    swp = p.add_argument_group("sweep")
    swp.add_argument("--seeds", type=int, default=G.N_SEEDS, help="Number of seeds (Table 4.1).")
    swp.add_argument("--base_seed", type=int, default=0, help="First seed value.")
    swp.add_argument("--gaits", default="all",
                     help="Comma separated gait names or ids to sweep, or 'all'.")
    swp.add_argument("--workers", type=int, default=1,
                     help="Episodes simulated in parallel processes. Worth raising only on "
                          "a machine with dedicated cores: on a shared/burstable VM the "
                          "processes throttle each other and the sweep gets slower.")
    swp.add_argument("--out_dir", default=None,
                     help="Output directory (default sweep/results/robustness_sweep/<policy>/mujoco).")
    swp.add_argument("--resume", action="store_true",
                     help="Append to an existing episodes.csv and skip the rows it already has.")
    swp.add_argument("--episode_steps", type=int, default=G.EPISODE_STEPS,
                     help="Control steps per episode (100 Hz).")
    swp.add_argument("--settle_steps", type=int, default=None,
                     help="Override the per-gait settle window excluded from the metrics.")
    swp.add_argument("--print_every", type=int, default=100,
                     help="Print the running survival table every N episodes.")
    swp.add_argument("--no_progress", action="store_true", help="Disable the progress bar.")
    swp.add_argument("--summarise_only", action="store_true",
                     help="Skip the sweep and only re-run the summary on an existing CSV.")

    prot = p.add_argument_group("protocol")
    prot.add_argument("--domain_rand", dest="domain_rand", action="store_true", default=True,
                      help="Keep the start-up mass/friction randomisation (default).")
    prot.add_argument("--no_domain_rand", dest="domain_rand", action="store_false",
                      help="Evaluate the nominal robot only.")
    prot.add_argument("--mass_range", type=float, nargs=2, default=(-1.0, 3.0),
                      help="Trunk mass offset range [kg]; align with the Isaac env cfg.")
    prot.add_argument("--friction_range", type=float, nargs=2, default=(0.6, 1.2),
                      help="Floor sliding friction range; align with the Isaac env cfg.")
    prot.add_argument("--push_robot", action="store_true", default=False,
                      help="Keep the interval push disturbance (off by default).")
    prot.add_argument("--push_interval", type=float, default=5.0,
                      help="Seconds between pushes when --push_robot is set.")
    prot.add_argument("--push_vel", type=float, default=0.5,
                      help="Push magnitude [m/s], uniform in [-v, v] on x and y.")
    prot.add_argument("--obs_noise", action="store_true", default=False,
                      help="Keep the observation corruption (off by default).")
    prot.add_argument("--deterministic_reset", action="store_true", default=False,
                      help="Zero the reset pose/joint-velocity randomisation.")
    prot.add_argument("--terminate_on_base_contact", dest="terminate_on_base_contact",
                      action="store_true", default=True,
                      help="End an episode when the trunk touches the ground (default).")
    prot.add_argument("--no_terminate_on_base_contact", dest="terminate_on_base_contact",
                      action="store_false", help="Fall height only.")
    prot.add_argument("--terminate_on_orientation", action="store_true", default=False,
                      help="Also end an episode when the base tilts past 60 deg.")
    prot.add_argument("--quiet_env", dest="quiet_env", action="store_true", default=True,
                      help="Suppress the per-episode log line (default).")
    prot.add_argument("--verbose_env", dest="quiet_env", action="store_false",
                      help="Log one line per episode.")

    live = p.add_argument_group("viewer")
    live.add_argument("--viewer", action="store_true", default=False,
                      help="Watch the sweep in the MuJoCo viewer while it runs (forces serial).")
    live.add_argument("--viewer_speed", type=float, default=1.0,
                      help="Playback speed: 1.0 = real time, 2.0 = twice as fast, "
                           "0 = as fast as the machine manages.")
    live.add_argument("--viewer_stride", type=int, default=1,
                      help="Show one episode every N; the rest run headless at full speed.")
    live.add_argument("--viewer_distance", type=float, default=3.0,
                      help="Camera distance from the robot [m].")

    vid = p.add_argument_group("video")
    vid.add_argument("--video", action="store_true", default=False,
                     help="Record one annotated mp4 of the sweep plus a manifest (forces serial).")
    vid.add_argument("--video_every", type=int, default=4,
                     help="Capture a frame every N control steps (4 -> 25 fps real time).")
    vid.add_argument("--video_fps", type=int, default=25, help="Frame rate of the mp4.")
    vid.add_argument("--video_stride", type=int, default=1,
                     help="Film one episode every N episodes.")
    vid.add_argument("--video_resolution", type=int, nargs=2, default=(1280, 720))
    vid.add_argument("--video_eye", type=float, nargs=3, default=(2.2, 2.2, 1.1),
                     help="Camera offset from the filmed robot.")
    return p


def parse_gaits(spec: str) -> list[int] | None:
    if spec.strip().lower() in ("all", ""):
        return None
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            gid = int(token)
            if gid not in G.GAIT_NAMES:
                raise SystemExit(f"unknown gait id {gid}")
        elif token in G.NAME_TO_GAIT_ID:
            gid = G.NAME_TO_GAIT_ID[token]
        else:
            raise SystemExit(
                f"unknown gait '{token}'; known: {', '.join(G.NAME_TO_GAIT_ID)}")
        out.append(gid)
    return out


def main() -> None:
    args = build_parser().parse_args()
    args.gaits = parse_gaits(args.gaits)

    policy_ref = resolve_policy(args.policy, args.policy_dir)
    print(f"[sweep] policy: {policy_ref.path}")

    from robustness_sweep.episode import check_policy
    check_policy(policy_ref.path)

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(policy_ref)

    from robustness_sweep.mujoco_backend import MujocoRobustnessSweep
    from robustness_sweep.summarise import summarise

    sweep = MujocoRobustnessSweep(args, policy_ref, out_dir)
    if not args.summarise_only:
        sweep.run()

    if sweep.csv_path.exists():
        summarise(sweep.csv_path, sweep.out_dir)
    print(f"[sweep] done: {sweep.out_dir}")


if __name__ == "__main__":
    main()
