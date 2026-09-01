"""MuJoCo backend of the robustness sweep: job loop, CSV, resume, video.

Runs every job from :mod:`robustness_sweep.grid` through
:class:`robustness_sweep.episode.EpisodeRunner` and streams one row per episode
into ``episodes.csv``. Episodes are independent, so they are farmed out to a
process pool; with ``--video`` the sweep runs in a single process because the
frames have to be captured between control steps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import platform
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from . import grid as G
from .episode import STEP_DT, EpisodeRunner, ProtocolCfg

try:
    from tqdm import tqdm
except ImportError:      # optional
    tqdm = None


# ── worker process ───────────────────────────────────────────────────────────

_WORKER: dict = {}


def _worker_init(xml_path: str, policy_path: str, cfg_kwargs: dict):
    _WORKER["runner"] = EpisodeRunner(xml_path, policy_path, ProtocolCfg(**cfg_kwargs))


def _worker_run(job: dict) -> tuple[dict, dict]:
    return job, run_job(_WORKER["runner"], job)


def run_job(runner: EpisodeRunner, job: dict) -> dict:
    """Run one episode; a crash is recorded as a row rather than killing the sweep."""
    t0 = time.time()
    try:
        metrics = runner.run_episode(
            gait_id=job["gait_id"],
            vel_cmd=[job["vx_cmd"], job["vy_cmd"], job["wz_cmd"]],
            seed=job["episode_seed"],
        )
    except Exception as exc:                      # noqa: BLE001 - reported in the row
        traceback.print_exc()
        metrics = {
            "gait_id": job["gait_id"],
            "gait_name": job["gait_name"],
            "vx_cmd": job["vx_cmd"],
            "vy_cmd": job["vy_cmd"],
            "wz_cmd": job["wz_cmd"],
            "survived": False,
            "termination_reason": f"error: {type(exc).__name__}",
            "survival_steps": -1,
            "survival_time_s": -1.0,
        }
    metrics["wall_time_s"] = round(time.time() - t0, 3)
    return metrics


# ── progress ─────────────────────────────────────────────────────────────────

class SweepStats:
    def __init__(self, total: int, gait_ids: list[int]):
        self.total = total
        self.done = self.survived = self.failed = self.errors = 0
        self.gait_done = {g: 0 for g in gait_ids}
        self.gait_survived = {g: 0 for g in gait_ids}
        self._t0 = time.time()

    def update(self, gait_id: int, survived: bool, error: bool):
        self.done += 1
        self.gait_done[gait_id] = self.gait_done.get(gait_id, 0) + 1
        if error:
            self.errors += 1
        elif survived:
            self.survived += 1
            self.gait_survived[gait_id] = self.gait_survived.get(gait_id, 0) + 1
        else:
            self.failed += 1

    def line(self) -> str:
        elapsed = time.time() - self._t0
        rate = self.done / max(elapsed, 1e-6)
        remaining = (self.total - self.done) / max(rate, 1e-9)
        h, rem = divmod(int(remaining), 3600)
        m, s = divmod(rem, 60)
        pct = 100 * self.survived / max(self.done - self.errors, 1)
        return (f"[{self.done}/{self.total}] survived={self.survived} ({pct:.1f}%)  "
                f"failed={self.failed}  errors={self.errors}  "
                f"{rate:.2f} ep/s  ETA {h:02d}h{m:02d}m{s:02d}s")

    def gait_table(self) -> str:
        rows = ["  survival by gait:"]
        for gid in sorted(self.gait_done):
            n, s = self.gait_done[gid], self.gait_survived.get(gid, 0)
            pct = 100 * s / n if n else 0.0
            rows.append(f"    {G.GAIT_NAMES[gid]:6s} {s:5d}/{n:<5d} ({pct:5.1f}%)")
        return "\n".join(rows)


# ── sweep ────────────────────────────────────────────────────────────────────

class MujocoRobustnessSweep:
    """Drives the whole sweep and owns ``episodes.csv``."""

    SIM_NAME = "mujoco"

    def __init__(self, args, policy_ref, out_dir: Path):
        self.args = args
        self.policy_ref = policy_ref
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "episodes.csv"
        self.video_path = self.out_dir / "sweep.mp4"

        self.cfg = ProtocolCfg(
            episode_steps=args.episode_steps,
            settle_steps=args.settle_steps,
            domain_rand=args.domain_rand,
            mass_range=tuple(args.mass_range),
            friction_range=tuple(args.friction_range),
            push_robot=args.push_robot,
            push_interval_s=args.push_interval,
            push_vel=args.push_vel,
            obs_noise=args.obs_noise,
            deterministic_reset=args.deterministic_reset,
            terminate_on_base_contact=args.terminate_on_base_contact,
            terminate_on_fall_height=True,
            terminate_on_orientation=args.terminate_on_orientation,
        )

        self.jobs = G.build_jobs(seeds=args.seeds, base_seed=args.base_seed,
                                 gaits=args.gaits)
        for run_id, job in enumerate(self.jobs):
            job["run_id"] = run_id

        self.gait_ids = sorted({job["gait_id"] for job in self.jobs})
        self._done_keys = self._existing_keys() if args.resume else set()
        self.pending = [j for j in self.jobs if G.job_key(j) not in self._done_keys]

    # ── csv ──────────────────────────────────────────────────────────────────

    def _existing_keys(self) -> set:
        if not self.csv_path.exists():
            return set()
        keys = set()
        with open(self.csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    keys.add(G.job_key(row))
                except (KeyError, TypeError, ValueError):
                    continue
        return keys

    def _row(self, job: dict, metrics: dict) -> dict:
        row = {
            "run_id": job["run_id"],
            "sim": self.SIM_NAME,
            "policy": self.policy_ref.name,
            "seed": job["seed"],
            "episode_seed": job["episode_seed"],
        }
        row.update(metrics)
        row.setdefault("zero_cmd", all(
            job[k] == 0.0 for k in ("vx_cmd", "vy_cmd", "wz_cmd")))
        return {field: row.get(field, "") for field in G.CSV_FIELDS}

    def _write_meta(self, n_pending: int):
        meta = {
            "sim": self.SIM_NAME,
            "created": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "policy": str(self.policy_ref.path),
            "policy_sha1": hashlib.sha1(self.policy_ref.path.read_bytes()).hexdigest(),
            "xml": str(self.args.xml),
            "protocol": asdict(self.cfg),
            "grid": {
                "vx": G.VX_CMDS, "vy": G.VY_CMDS, "wz": G.WZ_CMDS,
                "gaits": [G.GAIT_NAMES[g] for g in self.gait_ids],
                "seeds": self.args.seeds, "base_seed": self.args.base_seed,
            },
            "episodes_total": len(self.jobs),
            "episodes_this_run": n_pending,
            "workers": self.args.workers,
        }
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self):
        args = self.args
        total = len(self.pending)
        skipped = len(self.jobs) - total

        print(f"\n{'=' * 68}")
        print("  MuJoCo robustness sweep — thesis section 4.1")
        print(f"  policy      : {self.policy_ref.path}")
        print(f"  scene       : {args.xml}")
        print(f"  gaits       : {', '.join(G.GAIT_NAMES[g] for g in self.gait_ids)}")
        print(f"  seeds       : {args.seeds} (from {args.base_seed})")
        print(f"  episodes    : {len(self.jobs)}"
              + (f"  ({skipped} already recorded, {total} to run)" if skipped else ""))
        print(f"  episode len : {args.episode_steps * 0.01:.0f} s")
        print(f"  protocol    : domain_rand={self.cfg.domain_rand} "
              f"push={self.cfg.push_robot} obs_noise={self.cfg.obs_noise} "
              f"deterministic_reset={self.cfg.deterministic_reset}")
        serial_reason = ("  (serial: --viewer)" if args.viewer
                         else "  (serial: --video)" if args.video else "")
        print(f"  workers     : {args.workers}{serial_reason}")
        print(f"  output      : {self.out_dir}")
        print(f"{'=' * 68}\n")

        self._write_meta(total)
        if total == 0:
            print("[sweep] nothing to do — every episode is already in the CSV.")
            return

        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        stats = SweepStats(total, self.gait_ids)
        bar = tqdm(total=total, unit="ep") if (tqdm and not args.no_progress) else None

        with open(self.csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=G.CSV_FIELDS)
            if write_header:
                writer.writeheader()
                fh.flush()

            results = (self._run_serial()
                       if args.workers <= 1 or args.video or args.viewer
                       else self._run_parallel())
            try:
                for i, (job, metrics) in enumerate(results, start=1):
                    writer.writerow(self._row(job, metrics))
                    fh.flush()
                    error = str(metrics["termination_reason"]).startswith("error")
                    stats.update(job["gait_id"], bool(metrics["survived"]), error)

                    if bar:
                        bar.update(1)
                        bar.set_postfix_str(
                            f"{job['gait_name']} vx={job['vx_cmd']:.1f}")
                    elif not args.quiet_env:
                        print(f"  {job['gait_name']:6s} "
                              f"vx={job['vx_cmd']:+.2f} vy={job['vy_cmd']:+.2f} "
                              f"wz={job['wz_cmd']:+.2f} seed={job['seed']} -> "
                              f"{metrics['termination_reason']}")
                    if i % args.print_every == 0:
                        print(f"\n{stats.line()}\n{stats.gait_table()}\n")
            finally:
                if bar:
                    bar.close()

        print(f"\n{'=' * 68}")
        print(f"  sweep complete — {stats.line()}")
        print(stats.gait_table())
        print(f"  episodes -> {self.csv_path}")
        print(f"{'=' * 68}\n")

    # ── execution strategies ─────────────────────────────────────────────────

    def _run_serial(self):
        """Episodes one at a time, so the video recorder and the live viewer can
        follow the simulation step by step."""
        runner = EpisodeRunner(self.args.xml, self.policy_ref.path, self.cfg)
        video = self._open_video()
        viewer = self._open_viewer(runner)
        try:
            for index, job in enumerate(self.pending):
                if viewer is not None and not viewer.is_running():
                    self._report_viewer_stop(index)
                    return

                filming = video is not None and video.wants(index)
                watching = viewer is not None and viewer.wants(index)
                if not (filming or watching):
                    yield job, run_job(runner, job)
                    continue

                if filming:
                    video.start_clip(job, job["run_id"])
                if watching:
                    viewer.begin_episode(job)
                    runner.sync_hook = viewer.sync

                t0 = time.time()
                runner.reset(gait_id=job["gait_id"],
                             vel_cmd=[job["vx_cmd"], job["vy_cmd"], job["wz_cmd"]],
                             seed=job["episode_seed"])
                step = 0
                aborted = False
                while not runner.done:
                    runner.step()
                    if filming:
                        video.capture(runner, step)
                    if watching:
                        viewer.sync(STEP_DT)
                        if not viewer.is_running():
                            aborted = True
                            break
                    step += 1

                runner.sync_hook = None
                metrics = runner.get_episode_metrics()
                metrics["wall_time_s"] = round(time.time() - t0, 3)

                if aborted:
                    # The window was closed part way through: this episode never
                    # ran to a termination, so it is dropped rather than written
                    # as a row that --resume would then treat as done.
                    if filming:
                        metrics["termination_reason"] = "aborted"
                        video.end_clip(metrics)
                    self._report_viewer_stop(index)
                    return

                if filming:
                    video.end_clip(metrics)
                if watching:
                    viewer.end_episode(metrics)
                yield job, metrics
        finally:
            if video is not None:
                video.close()
                print(f"[sweep] video -> {video.path}")
                print(f"[sweep] manifest -> {video.path.with_name('video_manifest.csv')}")
            if viewer is not None:
                viewer.close()

    def _run_parallel(self):
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=self.args.workers,
            initializer=_worker_init,
            initargs=(str(self.args.xml), str(self.policy_ref.path), asdict(self.cfg)),
        ) as pool:
            yield from pool.imap(_worker_run, self.pending, chunksize=1)

    def _report_viewer_stop(self, index: int):
        print(f"[sweep] viewer closed — stopping ({index} of {len(self.pending)} "
              "episodes of this run recorded). Re-run with --resume to continue.")

    def _open_viewer(self, runner: EpisodeRunner):
        if not self.args.viewer:
            return None
        from .viewer import LiveViewer
        try:
            return LiveViewer(
                runner.m, runner.d,
                speed=self.args.viewer_speed,
                stride=self.args.viewer_stride,
                distance=self.args.viewer_distance,
            )
        except Exception as exc:                  # noqa: BLE001 - viewer is optional
            print(f"[sweep] viewer disabled: {type(exc).__name__}: {exc}")
            return None

    def _open_video(self):
        if not self.args.video:
            return None
        from .video import SweepVideo
        try:
            return SweepVideo(
                self.video_path,
                resolution=self.args.video_resolution,
                fps=self.args.video_fps,
                every=self.args.video_every,
                stride=self.args.video_stride,
                eye=self.args.video_eye,
            )
        except Exception as exc:                  # noqa: BLE001 - video is optional
            print(f"[sweep] video disabled: {type(exc).__name__}: {exc}")
            return None
