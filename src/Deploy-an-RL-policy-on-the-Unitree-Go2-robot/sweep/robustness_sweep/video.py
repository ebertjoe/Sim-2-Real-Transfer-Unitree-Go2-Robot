"""Annotated sweep video: one mp4 for the whole sweep plus a seekable manifest.

Every ``video_stride``-th episode is filmed, every ``video_every``-th control
step is captured, and each frame is annotated with the gait, the command, the
seed and the live state. ``video_manifest.csv`` maps every filmed episode to
its frame and time range in the mp4, so a row of ``episodes.csv`` can be found
by seeking rather than by scrubbing.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

MANIFEST_FIELDS = [
    "clip_index", "run_id", "gait_id", "gait_name",
    "vx_cmd", "vy_cmd", "wz_cmd", "seed",
    "frame_start", "frame_end", "t_start_s", "t_end_s",
    "survived", "termination_reason",
]


class SweepVideo:
    """Appends every filmed episode to a single mp4."""

    def __init__(self, path: Path, resolution=(1280, 720), fps: int = 25,
                 every: int = 4, stride: int = 1, eye=(2.2, 2.2, 1.1)):
        import cv2  # noqa: PLC0415 - optional dependency, only needed with --video
        import mujoco  # noqa: PLC0415

        self._cv2 = cv2
        self._mujoco = mujoco

        self.path = Path(path)
        self.width, self.height = int(resolution[0]), int(resolution[1])
        self.fps = int(fps)
        self.every = max(int(every), 1)
        self.stride = max(int(stride), 1)
        self.eye = np.asarray(eye, dtype=float)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps, (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open {self.path} for writing")

        self._manifest_path = self.path.with_name("video_manifest.csv")
        self._manifest_fh = open(self._manifest_path, "w", newline="")
        self._manifest = csv.DictWriter(self._manifest_fh, fieldnames=MANIFEST_FIELDS)
        self._manifest.writeheader()

        self._renderer = None
        self._camera = None
        self._frames = 0
        self._clips = 0
        self._clip_start = 0
        self._job = None

    # ── camera ───────────────────────────────────────────────────────────────

    def _ensure_renderer(self, model):
        if self._renderer is not None:
            return
        self._renderer = self._mujoco.Renderer(model, self.height, self.width)
        cam = self._mujoco.MjvCamera()
        cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        # MuJoCo places the camera at lookat + d * (-cos(el)cos(az),
        # -cos(el)sin(az), -sin(el)); invert that for the requested eye offset.
        ex, ey, ez = self.eye
        d = float(np.linalg.norm(self.eye)) or 1.0
        cam.distance = d
        cam.elevation = math.degrees(math.asin(np.clip(-ez / d, -1.0, 1.0)))
        cam.azimuth = math.degrees(math.atan2(-ey, -ex))
        self._camera = cam

    # ── recording ────────────────────────────────────────────────────────────

    def wants(self, episode_index: int) -> bool:
        return episode_index % self.stride == 0

    def start_clip(self, job: dict, run_id: int):
        self._job = dict(job, run_id=run_id)
        self._clip_start = self._frames

    def capture(self, runner, step: int):
        if self._job is None or step % self.every:
            return
        self._ensure_renderer(runner.m)
        self._camera.lookat[:] = runner.d.qpos[0:3]
        self._renderer.update_scene(runner.d, camera=self._camera)
        frame = self._renderer.render()
        self._writer.write(self._annotate(frame, runner, step))
        self._frames += 1

    def end_clip(self, metrics: dict):
        if self._job is None:
            return
        job = self._job
        self._manifest.writerow({
            "clip_index": self._clips,
            "run_id": job["run_id"],
            "gait_id": job["gait_id"],
            "gait_name": job["gait_name"],
            "vx_cmd": job["vx_cmd"],
            "vy_cmd": job["vy_cmd"],
            "wz_cmd": job["wz_cmd"],
            "seed": job["seed"],
            "frame_start": self._clip_start,
            "frame_end": max(self._frames - 1, self._clip_start),
            "t_start_s": round(self._clip_start / self.fps, 3),
            "t_end_s": round(max(self._frames - 1, self._clip_start) / self.fps, 3),
            "survived": metrics["survived"],
            "termination_reason": metrics["termination_reason"],
        })
        self._manifest_fh.flush()
        self._clips += 1
        self._job = None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        self._writer.release()
        self._manifest_fh.close()

    # ── overlay ──────────────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, runner, step: int) -> np.ndarray:
        cv2 = self._cv2
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        job = self._job
        lin = runner.d.sensordata[52:55]
        lines = [
            f"{job['gait_name']}  (gait {job['gait_id']})   seed {job['seed']}"
            f"   run {job['run_id']}",
            f"cmd  vx={job['vx_cmd']:+.2f}  vy={job['vy_cmd']:+.2f}"
            f"  wz={job['wz_cmd']:+.2f}",
            f"act  vx={float(lin[0]):+.2f}  vy={float(lin[1]):+.2f}"
            f"  wz={float(runner.d.sensordata[42]):+.2f}",
            f"t={step * 0.01:5.2f}s   h={float(runner.d.qpos[2]):.3f}m",
        ]
        y = 28
        for line in lines:
            cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 26
        return img
