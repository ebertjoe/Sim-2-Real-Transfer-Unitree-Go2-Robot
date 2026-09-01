"""Live MuJoCo viewer for the sweep: watch the episodes as they are evaluated.

The passive viewer shares the model and data of the running
:class:`~robustness_sweep.episode.EpisodeRunner`, so what is on screen is the
episode being scored - nothing is re-simulated for the picture. The viewer
paces the simulation against the wall clock (``--viewer_speed``, 0 = as fast as
the machine manages) and tracks the base with the camera; closing the window
stops the sweep after the current episode, and ``--resume`` picks it up again.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from . import grid as G


class LiveViewer:
    """Passive viewer wrapper that paces and annotates the running sweep."""

    def __init__(self, model, data, speed: float = 1.0, stride: int = 1,
                 track: bool = True, distance: float = 3.0,
                 azimuth: float = 135.0, elevation: float = -20.0):
        self.speed = float(speed)
        self.stride = max(int(stride), 1)
        self.track = track

        self._viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False)
        self._data = data
        with self._viewer.lock():
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._viewer.cam.distance = float(distance)
            self._viewer.cam.azimuth = float(azimuth)
            self._viewer.cam.elevation = float(elevation)

        self._sim_t = 0.0
        self._wall_t0 = time.time()

    # ── episode framing ──────────────────────────────────────────────────────

    def wants(self, episode_index: int) -> bool:
        """Only every ``stride``-th episode is drawn; the rest run at full speed."""
        return episode_index % self.stride == 0

    def begin_episode(self, job: dict):
        print(f"[viewer] {G.GAIT_NAMES[job['gait_id']]:6s} "
              f"vx={job['vx_cmd']:+.2f} vy={job['vy_cmd']:+.2f} wz={job['wz_cmd']:+.2f} "
              f"seed={job['seed']}  (run {job['run_id']})")
        self._sim_t = 0.0
        self._wall_t0 = time.time()

    def end_episode(self, metrics: dict):
        print(f"[viewer]   -> {metrics['termination_reason']} after "
              f"{metrics['survival_time_s']:.2f}s")

    # ── per-step sync ────────────────────────────────────────────────────────

    def sync(self, sim_dt: float):
        """Draw the current state and wait until the wall clock catches up."""
        if self.track:
            with self._viewer.lock():
                self._viewer.cam.lookat[:] = self._data.qpos[0:3]
        self._viewer.sync()

        self._sim_t += sim_dt
        if self.speed > 0.0:
            wait = self._wall_t0 + self._sim_t / self.speed - time.time()
            if wait > 0:
                time.sleep(wait)

    def is_running(self) -> bool:
        return self._viewer.is_running()

    def close(self):
        self._viewer.close()
