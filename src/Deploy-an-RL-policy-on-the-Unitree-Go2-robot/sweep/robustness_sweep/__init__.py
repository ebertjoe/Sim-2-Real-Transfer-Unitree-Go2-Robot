"""Robustness sweep of a Go2 locomotion policy (thesis section 4.1).

``grid`` holds the command grid, the gait table and the CSV schema shared by
both simulators; ``mujoco_backend`` runs the sweep in MuJoCo; ``summarise``
aggregates ``episodes.csv``.
"""
