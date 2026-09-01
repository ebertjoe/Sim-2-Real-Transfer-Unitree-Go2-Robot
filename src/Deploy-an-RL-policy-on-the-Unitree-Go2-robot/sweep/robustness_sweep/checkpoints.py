"""Policy resolution for the MuJoCo sweep.

The Isaac backend resolves ``--run``/``--checkpoint`` inside the rsl_rl log
tree; on the MuJoCo side a policy is a single TorchScript ``.pt`` export, so
the equivalent is: a path, a bare file name inside ``resources/go2``, or
``latest`` (the most recently modified export).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# <repo>/sweep/robustness_sweep/checkpoints.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = REPO_ROOT / "resources" / "go2"
DEFAULT_XML = REPO_ROOT / "resources" / "go2" / "scene_flat.xml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "sweep" / "results" / "robustness_sweep"


@dataclass(frozen=True)
class PolicyRef:
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem


def resolve_policy(policy: str = "latest",
                   policy_dir: str | Path = DEFAULT_POLICY_DIR) -> PolicyRef:
    """Return the checkpoint referred to by ``policy``.

    ``latest``          most recently modified ``*.pt`` in ``policy_dir``
    ``policy53.pt``     that file inside ``policy_dir`` (``.pt`` optional)
    ``/some/where.pt``  used as given
    """
    policy_dir = Path(policy_dir).expanduser()

    if policy.strip().lower() == "latest":
        candidates = sorted(policy_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit(f"no *.pt checkpoint found in {policy_dir}")
        return PolicyRef(candidates[-1].resolve())

    direct = Path(policy).expanduser()
    if direct.is_file():
        return PolicyRef(direct.resolve())

    for candidate in (policy_dir / policy, policy_dir / f"{policy}.pt"):
        if candidate.is_file():
            return PolicyRef(candidate.resolve())

    raise SystemExit(f"policy '{policy}' not found (looked in {policy_dir})")


def default_out_dir(policy_ref: PolicyRef) -> Path:
    """``sweep/results/robustness_sweep/<policy>/mujoco``.

    Mirrors the Isaac layout ``<run>/robustness_sweep/isaac``, so the two
    simulators end up as sibling directories under the same policy.
    """
    return DEFAULT_RESULTS_ROOT / policy_ref.name / "mujoco"
