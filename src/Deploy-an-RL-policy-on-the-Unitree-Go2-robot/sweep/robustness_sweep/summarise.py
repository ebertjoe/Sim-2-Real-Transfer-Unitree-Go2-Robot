"""Aggregate ``episodes.csv`` into the tables used in the thesis.

Writes next to the episode CSV:

``summary_by_gait.csv``           one row per gait, averaged over the grid
``summary_by_gait_command.csv``   one row per gait x command, averaged over seeds
``summary_by_gait_speed.csv``     one row per gait x vx, the robustness curve
``summary_terminations.csv``      termination reason counts per gait

and prints the per-gait table to the console.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Metrics averaged over episodes wherever a group is aggregated.
METRIC_COLUMNS = [
    "mean_vx_error", "mean_vy_error", "mean_wz_error",
    "mean_vx", "mean_vy", "mean_wz",
    "mean_base_height", "std_base_height",
    "mean_roll", "mean_pitch", "mean_abs_roll", "mean_abs_pitch",
    "contact_frac_FR", "contact_frac_FL", "contact_frac_RR", "contact_frac_RL",
    "mean_contact_frac", "gait_contact_accuracy",
    "mean_torque_norm", "survival_time_s",
]


def _aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Episode count, survival rate and the mean of every metric per group.

    Metrics are averaged over surviving episodes only: a fall truncates the
    buffers, so the numbers of a failed episode describe a different (shorter)
    window and would bias the tracking and contact means.
    """
    grouped = df.groupby(keys, dropna=False)
    out = grouped.agg(
        n_episodes=("survived", "size"),
        n_survived=("survived", "sum"),
        mean_survival_time_s=("survival_time_s", "mean"),
    )
    out["survival_rate"] = out["n_survived"] / out["n_episodes"]

    survivors = df[df["survived"]].groupby(keys, dropna=False)
    present = [c for c in METRIC_COLUMNS if c in df.columns]
    out = out.join(survivors[present].mean(), how="left")

    zero = df[df["zero_cmd"]]
    if not zero.empty:
        out = out.join(
            zero.groupby(keys, dropna=False)["planar_drift_m"]
                .mean().rename("zero_cmd_planar_drift_m"),
            how="left",
        )
    return out.reset_index()


def summarise(csv_path: str | Path, out_dir: str | Path | None = None) -> dict:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[summary] episodes.csv is empty — nothing to summarise.")
        return {}

    for col in ("survived", "zero_cmd"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1", "yes"))

    tables = {
        "summary_by_gait": _aggregate(df, ["gait_id", "gait_name"]),
        "summary_by_gait_command": _aggregate(
            df, ["gait_id", "gait_name", "vx_cmd", "vy_cmd", "wz_cmd"]),
        "summary_by_gait_speed": _aggregate(df, ["gait_id", "gait_name", "vx_cmd"]),
    }

    terminations = (df.groupby(["gait_name", "termination_reason"])
                      .size().rename("n").reset_index())
    tables["summary_terminations"] = terminations

    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    _print_gait_table(tables["summary_by_gait"])
    print(f"[summary] tables -> {out_dir}")
    return tables


def _print_gait_table(by_gait: pd.DataFrame):
    header = (f"{'gait':7s}{'n':>6s}{'surv':>8s}{'vx err':>9s}{'vy err':>9s}"
              f"{'wz err':>9s}{'height':>9s}{'h std':>8s}{'|roll|':>8s}"
              f"{'|pitch|':>9s}{'contact':>9s}{'drift':>8s}")
    print(f"\n{header}\n{'-' * len(header)}")
    for _, r in by_gait.sort_values("gait_id").iterrows():
        drift = r.get("zero_cmd_planar_drift_m", float("nan"))
        print(f"{r['gait_name']:7s}{int(r['n_episodes']):6d}"
              f"{100 * r['survival_rate']:7.1f}%"
              f"{r['mean_vx_error']:9.3f}{r['mean_vy_error']:9.3f}"
              f"{r['mean_wz_error']:9.3f}{r['mean_base_height']:9.3f}"
              f"{r['std_base_height']:8.3f}{r['mean_abs_roll']:8.3f}"
              f"{r['mean_abs_pitch']:9.3f}{r['gait_contact_accuracy']:9.3f}"
              f"{drift:8.3f}")
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: PYTHONPATH=sweep python3 -m robustness_sweep.summarise <episodes.csv>")
    summarise(sys.argv[1])
