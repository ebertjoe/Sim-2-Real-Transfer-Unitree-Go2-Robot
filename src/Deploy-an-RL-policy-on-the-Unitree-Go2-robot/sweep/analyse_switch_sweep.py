#!/usr/bin/env python3
"""
analyse_switch_sweep.py — Visualise gait-switch sweep results.

Usage:
    python3 sweep/analyse_switch_sweep.py \
        --mujoco sweep/results/switch_results_raw.csv \
        --isaac  sweep/results/isaac_switch_results_raw.csv \
        --out-dir sweep/results/plots_switch

    # MuJoCo only:
    python3 sweep/analyse_switch_sweep.py \
        --mujoco sweep/results/switch_results_raw.csv \
        --out-dir sweep/results/plots_switch
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

GAIT_NAMES = {
    0:"bound",1:"trot",2:"hop",3:"amble",
    4:"pronk",5:"limp",6:"stand",7:"run",
}
GAIT_ORDER = ["bound","trot","hop","amble","pronk","limp","stand","run"]

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["survived"] = df["survived"].astype(bool)
    for col in ["vx_cmd","vy_cmd","wz_cmd"]:
        df[col] = df[col].round(2)
    return df


# ── 1. Survival rate heatmap (gait_from × gait_to) ───────────────────────────

def plot_survival_heatmap(dfs: dict, out_dir: Path, metric="survived",
                          title="Survival rate", cmap="RdYlGn",
                          vmin=0, vmax=1, fmt=".0%"):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(n*6.5, 5.5), constrained_layout=True)
    if n == 1: axes = [axes]

    for ax, (sim_name, df) in zip(axes, dfs.items()):
        pivot = (
            df.groupby(["gait_from_name","gait_to_name"])[metric]
              .mean()
              .unstack("gait_to_name")
              .reindex(index=GAIT_ORDER, columns=GAIT_ORDER)
        )
        im = ax.imshow(pivot.values, aspect="auto",
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       interpolation="nearest")
        ax.set_xticks(range(8)); ax.set_xticklabels(GAIT_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(8)); ax.set_yticklabels(GAIT_ORDER, fontsize=8)
        ax.set_xlabel("gait_to", fontsize=9)
        ax.set_ylabel("gait_from", fontsize=9)
        ax.set_title(sim_name, fontsize=11, fontweight="bold")
        for r in range(8):
            for c in range(8):
                v = pivot.values[r, c]
                if r == c or np.isnan(v): continue
                txt = f"{v:{fmt[1:]}}" if fmt != ".0%" else f"{v:.0%}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if (v < 0.3 or v > 0.8) else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title + "  (marginalised over vel cmds & seeds)",
                 fontsize=12, fontweight="bold")
    return fig


# ── 2. Recovery steps heatmap ─────────────────────────────────────────────────

def plot_recovery_heatmap(dfs: dict, metric: str, label: str, out_dir: Path):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(n*6.5, 5.5), constrained_layout=True)
    if n == 1: axes = [axes]

    all_vals = pd.concat(dfs.values())[metric].dropna()
    vmax = float(np.percentile(all_vals, 95)) if len(all_vals) else 200

    for ax, (sim_name, df) in zip(axes, dfs.items()):
        surv = df[df["survived"]]
        pivot = (
            surv.groupby(["gait_from_name","gait_to_name"])[metric]
                .mean()
                .unstack("gait_to_name")
                .reindex(index=GAIT_ORDER, columns=GAIT_ORDER)
        )
        im = ax.imshow(pivot.values, aspect="auto",
                       vmin=0, vmax=vmax, cmap="YlOrRd",
                       interpolation="nearest")
        ax.set_xticks(range(8)); ax.set_xticklabels(GAIT_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(8)); ax.set_yticklabels(GAIT_ORDER, fontsize=8)
        ax.set_xlabel("gait_to", fontsize=9)
        ax.set_ylabel("gait_from", fontsize=9)
        ax.set_title(sim_name, fontsize=11, fontweight="bold")
        for r in range(8):
            for c in range(8):
                v = pivot.values[r, c]
                if r == c or np.isnan(v): continue
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{label}  (surviving episodes, marginalised over vel & seeds)",
                 fontsize=12, fontweight="bold")
    return fig


# ── 3. Failure phase breakdown ────────────────────────────────────────────────

def plot_failure_phases(dfs: dict, out_dir: Path):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(n*7, 4.5), constrained_layout=True)
    if n == 1: axes = [axes]

    for ax, (sim_name, df) in zip(axes, dfs.items()):
        failed = df[~df["survived"]]
        if len(failed) == 0:
            ax.text(0.5, 0.5, f"{sim_name}\nNo failures", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{sim_name} — failure phase breakdown", fontsize=10, fontweight="bold")
            continue
        counts = (
            failed.groupby(["gait_from_name","termination_reason"])
                  .size()
                  .unstack("termination_reason", fill_value=0)
        )
        cols = [c for c in ["fall_phase1","fall_transition","fall_phase2"]
                if c in counts.columns]
        if not cols:
            ax.text(0.5, 0.5, f"{sim_name}\nNo failures", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{sim_name} — failure phase breakdown", fontsize=10, fontweight="bold")
            continue
        colors = {"fall_phase1":"#e74c3c",
                  "fall_transition":"#e67e22",
                  "fall_phase2":"#3498db"}
        counts[cols].plot(
            kind="bar", stacked=True, ax=ax,
            color=[colors[c] for c in cols],
            edgecolor="white", linewidth=0.4,
        )
        ax.set_xlabel("gait_from")
        ax.set_ylabel("Number of failures")
        ax.set_title(f"{sim_name} — failure phase breakdown", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Phase", fontsize=8)

    fig.suptitle("When do failures occur during the switch episode?",
                 fontsize=12, fontweight="bold")
    return fig


# ── 4. Transition impact: trans_min_height and trans_max_vx_error ─────────────

def plot_transition_impact(dfs: dict, out_dir: Path):
    n = len(dfs)
    fig, axes = plt.subplots(2, n, figsize=(n*6.5, 9), constrained_layout=True)

    for col, (sim_name, df) in enumerate(dfs.items()):
        surv = df[df["survived"]]

        # Min height during transition
        ax = axes[0, col] if n > 1 else axes[0]
        pivot = (
            surv.groupby(["gait_from_name","gait_to_name"])["trans_min_height"]
                .mean()
                .unstack("gait_to_name")
                .reindex(index=GAIT_ORDER, columns=GAIT_ORDER)
        )
        im = ax.imshow(pivot.values, aspect="auto",
                       vmin=0.15, vmax=0.35, cmap="RdYlGn",
                       interpolation="nearest")
        ax.set_xticks(range(8)); ax.set_xticklabels(GAIT_ORDER, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(8)); ax.set_yticklabels(GAIT_ORDER, fontsize=7)
        ax.set_title(f"{sim_name}\nMin height during transition (m)",
                     fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Max vx error during transition
        ax = axes[1, col] if n > 1 else axes[1]
        pivot2 = (
            surv.groupby(["gait_from_name","gait_to_name"])["trans_max_vx_error"]
                .mean()
                .unstack("gait_to_name")
                .reindex(index=GAIT_ORDER, columns=GAIT_ORDER)
        )
        all_v = surv["trans_max_vx_error"].dropna()
        vmax  = float(np.percentile(all_v, 95)) if len(all_v) else 1.0
        im2 = ax.imshow(pivot2.values, aspect="auto",
                        vmin=0, vmax=vmax, cmap="YlOrRd",
                        interpolation="nearest")
        ax.set_xticks(range(8)); ax.set_xticklabels(GAIT_ORDER, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(8)); ax.set_yticklabels(GAIT_ORDER, fontsize=7)
        ax.set_title(f"{sim_name}\nMax vx error during transition (m/s)",
                     fontsize=9, fontweight="bold")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Transition window impact  (first 1s after switch, surviving episodes)",
                 fontsize=11, fontweight="bold")
    return fig


# ── 5. Pre vs post tracking error ─────────────────────────────────────────────

def plot_pre_post_error(dfs: dict, out_dir: Path):
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(n*8, 4.5), constrained_layout=True)
    if n == 1: axes = [axes]

    for ax, (sim_name, df) in zip(axes, dfs.items()):
        surv = df[df["survived"]]
        grp  = surv.groupby("gait_to_name")[["pre_mean_vx_error","post_mean_vx_error"]].mean()
        grp  = grp.reindex([g for g in GAIT_ORDER if g in grp.index])
        x    = np.arange(len(grp))
        w    = 0.35
        ax.bar(x - w/2, grp["pre_mean_vx_error"],  width=w, label="pre-switch",
               color="#3498db", alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, grp["post_mean_vx_error"], width=w, label="post-switch",
               color="#e74c3c", alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(grp.index, rotation=30, ha="right")
        ax.set_ylabel("Mean |vx error| (m/s)")
        ax.set_title(f"{sim_name}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("vx tracking error before vs after gait switch  (surviving, all from-gaits)",
                 fontsize=11, fontweight="bold")
    return fig


# ── 6. Summary table ──────────────────────────────────────────────────────────

def print_summary(dfs: dict):
    for sim_name, df in dfs.items():
        print(f"\n{'='*65}")
        print(f"  {sim_name} — top 10 hardest gait-switch pairs")
        print(f"  (lowest survival rate, marginalised over vel & seeds)")
        print(f"{'='*65}")
        grp = (
            df.groupby(["gait_from_name","gait_to_name"])["survived"]
              .agg(n="count", sr="mean")
              .sort_values("sr")
              .head(10)
              .round(3)
        )
        print(grp.to_string())

        print(f"\n  Slowest recovery pairs (mean height_recovery_steps):")
        surv = df[df["survived"]]
        rec = (
            surv.groupby(["gait_from_name","gait_to_name"])["height_recovery_steps"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .round(1)
        )
        print(rec.to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mujoco",  default=None, help="MuJoCo switch CSV")
    p.add_argument("--isaac",   default=None, help="Isaac switch CSV")
    p.add_argument("--out-dir", default="sweep/results/plots_switch")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    if args.mujoco and Path(args.mujoco).exists():
        dfs["MuJoCo"] = load(args.mujoco)
        print(f"Loaded MuJoCo: {len(dfs['MuJoCo'])} rows")
    if args.isaac and Path(args.isaac).exists():
        dfs["Isaac"] = load(args.isaac)
        print(f"Loaded Isaac:  {len(dfs['Isaac'])} rows")

    if not dfs:
        print("[error] No valid CSV paths provided.")
        sys.exit(1)

    print(f"\nOutput directory: {out_dir}\nGenerating plots...\n")

    fig = plot_survival_heatmap(dfs, out_dir)
    _save(fig, out_dir / "1_survival_heatmap.png")

    fig = plot_recovery_heatmap(dfs, "height_recovery_steps",
                                "Height recovery steps", out_dir)
    _save(fig, out_dir / "2_height_recovery_heatmap.png")

    fig = plot_recovery_heatmap(dfs, "vel_recovery_steps",
                                "Velocity recovery steps", out_dir)
    _save(fig, out_dir / "3_vel_recovery_heatmap.png")

    fig = plot_failure_phases(dfs, out_dir)
    _save(fig, out_dir / "4_failure_phases.png")

    fig = plot_transition_impact(dfs, out_dir)
    _save(fig, out_dir / "5_transition_impact.png")

    fig = plot_pre_post_error(dfs, out_dir)
    _save(fig, out_dir / "6_pre_post_vx_error.png")

    print_summary(dfs)
    print(f"\nAll plots saved to: {out_dir}\n")


if __name__ == "__main__":
    main()