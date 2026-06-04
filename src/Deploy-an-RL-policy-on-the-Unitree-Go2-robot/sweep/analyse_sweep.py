#!/usr/bin/env python3
"""
analyse_sweep.py — Visualise results from run_sweep.py

Usage:
    python analyse_sweep.py [--csv results/results_raw.csv] [--out-dir results/plots]

Sections:
    1. Load & sanity-check
    2. Success rate heatmaps (vx × vy per gait)
    3. Velocity tracking error heatmaps
    4. Yaw response — did the policy learn to turn?
    5. Survival time distributions (failed episodes)
    6. Contact accuracy per gait
    7. Stand gait stability
    8. Torque effort per gait
    9. Summary tables
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves PNGs, no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────

GAIT_NAMES = {
    0: "bound", 1: "trot", 2: "hop",   3: "amble",
    4: "pronk", 5: "limp", 6: "stand", 7: "run",
}
LOCOMOTION_GAITS = [g for g in GAIT_NAMES if g != 6]

GAIT_TABLE_THRESHOLD = {
    0: 0.4, 1: 0.5, 2: 0.5, 3: 0.625, 4: 0.5, 5: 0.5, 7: 0.4
}

plt.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
})


def make_palette():
    colours = sns.color_palette("tab10", len(GAIT_NAMES))
    return {g: c for g, c in zip(GAIT_NAMES.keys(), colours)}


# ── 1 · Load & sanity-check ───────────────────────────────────────────────────

def load(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw    = pd.read_csv(csv_path)
    errors = raw[raw["termination_reason"].str.startswith("error", na=False)].copy()
    df     = raw[~raw["termination_reason"].str.startswith("error", na=False)].copy()

    df["gait_name"] = df["gait_id"].map(GAIT_NAMES)
    df["survived"]  = df["survived"].astype(bool)
    for col in ["vx_cmd", "vy_cmd", "wz_cmd"]:
        df[col] = df[col].round(2)

    print(f"\n{'='*55}")
    print(f"  SWEEP RESULTS — {csv_path}")
    print(f"{'='*55}")
    print(f"  Total episodes : {len(raw)}")
    print(f"  Valid          : {len(df)}")
    print(f"  Errors         : {len(errors)}")
    print(f"  Survived       : {df['survived'].sum()}  "
          f"({100*df['survived'].mean():.1f}%)")
    print(f"  Failed         : {(~df['survived']).sum()}")
    print()
    print("  Termination reasons:")
    for reason, count in df["termination_reason"].value_counts().items():
        print(f"    {reason:<30} {count}")
    print()
    print("  Episodes per gait:")
    sr = (
        df.groupby("gait_name")["survived"]
          .agg(n="count", n_survived="sum", survival_rate="mean")
          .round(3)
    )
    print(sr.to_string())
    print()
    return df, errors


# ── 2 · Success rate heatmaps ─────────────────────────────────────────────────

def plot_success_heatmaps(df: pd.DataFrame, out_dir: Path, palette: dict):
    gaits = LOCOMOTION_GAITS
    ncols = 4
    nrows = int(np.ceil(len(gaits) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.6, nrows * 3.4),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, gait_id in zip(axes, gaits):
        sub = df[df["gait_id"] == gait_id]
        pivot = (
            sub.groupby(["vx_cmd", "vy_cmd"])["survived"]
               .mean()
               .unstack("vy_cmd")
               .sort_index(ascending=False)
        )
        im = ax.imshow(
            pivot.values, aspect="auto",
            vmin=0, vmax=1, cmap="RdYlGn", interpolation="nearest",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=8)
        ax.set_xlabel("vy_cmd (m/s)", fontsize=8)
        ax.set_ylabel("vx_cmd (m/s)", fontsize=8)
        ax.set_title(f"{GAIT_NAMES[gait_id]}  (id={gait_id})",
                     fontsize=10, fontweight="bold")
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.0%}", ha="center", va="center",
                            fontsize=7,
                            color="black" if 0.2 < v < 0.8 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(gaits):]:
        ax.set_visible(False)

    fig.suptitle("Survival rate  (marginalised over wz & seeds)",
                 fontsize=13, fontweight="bold")
    _save(fig, out_dir / "2_success_heatmaps.png")


# ── 3 · Tracking error heatmaps ───────────────────────────────────────────────

def _tracking_grid(df, metric, label, out_path, cmap="YlOrRd"):
    survived = df[df["survived"]]
    gaits    = LOCOMOTION_GAITS
    ncols    = 4
    nrows    = int(np.ceil(len(gaits) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.6, nrows * 3.4),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    all_vals = survived.loc[survived["gait_id"].isin(gaits), metric].dropna()
    auto_vmax = float(np.percentile(all_vals, 95)) if len(all_vals) else 1.0

    for ax, gait_id in zip(axes, gaits):
        sub = survived[survived["gait_id"] == gait_id]
        if sub.empty:
            ax.set_title(f"{GAIT_NAMES[gait_id]} (no survivors)")
            continue
        pivot = (
            sub.groupby(["vx_cmd", "vy_cmd"])[metric]
               .mean()
               .unstack("vy_cmd")
               .sort_index(ascending=False)
        )
        im = ax.imshow(
            pivot.values, aspect="auto",
            vmin=0, vmax=auto_vmax,
            cmap=cmap, interpolation="nearest",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=8)
        ax.set_xlabel("vy_cmd (m/s)", fontsize=8)
        ax.set_ylabel("vx_cmd (m/s)", fontsize=8)
        ax.set_title(f"{GAIT_NAMES[gait_id]}  (id={gait_id})",
                     fontsize=10, fontweight="bold")
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(gaits):]:
        ax.set_visible(False)

    fig.suptitle(f"{label}  — surviving episodes only  (marginalised over wz & seeds)",
                 fontsize=12, fontweight="bold")
    _save(fig, out_path)


def plot_tracking_heatmaps(df: pd.DataFrame, out_dir: Path):
    _tracking_grid(df, "mean_vx_error", "Mean |vx error| (m/s)",
                   out_dir / "3a_vx_error_heatmaps.png")
    _tracking_grid(df, "mean_vy_error", "Mean |vy error| (m/s)",
                   out_dir / "3b_vy_error_heatmaps.png")


# ── 4 · Yaw response ─────────────────────────────────────────────────────────

def plot_yaw_response(df: pd.DataFrame, out_dir: Path, palette: dict):
    survived = df[df["survived"] & df["gait_id"].isin(LOCOMOTION_GAITS)]
    gaits    = LOCOMOTION_GAITS

    fig, axes = plt.subplots(2, 4, figsize=(16, 7),
                              constrained_layout=True, sharey=True)
    axes = axes.flatten()

    for ax, gait_id in zip(axes, gaits):
        sub = survived[survived["gait_id"] == gait_id]
        if sub.empty:
            ax.set_visible(False)
            continue

        grp = (sub.groupby("wz_cmd")["mean_wz_actual"]
                  .agg(["mean", "std"])
                  .reset_index())
        ax.errorbar(
            grp["wz_cmd"], grp["mean"], yerr=grp["std"],
            fmt="o-", color=palette[gait_id],
            capsize=4, linewidth=1.5, markersize=5,
        )
        lims = [-0.6, 0.6]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="ideal 1:1")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlim(-0.65, 0.65)
        ax.set_xlabel("wz_cmd (rad/s)", fontsize=9)
        ax.set_ylabel("mean wz_actual (rad/s)", fontsize=9)
        ax.set_title(GAIT_NAMES[gait_id], fontsize=10, fontweight="bold")

        if len(sub) > 5:
            r = sub["wz_cmd"].corr(sub["mean_wz_actual"])
            ax.text(0.05, 0.92, f"r = {r:.2f}",
                    transform=ax.transAxes, fontsize=8, color="dimgray")

    for ax in axes[len(gaits):]:
        ax.set_visible(False)

    fig.suptitle("Yaw response: actual ω_z vs commanded ω_z  (surviving episodes)",
                 fontsize=13, fontweight="bold")
    _save(fig, out_dir / "4_yaw_response.png")


# ── 5 · Survival time distributions ──────────────────────────────────────────

def plot_survival_distributions(df: pd.DataFrame, out_dir: Path, palette: dict):
    failed  = df[(~df["survived"]) & df["gait_id"].isin(LOCOMOTION_GAITS)]
    order   = [GAIT_NAMES[g] for g in LOCOMOTION_GAITS]
    pal     = {GAIT_NAMES[g]: palette[g] for g in LOCOMOTION_GAITS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    if not failed.empty:
        sns.violinplot(
            data=failed, x="gait_name", y="survival_time_s",
            order=order, palette=pal,
            inner="quartile", cut=0, ax=ax,
        )
    ax.set_xlabel("Gait")
    ax.set_ylabel("Survival time (s)")
    ax.set_title("Survival time distribution — failed episodes")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    if not failed.empty:
        reason_counts = (
            failed.groupby(["gait_name", "termination_reason"])
                  .size()
                  .unstack("termination_reason", fill_value=0)
                  .reindex(order)
        )
        reason_counts.plot(
            kind="bar", stacked=True, ax=ax,
            color=["#e74c3c", "#e67e22", "#3498db"][:len(reason_counts.columns)],
            edgecolor="white", linewidth=0.5,
        )
    ax.set_xlabel("Gait")
    ax.set_ylabel("Number of failures")
    ax.set_title("Failure type breakdown per gait")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Reason", fontsize=8)

    fig.suptitle("Failure analysis", fontsize=12, fontweight="bold")
    _save(fig, out_dir / "5_survival_distributions.png")


# ── 6 · Contact accuracy ──────────────────────────────────────────────────────

def plot_contact_accuracy(df: pd.DataFrame, out_dir: Path, palette: dict):
    loco  = df[df["gait_id"].isin(LOCOMOTION_GAITS)]
    order = [GAIT_NAMES[g] for g in LOCOMOTION_GAITS]
    pal   = {GAIT_NAMES[g]: palette[g] for g in LOCOMOTION_GAITS}
    fprops = dict(marker=".", markersize=3, alpha=0.4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    ax = axes[0]
    sns.boxplot(
        data=loco, x="gait_name", y="mean_contact_acc",
        order=order, palette=pal, width=0.5, ax=ax, flierprops=fprops,
    )
    ax.axhline(1.0, color="green", linestyle="--",
               linewidth=0.8, alpha=0.6, label="perfect")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Gait")
    ax.set_ylabel("Contact accuracy (fraction)")
    ax.set_title("Foot contact accuracy  (actual vs desFeetContact)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

    ax = axes[1]
    sns.boxplot(
        data=loco, x="gait_name", y="mean_contact_frac",
        order=order, palette=pal, width=0.5, ax=ax, flierprops=fprops,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Gait")
    ax.set_ylabel("Mean contact fraction (per foot)")
    ax.set_title("Mean fraction of feet in contact")
    ax.tick_params(axis="x", rotation=30)

    # Expected stance fraction from gait table
    for gait_id, thr in GAIT_TABLE_THRESHOLD.items():
        if gait_id in LOCOMOTION_GAITS:
            idx = order.index(GAIT_NAMES[gait_id])
            ax.hlines(thr, idx - 0.4, idx + 0.4,
                      colors="red", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.plot([], [], "r--", linewidth=1.5, alpha=0.7, label="expected (threshold)")
    ax.legend(fontsize=8)

    fig.suptitle("Contact quality per gait", fontsize=12, fontweight="bold")
    _save(fig, out_dir / "6_contact_accuracy.png")


# ── 7 · Stand gait stability ──────────────────────────────────────────────────

def plot_stand_stability(df: pd.DataFrame, out_dir: Path):
    stand = df[df["gait_id"] == 6]
    if stand.empty:
        print("  [skip] No stand gait rows found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metrics = [
        ("mean_height", "Mean base height (m)",     "steelblue"),
        ("std_height",  "Std base height (m)",       "darkorange"),
        ("xy_drift_m",  "XY positional drift (m)",   "seagreen"),
    ]
    for ax, (col, label, color) in zip(axes, metrics):
        vals = stand[col].dropna()
        ax.hist(vals, bins=max(5, len(vals) // 2),
                color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={vals.mean():.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Stand gait stability  (all seeds)",
                 fontsize=12, fontweight="bold")
    _save(fig, out_dir / "7_stand_stability.png")

    print("\n  Stand gait stats:")
    print(stand[["mean_height", "std_height", "xy_drift_m",
                 "mean_roll", "mean_pitch"]]
          .describe().round(4).to_string())
    print()


# ── 8 · Torque effort ────────────────────────────────────────────────────────

def plot_torque_effort(df: pd.DataFrame, out_dir: Path, palette: dict):
    loco_surv = df[df["gait_id"].isin(LOCOMOTION_GAITS) & df["survived"]]
    order = [GAIT_NAMES[g] for g in LOCOMOTION_GAITS]
    pal   = {GAIT_NAMES[g]: palette[g] for g in LOCOMOTION_GAITS}
    fprops = dict(marker=".", markersize=3, alpha=0.4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    ax = axes[0]
    sns.boxplot(
        data=loco_surv, x="gait_name", y="mean_torque_norm",
        order=order, palette=pal, width=0.5, ax=ax, flierprops=fprops,
    )
    ax.set_xlabel("Gait")
    ax.set_ylabel("Mean torque L2 norm (N·m)")
    ax.set_title("Joint torque effort per gait  (surviving episodes)")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    for gait_id in LOCOMOTION_GAITS:
        sub = loco_surv[loco_surv["gait_id"] == gait_id]
        if sub.empty:
            continue
        grp = sub.groupby("vx_cmd")["mean_torque_norm"].mean().reset_index()
        ax.plot(
            grp["vx_cmd"].to_numpy(), grp["mean_torque_norm"].to_numpy(),
            "o-", color=palette[gait_id],
            label=GAIT_NAMES[gait_id], linewidth=1.5, markersize=5,
        )
    ax.set_xlabel("vx_cmd (m/s)")
    ax.set_ylabel("Mean torque L2 norm (N·m)")
    ax.set_title("Torque vs forward speed  (marginalised over vy, wz)")
    ax.legend(title="Gait", fontsize=8, ncol=2)

    fig.suptitle("Torque effort", fontsize=12, fontweight="bold")
    _save(fig, out_dir / "8_torque_effort.png")


# ── 9 · Summary tables ────────────────────────────────────────────────────────

def print_summary_tables(df: pd.DataFrame):
    print("\n" + "="*65)
    print("  SUMMARY: worst-case (vx, vy) cell per gait")
    print("  (marginalised over wz & seeds)")
    print("="*65)

    rows = []
    for gait_id in LOCOMOTION_GAITS:
        sub = df[df["gait_id"] == gait_id]
        if sub.empty:
            continue
        grp = sub.groupby(["vx_cmd", "vy_cmd"])["survived"].mean()
        worst_idx = grp.idxmin()
        best_idx  = grp.idxmax()
        rows.append({
            "gait":       GAIT_NAMES[gait_id],
            "overall_sr": f"{sub['survived'].mean():.0%}",
            "worst_cell": f"vx={worst_idx[0]:.1f} vy={worst_idx[1]:.1f}",
            "worst_sr":   f"{grp.min():.0%}",
            "best_cell":  f"vx={best_idx[0]:.1f}  vy={best_idx[1]:.1f}",
            "best_sr":    f"{grp.max():.0%}",
        })

    print(pd.DataFrame(rows).set_index("gait").to_string())

    print("\n\n" + "="*65)
    print("  SUMMARY: mean tracking errors  (surviving episodes)")
    print("="*65)
    err = (
        df[df["survived"] & df["gait_id"].isin(LOCOMOTION_GAITS)]
          .groupby("gait_name")[
              ["mean_vx_error", "mean_vy_error", "mean_wz_error",
               "mean_contact_acc", "mean_torque_norm"]
          ]
          .mean()
          .round(3)
    )
    print(err.to_string())
    print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def plot_comparison(df_m, df_i, out_dir: Path):
    """Side-by-side MuJoCo vs Isaac comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    loco = [g for g in GAIT_NAMES.values() if g != "stand"]

    # ── 1. Survival rate bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    sr_m = df_m.groupby("gait_name")["survived"].mean().reindex(
        [GAIT_NAMES[g] for g in sorted(GAIT_NAMES) if GAIT_NAMES[g] != "stand"])
    sr_i = df_i.groupby("gait_name")["survived"].mean().reindex(sr_m.index)
    x = np.arange(len(sr_m))
    w = 0.35
    ax.bar(x - w/2, sr_m * 100, width=w, label="MuJoCo", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, sr_i * 100, width=w, label="Isaac",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(sr_m.index, rotation=30, ha="right")
    ax.set_ylabel("Survival rate (%)"); ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(); ax.set_title("Survival rate — MuJoCo vs Isaac", fontweight="bold")
    fig.savefig(out_dir / "C1_survival_rate.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_dir}/C1_survival_rate.png")

    # ── 2. Tracking error comparison ─────────────────────────────────────────
    metrics = ["mean_vx_error", "mean_vy_error", "mean_wz_error"]
    labels  = ["vx error (m/s)", "vy error (m/s)", "wz error (rad/s)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    gaits = [GAIT_NAMES[g] for g in sorted(GAIT_NAMES) if GAIT_NAMES[g] != "stand"]
    x = np.arange(len(gaits)); w = 0.35
    for ax, metric, label in zip(axes, metrics, labels):
        vm = df_m.groupby("gait_name")[metric].mean().reindex(gaits)
        vi = df_i.groupby("gait_name")[metric].mean().reindex(gaits)
        ax.bar(x - w/2, vm, width=w, label="MuJoCo", color="#3498db", alpha=0.85)
        ax.bar(x + w/2, vi, width=w, label="Isaac",  color="#e74c3c", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(gaits, rotation=30, ha="right")
        ax.set_ylabel(label); ax.legend(fontsize=8)
        ax.set_title(label, fontweight="bold")
    fig.suptitle("Mean tracking errors — MuJoCo vs Isaac", fontweight="bold")
    fig.savefig(out_dir / "C2_tracking_errors.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_dir}/C2_tracking_errors.png")

    # ── 3. Contact accuracy comparison ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    cm = df_m.groupby("gait_name")["mean_contact_acc"].mean().reindex(gaits)
    ci = df_i.groupby("gait_name")["mean_contact_acc"].mean().reindex(gaits)
    ax.bar(x - w/2, cm, width=w, label="MuJoCo", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, ci, width=w, label="Isaac",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(gaits, rotation=30, ha="right")
    ax.set_ylabel("Contact accuracy"); ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(); ax.set_title("Contact accuracy — MuJoCo vs Isaac", fontweight="bold")
    fig.savefig(out_dir / "C3_contact_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_dir}/C3_contact_accuracy.png")

    # ── 4. Torque norm comparison ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    tm = df_m.groupby("gait_name")["mean_torque_norm"].mean().reindex(gaits)
    ti = df_i.groupby("gait_name")["mean_torque_norm"].mean().reindex(gaits)
    ax.bar(x - w/2, tm, width=w, label="MuJoCo", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, ti, width=w, label="Isaac",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(gaits, rotation=30, ha="right")
    ax.set_ylabel("Mean torque norm (N·m)")
    ax.legend(); ax.set_title("Torque effort — MuJoCo vs Isaac", fontweight="bold")
    fig.savefig(out_dir / "C4_torque_effort.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_dir}/C4_torque_effort.png")

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  MuJoCo vs Isaac — key metrics per gait")
    print(f"{'='*65}")
    cols = ["mean_vx_error","mean_vy_error","mean_contact_acc","mean_torque_norm"]
    sm = df_m.groupby("gait_name")[cols].mean().reindex(gaits).round(3)
    si = df_i.groupby("gait_name")[cols].mean().reindex(gaits).round(3)
    sm.columns = [f"MuJoCo_{c}" for c in cols]
    si.columns = [f"Isaac_{c}"  for c in cols]
    combined = pd.concat([sm, si], axis=1)
    # interleave columns
    out_cols = []
    for c in cols:
        out_cols += [f"MuJoCo_{c}", f"Isaac_{c}"]
    print(combined[out_cols].to_string())


def parse_args():
    p = argparse.ArgumentParser(description="Analyse gait sweep results")
    p.add_argument("--csv",        default="results/results_raw.csv",
                   help="Path to MuJoCo results CSV")
    p.add_argument("--isaac-csv",  default=None,
                   help="Path to Isaac results CSV (optional, enables comparison plots)")
    p.add_argument("--out-dir",    default="results/plots",
                   help="Directory for output PNGs (default: results/plots)")
    return p.parse_args()


def main():
    args     = parse_args()
    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)

    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    palette = make_palette()

    print(f"\nOutput directory: {out_dir}\n")

    df, errors = load(csv_path)

    # Single-sim plots
    print("Generating plots...")
    plot_success_heatmaps(df, out_dir, palette)
    plot_tracking_heatmaps(df, out_dir)
    plot_yaw_response(df, out_dir, palette)
    plot_survival_distributions(df, out_dir, palette)
    plot_contact_accuracy(df, out_dir, palette)
    plot_stand_stability(df, out_dir)
    plot_torque_effort(df, out_dir, palette)
    print_summary_tables(df)

    # Comparison plots (if Isaac CSV provided)
    if args.isaac_csv:
        isaac_path = Path(args.isaac_csv)
        if not isaac_path.exists():
            print(f"[warning] Isaac CSV not found: {isaac_path} — skipping comparison")
        else:
            df_i, _ = load(isaac_path)
            print("\nGenerating comparison plots...")
            plot_comparison(df, df_i, out_dir)

    print(f"\nAll plots saved to: {out_dir}\n")


if __name__ == "__main__":
    main()