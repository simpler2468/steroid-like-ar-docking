#!/usr/bin/env python3
"""Create Step 4.4 plots and analysis tables from orientation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ROB = ROOT / "results" / "orientation_robustness"
FIG = ROOT / "figures" / "orientation_robustness"
FIG.mkdir(parents=True, exist_ok=True)
ORDER = ["cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]
SHORT = {"cluster_416_row_5373": "5373", "cluster_176_row_705": "705", "cluster_176_row_5378": "5378", "cluster_3_row_2412": "2412", "cluster_246_row_5379": "5379"}
CLASS_COLORS = {"consistent_with_native_orientation": "#1f77b4", "head_to_tail_reversal_supported": "#d62728", "alternative_orientation": "#ff7f0e", "ambiguous": "#7f7f7f"}
CLASS_LABELS = {"consistent_with_native_orientation": "Native-like", "head_to_tail_reversal_supported": "Reversed-supported", "alternative_orientation": "Alternative", "ambiguous": "Ambiguous"}


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    top = pd.read_csv(ROB / "multi_seed_top_poses.csv")
    allp = pd.read_csv(ROB / "all_pose_orientation.csv")
    if len(top) != 50 or set(top.compound_id) != set(ORDER):
        raise ValueError(f"Expected 50 top poses for five compounds, found {len(top)}")

    # Figure 2: top-pose angle distribution.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    rng = np.random.default_rng(44)
    for i, cid in enumerate(ORDER):
        sub = top[top.compound_id == cid]
        jitter = rng.uniform(-0.16, 0.16, len(sub))
        for cls, g in sub.groupby("orientation_class"):
            idx = g.index.to_numpy()
            ax.scatter(np.full(len(g), i) + jitter[[sub.index.get_loc(k) for k in idx]], g.angle_to_TES_deg, s=52, alpha=0.9, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls] if i == 0 else None, edgecolor="white", linewidth=0.4)
        frac = (sub.orientation_class == "head_to_tail_reversal_supported").mean()
        ax.text(i, 184, f"rev {frac:.0%}", ha="center", va="bottom", fontsize=9)
    ax.axhline(60, color="0.55", lw=1, ls="--", label="60° native-like screen")
    ax.axhline(120, color="0.35", lw=1, ls=":", label="120° reversal screen")
    ax.set_xticks(range(len(ORDER)), [SHORT[x] for x in ORDER])
    ax.set_xlabel("Compound row")
    ax.set_ylabel("C3→C17 angle relative to TES (degree)")
    ax.set_ylim(0, 190)
    ax.set_title("Figure 2. Multi-seed top-pose orientation distribution")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c], markersize=8, label=CLASS_LABELS[c]) for c in CLASS_COLORS]
    handles += [Line2D([0], [0], color="0.55", ls="--", label="60° native-like screen"), Line2D([0], [0], color="0.35", ls=":", label="120° reversal screen")]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "multi_seed_orientation")

    # Figure 3: score distribution with seed 42 marked.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    vals = [top.loc[top.compound_id == cid, "vina_score_kcal_mol"].to_numpy() for cid in ORDER]
    bp = ax.boxplot(vals, positions=np.arange(len(ORDER)), widths=0.48, patch_artist=True, showfliers=False, medianprops={"color": "black", "lw": 1.4})
    for patch in bp["boxes"]:
        patch.set_facecolor("#d9e6f2")
        patch.set_alpha(0.9)
    for i, cid in enumerate(ORDER):
        sub = top[top.compound_id == cid]
        ax.scatter(np.full(len(sub), i) + rng.uniform(-0.14, 0.14, len(sub)), sub.vina_score_kcal_mol, color="#3f3f3f", s=36, alpha=0.8, zorder=3)
        s42 = sub[sub.seed == 42]
        ax.scatter([i], s42.vina_score_kcal_mol, marker="*", s=150, color="#b22222", edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_xticks(range(len(ORDER)), [SHORT[x] for x in ORDER])
    ax.set_xlabel("Compound row")
    ax.set_ylabel("Top-pose Vina score (kcal/mol)")
    ax.set_title("Figure 3. Multi-seed top-ranked Vina scores (★ seed 42)")
    save(fig, "multi_seed_scores")

    # Figure 4: all returned poses, not independent observations.
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for cls, g in allp.groupby("orientation_class"):
        ax.scatter(g.angle_to_TES_deg, g.vina_score_kcal_mol, s=18, alpha=0.42, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], edgecolors="none")
    label_offsets = {"cluster_416_row_5373": (8, 10), "cluster_176_row_705": (8, -10), "cluster_176_row_5378": (-50, 10), "cluster_3_row_2412": (8, -18), "cluster_246_row_5379": (8, 18)}
    for cid in ORDER:
        sub = allp[allp.compound_id == cid]
        mx, my = sub.angle_to_TES_deg.mean(), sub.vina_score_kcal_mol.mean()
        ax.scatter(mx, my, marker="x", s=65, color="black", linewidth=1.2)
        ax.annotate(SHORT[cid], (mx, my), xytext=label_offsets[cid], textcoords="offset points", fontsize=9, va="center", bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7})
    ax.set_xlabel("C3→C17 angle relative to TES (degree)")
    ax.set_ylabel("Vina score (kcal/mol)")
    ax.set_title("Figure 4. Orientation versus Vina score across all returned poses")
    ax.legend(frameon=False)
    ax.text(0.01, 0.01, "Points from the same docking run are not independent observations.", transform=ax.transAxes, fontsize=8, color="0.35")
    save(fig, "orientation_vs_score")

    # Summary by compound and orientation class for the report.
    class_scores = allp.groupby(["compound_id", "orientation_class"]).agg(pose_count=("vina_score_kcal_mol", "size"), best_score=("vina_score_kcal_mol", "min"), median_score=("vina_score_kcal_mol", "median")).reset_index()
    class_scores.to_csv(ROB / "pose_class_score_summary.csv", index=False)
    top_summary = top.groupby("compound_id").agg(angle_mean_deg=("angle_to_TES_deg", "mean"), angle_sd_deg=("angle_to_TES_deg", "std"), core_raw_rmsd_mean_A=("core_raw_rmsd_TES_A", "mean"), core_centroid_shift_mean_A=("core_centroid_displacement_TES_A", "mean"), swap_margin_mean_A=("swap_margin_TES_A", "mean"), c17_min_distance_mean_A=("C17_min_receptor_distance_A", "mean"), short_contact_runs=("C17_short_contacts_lt_2_5", lambda x: int((x > 0).sum()))).reset_index()
    top_summary.to_csv(ROB / "top_pose_geometry_summary.csv", index=False)
    print(json.dumps({"status": "success", "top_rows": len(top), "all_pose_rows": len(allp), "figures": ["multi_seed_orientation", "multi_seed_scores", "orientation_vs_score"], "class_summary_rows": len(class_scores)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
