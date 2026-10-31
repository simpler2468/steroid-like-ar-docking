"""Step 5.5: compare 2AM9 and eligible receptor-ensemble results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr

from ensemble_utils import ROOT, read_csv_rows, write_csv


GROUP_A = {705, 699, 711, 706, 704}
GROUP_B = {2412, 5375, 5379}
FOCUS = ["cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]
COLORS = {"2AM9": "#3b6fb6", "2AMA": "#e67e22", "2AMB": "#999999"}
GROUP_COLORS = {"A_ethynyl": "#2f6fb3", "B_long_ester": "#d9822b", "C_rest": "#4c9b61"}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def group(row_index: int) -> str:
    if row_index in GROUP_A:
        return "A_ethynyl"
    if row_index in GROUP_B:
        return "B_long_ester"
    return "C_rest"


def step44_module():
    path = ROOT / "scripts" / "08_orientation_verification.py"
    spec = importlib.util.spec_from_file_location("step44_for_figures", path)
    mod = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    return mod


def savefig(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def score_data() -> pd.DataFrame:
    old = read(ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv")[["compound_id", "cluster_id", "row_index", "vina_score_kcal_mol"]].copy()
    old = old.rename(columns={"vina_score_kcal_mol": "score_2AM9"})
    new = read(ROOT / "results" / "receptor_ensemble" / "batch_docking" / "2AMA" / "docking_top_poses.csv")[["compound_id", "cluster_id", "row_index", "vina_score_kcal_mol"]].copy()
    new = new.rename(columns={"vina_score_kcal_mol": "score_2AMA"})
    merged = old.merge(new, on=["compound_id", "cluster_id", "row_index"], how="outer", validate="one_to_one")
    merged["group"] = merged["row_index"].astype(int).map(group)
    merged.to_csv(ROOT / "results" / "receptor_ensemble" / "score_matrix.csv", index=False)
    return merged


def figure1(scores: pd.DataFrame, figdir: Path) -> None:
    long = scores.melt(id_vars=["compound_id", "cluster_id", "row_index", "group"], value_vars=["score_2AM9", "score_2AMA"], var_name="receptor", value_name="score")
    long["receptor"] = long["receptor"].str.replace("score_", "", regex=False)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    positions = {(g, r): i for i, (g, r) in enumerate((g, r) for g in ["A_ethynyl", "B_long_ester", "C_rest"] for r in ["2AM9", "2AMA"])}
    for g in ["A_ethynyl", "B_long_ester", "C_rest"]:
        for r in ["2AM9", "2AMA"]:
            vals = long[(long.group == g) & (long.receptor == r)].score.to_numpy()
            x = positions[(g, r)]
            ax.boxplot(vals, positions=[x], widths=0.54, patch_artist=True, boxprops={"facecolor": GROUP_COLORS[g], "alpha": 0.24}, medianprops={"color": "black"}, whiskerprops={"color": "#555555"}, capprops={"color": "#555555"}, flierprops={"marker": "", "alpha": 0})
            jitter = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=28, color=COLORS[r], edgecolor="white", linewidth=0.5, label=r if (g == "A_ethynyl") else None, zorder=3)
    ax.set_xticks([0.5, 2.5, 4.5], ["A: ethynyl\n(n=5)", "B: long ester\n(n=3)", "C: rest\n(n=16)"])
    ax.set_ylabel("Top-pose Vina score (kcal/mol)"); ax.set_title("Score robustness across AR receptor conformations")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, title="Receptor")
    savefig(fig, figdir / "figure1_score_robustness")


def orientation_data() -> pd.DataFrame:
    old = read(ROOT / "results" / "orientation_robustness" / "multi_seed_top_poses.csv")
    old = old[old.compound_id.isin(FOCUS)].copy(); old["receptor"] = "2AM9"; old["angle_deg"] = old["angle_to_TES_deg"]; old["score"] = old["vina_score_kcal_mol"]
    new = read(ROOT / "results" / "receptor_ensemble" / "multiseed_orientation" / "2AMA" / "orientation_top_poses.csv")
    new = new.copy(); new["receptor"] = "2AMA"; new["angle_deg"] = new["c3_c17_angle_to_TES_deg"]; new["score"] = new["vina_score_kcal_mol"]
    return pd.concat([old[["compound_id", "seed", "orientation_class", "receptor", "angle_deg", "score"]], new[["compound_id", "seed", "orientation_class", "receptor", "angle_deg", "score"]]], ignore_index=True)


def figure2(orient: pd.DataFrame, figdir: Path) -> None:
    order = FOCUS
    fig, ax = plt.subplots(figsize=(10, 5.8))
    xbase = np.arange(len(order)) * 2.0
    for i, cid in enumerate(order):
        for j, receptor in enumerate(["2AM9", "2AMA"]):
            sub = orient[(orient.compound_id == cid) & (orient.receptor == receptor)]
            x = xbase[i] + (j - 0.5) * 0.62
            for cls, color in [("consistent_with_native_orientation", "#3b82a0"), ("head_to_tail_reversal_supported", "#c75b39"), ("alternative_orientation", "#8a73a9"), ("ambiguous", "#777777")]:
                q = sub[sub.orientation_class == cls]
                if len(q):
                    ax.scatter(np.full(len(q), x), q.angle_deg, s=30, marker="o" if receptor == "2AM9" else "s", color=color, edgecolor="white", linewidth=0.45, alpha=0.85)
            ax.hlines([60, 120], x - 0.27, x + 0.27, color="#bbbbbb", linewidth=0.8, linestyles="--")
    ax.set_xticks(xbase, [f"{cid.rsplit('_', 1)[1]}" for cid in order]); ax.set_xlabel("Focused compound row index"); ax.set_ylabel("C3→C17 angle to TES (degrees)"); ax.set_ylim(-5, 185); ax.set_title("C17 orientation stability across receptor conformations")
    ax.text(0.01, 0.98, "≤60° native-like; ≥120° reversed-supported", transform=ax.transAxes, va="top", fontsize=9, color="#555555")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    class_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="white", markersize=7, label=label) for c, label in [("#3b82a0", "native-like"), ("#c75b39", "reversed-supported"), ("#8a73a9", "alternative"), ("#777777", "ambiguous")]]
    receptor_handles = [Line2D([0], [0], marker=m, color="black", linestyle="None", markersize=6, label=r) for r, m in [("2AM9", "o"), ("2AMA", "s")]]
    ax.legend(handles=class_handles + receptor_handles, frameon=False, fontsize=8, loc="upper right", ncol=2)
    savefig(fig, figdir / "figure2_orientation_stability")


def figure3(figdir: Path) -> pd.DataFrame:
    rows = []
    for receptor in ["2AM9", "2AMA"]:
        if receptor == "2AM9":
            continue
        data = json.loads((ROOT / "data" / "prepared" / receptor / "preparation_summary.json").read_text())
        for item in data["alignment"]["key_residue_backbone_displacements"]:
            rows.append({"receptor": receptor, "residue": int(item["resseq"]), "displacement_A": float(item["backbone_rmsd_A"])})
    df = pd.DataFrame(rows); df.to_csv(ROOT / "results" / "receptor_ensemble" / "pocket_alignment_metrics.csv", index=False)
    pivot = df.pivot(index="residue", columns="receptor", values="displacement_A").sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 9.0))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=max(0.5, float(np.nanmax(pivot.to_numpy()))))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns); ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index]); ax.set_xlabel("Aligned receptor"); ax.set_ylabel("AR residue number"); ax.set_title("Aligned pocket backbone displacement vs 2AM9")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value): ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > pivot.to_numpy().max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Backbone RMSD / displacement (Å)"); savefig(fig, figdir / "figure3_pocket_alignment")
    return df


def figure4(orient: pd.DataFrame, figdir: Path) -> None:
    mod = step44_module()
    fig = plt.figure(figsize=(16, 7)); axes = [fig.add_subplot(2, 5, i + 1, projection="3d") for i in range(10)]
    rows = [("2AM9", cid) for cid in FOCUS] + [("2AMA", cid) for cid in FOCUS]
    for idx, (receptor, cid) in enumerate(rows):
        ax = axes[idx]
        if receptor == "2AM9":
            sdf = ROOT / "results" / "orientation_robustness" / "docking_poses" / cid / "seed_42" / "pose_01.sdf"
        else:
            sdf = ROOT / "results" / "receptor_ensemble" / "multiseed_orientation" / "2AMA" / "pose_files" / cid / "seed_42" / "pose_01.sdf"
        mol = Chem.RemoveHs(Chem.MolFromMolFile(str(sdf), removeHs=False))
        anchor = mod.find_anchor_atoms(mol); c3, c17 = mod.xyz(mol, anchor["C3"]), mod.xyz(mol, anchor["C17"])
        ax.plot([c3[0], c17[0]], [c3[1], c17[1]], [c3[2], c17[2]], color="#3b6fb6" if receptor == "2AM9" else "#e67e22", linewidth=2.4)
        ax.scatter([c3[0], c17[0]], [c3[1], c17[1]], [c3[2], c17[2]], color=["#2468a6", "#c23b22"], s=35)
        ax.text(c3[0], c3[1], c3[2], "C3", fontsize=8); ax.text(c17[0], c17[1], c17[2], "C17", fontsize=8)
        ax.set_title(f"{receptor} · row {cid.rsplit('_',1)[1]}", fontsize=9); ax.set_axis_off(); ax.view_init(elev=18, azim=-55)
    # Two receptor rows and five matched compounds per row.
    fig.suptitle("Representative C3–C17 vectors from actual top-pose coordinates", y=0.98, fontsize=13); fig.tight_layout(); savefig(fig, figdir / "figure4_representative_poses")


def write_pymol_scripts(figdir: Path) -> None:
    pml = figdir / "pocket_alignment.pml"
    lines = ["reinitialize", f"load {((ROOT/'data/prepared/2AM9/receptor_repaired.pdb').as_posix())}, ar_2am9", f"load {((ROOT/'data/prepared/2AMA/receptor_aligned.pdb').as_posix())}, ar_2ama", "hide everything", "show cartoon, ar_2am9 or ar_2ama", "color gray70, ar_2am9", "color marine, ar_2ama", "select key_pocket, (ar_2am9 or ar_2ama) and resi 705+711+752+877", "show sticks, key_pocket", "label key_pocket and name CA, resn+resi", "set label_size, 12", "orient ar_2am9", "zoom ar_2am9, 5", "save figures/receptor_ensemble/pocket_alignment.pse"]
    pml.write_text("\n".join(lines) + "\n")
    rep = figdir / "representative_poses.pml"
    lines = ["reinitialize", f"load {(ROOT/'data/prepared/2AM9/receptor_repaired.pdb').as_posix()}, ar_2am9", f"load {(ROOT/'data/prepared/2AMA/receptor_aligned.pdb').as_posix()}, ar_2ama", "hide everything", "show cartoon, ar_2am9 or ar_2ama", "color gray70, ar_2am9", "color marine, ar_2ama"]
    for cid in ["cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]:
        p = ROOT / "results" / "receptor_ensemble" / "multiseed_orientation" / "2AMA" / "pose_files" / cid / "seed_42" / "pose_01.sdf"
        lines += [f"load {p.as_posix()}, ama_{cid}", f"show sticks, ama_{cid}", f"color cyan, ama_{cid}"]
    lines += ["select key_pocket, (ar_2am9 or ar_2ama) and resi 705+711+752+877", "show sticks, key_pocket", "orient ar_2am9", "zoom ar_2am9, 5", "save figures/receptor_ensemble/representative_poses.pse"]
    (figdir / "representative_poses.pml").write_text("\n".join(lines) + "\n")


def main() -> None:
    figdir = ROOT / "figures" / "receptor_ensemble"
    scores = score_data(); orient = orientation_data()
    figure1(scores, figdir); figure2(orient, figdir); pocket = figure3(figdir); figure4(orient, figdir); write_pymol_scripts(figdir)
    stats = []
    for g, sub in scores.groupby("group"):
        for receptor in ["2AM9", "2AMA"]:
            vals = sub[f"score_{receptor}"]
            stats.append({"group": g, "receptor": receptor, "n": len(vals), "mean": vals.mean(), "sd": vals.std(ddof=1), "median": vals.median(), "min": vals.min(), "max": vals.max()})
    write_csv(ROOT / "results" / "receptor_ensemble" / "score_group_stats.csv", stats)
    correlations = []
    paired = scores.dropna(subset=["score_2AM9", "score_2AMA"])
    correlations.append({"comparison": "top_score_2AM9_vs_2AMA", "n": len(paired), "spearman_rho": spearmanr(paired.score_2AM9, paired.score_2AMA).statistic, "p_value_descriptive": spearmanr(paired.score_2AM9, paired.score_2AMA).pvalue})
    pd.DataFrame(correlations).to_csv(ROOT / "results" / "receptor_ensemble" / "score_correlations.csv", index=False)
    # H1 membership is fixed before looking at the ensemble scores.
    scores[["compound_id", "row_index", "cluster_id", "group"]].to_csv(ROOT / "results" / "receptor_ensemble" / "h1_group_membership.csv", index=False)
    print(json.dumps({"score_rows": len(scores), "orientation_rows": len(orient), "figures": sorted(x.name for x in figdir.glob("figure*.*")), "pocket_rows": len(pocket)}, indent=2))


if __name__ == "__main__":
    main()

