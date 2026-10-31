"""Step 4.1: descriptive analysis of the 24 Step 3.2 docking-ready ligands.

This script is deliberately analysis-only.  It reads the existing docking and
ligand-preparation outputs, does not import Vina, and never changes any input
structure, PDBQT, docking box, or original workbook.
"""

from __future__ import annotations

import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv"
QC = ROOT / "results" / "docking_2AM9" / "docking_qc.csv"
READY = ROOT / "results" / "ligand_preparation" / "docking_ready.csv"
MANIFEST = ROOT / "results" / "ligand_preparation" / "ligand_manifest.csv"
SOURCE_XLSX = ROOT / "data" / "androgens_clusters_teacher_profile.xlsx"
OUT = ROOT / "results" / "docking_analysis"
FIG = ROOT / "figures" / "docking_analysis"
REPORT = ROOT / "reports" / "docking_score_structure_analysis.md"

CLUSTERS = [3, 176, 246, 416]
COLORS = {3: "#4C78A8", 176: "#F58518", 246: "#54A24B", 416: "#E45756"}
NATIVE_REFERENCE_SCORE = -10.905
HIGH_SIMILARITY = 0.70
LOW_SIMILARITY = 0.30
LARGE_SCORE_DIFFERENCE = 2.0
CLOSE_SCORE_DIFFERENCE = 0.50


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def summary_stats(values: pd.Series | list[float]) -> dict[str, Any]:
    s = pd.Series(values, dtype=float).dropna()
    if len(s) == 0:
        return {"n": 0}
    q1, q3 = s.quantile([0.25, 0.75])
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "sd_sample": float(s.std(ddof=1)) if len(s) > 1 else None,
        "minimum": float(s.min()),
        "maximum": float(s.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "range": float(s.max() - s.min()),
    }


def rdkit_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def nonisomeric_key(mol: Chem.Mol) -> str:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return Chem.MolToSmiles(copy, isomericSmiles=False)


def fp_matrix(mols: list[Chem.Mol], include_chirality: bool) -> np.ndarray:
    generator = GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=include_chirality
    )
    fps = [generator.GetFingerprint(mol) for mol in mols]
    matrix = np.eye(len(fps), dtype=float)
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = float(Chem.DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            matrix[i, j] = matrix[j, i] = sim
    return matrix


def pairwise_records(df: pd.DataFrame, matrix_no: np.ndarray, matrix_chiral: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, j in combinations(range(len(df)), 2):
        a, b = df.iloc[i], df.iloc[j]
        rows.append(
            {
                "compound_id_1": a.compound_id,
                "compound_id_2": b.compound_id,
                "cluster_id_1": int(a.cluster_id),
                "cluster_id_2": int(b.cluster_id),
                "pair_type": "intra-cluster" if a.cluster_id == b.cluster_id else "inter-cluster",
                "same_connectivity": bool(a.connectivity_key == b.connectivity_key),
                "stereo_difference_same_connectivity": bool(
                    a.connectivity_key == b.connectivity_key and a.isomeric_key != b.isomeric_key
                ),
                "similarity_no_chirality": matrix_no[i, j],
                "similarity_with_chirality": matrix_chiral[i, j],
                "score_1_kcal_mol": float(a.vina_score_kcal_mol),
                "score_2_kcal_mol": float(b.vina_score_kcal_mol),
                "abs_score_difference_kcal_mol": abs(
                    float(a.vina_score_kcal_mol) - float(b.vina_score_kcal_mol)
                ),
            }
        )
    return pd.DataFrame(rows)


def group_similarity_stats(pairs: pd.DataFrame, similarity_col: str) -> pd.DataFrame:
    rows = []
    for pair_type in ["intra-cluster", "inter-cluster"]:
        sub = pairs[pairs.pair_type == pair_type]
        stats = summary_stats(sub[similarity_col])
        stats.update({"pair_type": pair_type, "similarity_definition": similarity_col})
        rows.append(stats)
    return pd.DataFrame(rows)


def cluster_similarity_stats(pairs: pd.DataFrame, similarity_col: str) -> pd.DataFrame:
    rows = []
    for cluster in CLUSTERS:
        sub = pairs[
            (pairs.pair_type == "intra-cluster")
            & (pairs.cluster_id_1 == cluster)
            & (pairs.cluster_id_2 == cluster)
        ]
        stats = summary_stats(sub[similarity_col])
        stats.update({"cluster_id": cluster, "similarity_definition": similarity_col})
        rows.append(stats)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_score_figure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    order = [str(c) for c in CLUSTERS]
    plot_df = df.copy()
    plot_df["cluster_label"] = plot_df.cluster_id.astype(str)
    sns.boxplot(
        data=plot_df,
        x="cluster_label",
        y="vina_score_kcal_mol",
        hue="cluster_label",
        order=order,
        hue_order=order,
        palette={str(c): COLORS[c] for c in CLUSTERS},
        legend=False,
        width=0.55,
        fliersize=0,
        saturation=0.85,
        ax=ax,
    )
    rng = np.random.default_rng(20261030)
    for xpos, cluster in enumerate(CLUSTERS):
        vals = plot_df.loc[plot_df.cluster_id == cluster, "vina_score_kcal_mol"]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            xpos + jitter,
            vals,
            s=42,
            color="white",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=4,
        )
    ax.axhline(NATIVE_REFERENCE_SCORE, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(3.45, NATIVE_REFERENCE_SCORE, "native TES redocking reference", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(range(len(CLUSTERS)))
    ax.set_xticklabels([f"Cluster {c}\n(n={int((plot_df.cluster_id == c).sum())})" for c in CLUSTERS])
    ax.set_xlabel("Bioactivity cluster")
    ax.set_ylabel("Top-ranked Vina docking score (kcal/mol)\n(more negative = more favorable model score)")
    ax.set_title("Docking Score Distribution across Four Bioactivity Clusters")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figure1_score_distribution")


def make_descriptor_figure(df: pd.DataFrame) -> None:
    descriptors = [
        ("molecular_weight", "Molecular weight (Da)"),
        ("clogp", "cLogP"),
        ("heavy_atom_count", "Heavy atom count"),
        ("tpsa", "TPSA (Å²)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (col, xlabel) in zip(axes.flat, descriptors):
        for cluster in CLUSTERS:
            sub = df[df.cluster_id == cluster]
            ax.scatter(sub[col], sub.vina_score_kcal_mol, s=42, color=COLORS[cluster], label=f"Cluster {cluster}", alpha=0.9, edgecolor="white", linewidth=0.4)
        rho, _ = spearmanr(df[col], df.vina_score_kcal_mol)
        ax.text(0.04, 0.06, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top Vina score (kcal/mol)")
        ax.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Molecular Physicochemical Properties vs Docking Score", y=1.07)
    save_figure(fig, "figure2_descriptor_score")


def make_heatmap(matrix: np.ndarray, df: pd.DataFrame, stem: str, title: str) -> None:
    labels = [f"C{int(r.cluster_id)}:{int(r.row_index)}" for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Tanimoto similarity"},
    )
    boundaries = np.cumsum([int((df.cluster_id == c).sum()) for c in CLUSTERS])
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="white", linewidth=1.5)
        ax.axvline(boundary, color="white", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Compound (ordered by cluster)")
    ax.set_ylabel("Compound (ordered by cluster)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    save_figure(fig, stem)


def make_pairwise_figure(pairs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.7))
    for pair_type, marker, color in [("intra-cluster", "o", "#1f77b4"), ("inter-cluster", "s", "#ff7f0e")]:
        sub = pairs[pairs.pair_type == pair_type]
        ax.scatter(sub.similarity_no_chirality, sub.abs_score_difference_kcal_mol, s=30, alpha=0.65, marker=marker, color=color, label=pair_type)
    special = pairs[pairs.stereo_difference_same_connectivity]
    if len(special):
        ax.scatter(special.similarity_no_chirality, special.abs_score_difference_kcal_mol, s=75, facecolors="none", edgecolors="#d62728", linewidth=1.4, label="same connectivity, different stereo")
    ax.set_xlabel("Morgan Tanimoto similarity (no chirality)")
    ax.set_ylabel("Absolute top-score difference (kcal/mol)")
    ax.set_title("Molecular Similarity vs Absolute Docking Score Difference")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "figure4_similarity_score_difference")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def pair_label(row: pd.Series) -> str:
    return f"{row.compound_id_1} ↔ {row.compound_id_2}"


def write_report(
    top: pd.DataFrame,
    score_stats: pd.DataFrame,
    cluster_sim: pd.DataFrame,
    pairs: pd.DataFrame,
    reps: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    score_lines = ["| Cluster | n | Mean | Median | SD | Min | Max | Q1 | Q3 | IQR | Range |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in score_stats.iterrows():
        label = str(r.cluster_id)
        score_lines.append(f"| {label} | {int(r.n)} | {fmt(r['mean'])} | {fmt(r['median'])} | {fmt(r['sd_sample'])} | {fmt(r['minimum'])} | {fmt(r['maximum'])} | {fmt(r['q1'])} | {fmt(r['q3'])} | {fmt(r['iqr'])} | {fmt(r['range'])} |")

    sim_lines = ["| Cluster | n pairs | No chirality mean | No chirality median | With chirality mean | With chirality median |", "|---:|---:|---:|---:|---:|---:|"]
    for cluster in CLUSTERS:
        a = cluster_sim[(cluster_sim.cluster_id == cluster) & (cluster_sim.similarity_definition == "similarity_no_chirality")].iloc[0]
        b = cluster_sim[(cluster_sim.cluster_id == cluster) & (cluster_sim.similarity_definition == "similarity_with_chirality")].iloc[0]
        sim_lines.append(f"| {cluster} | {int(a.n)} | {fmt(a['mean'])} | {fmt(a['median'])} | {fmt(b['mean'])} | {fmt(b['median'])} |")

    def example_lines(sub: pd.DataFrame, limit: int = 5, sort_col: str = "abs_score_difference_kcal_mol", ascending: bool = False) -> list[str]:
        if len(sub) == 0:
            return ["None under the stated exploratory threshold."]
        lines = []
        for _, row in sub.sort_values(sort_col, ascending=ascending).head(limit).iterrows():
            lines.append(f"- `{pair_label(row)}`: no-chirality similarity {fmt(row.similarity_no_chirality)}, with-chirality similarity {fmt(row.similarity_with_chirality)}, |Δscore| {fmt(row.abs_score_difference_kcal_mol)} kcal/mol ({row.pair_type}).")
        return lines

    high_div = pairs[(pairs.similarity_no_chirality >= HIGH_SIMILARITY) & (pairs.abs_score_difference_kcal_mol >= LARGE_SCORE_DIFFERENCE)]
    low_close = pairs[(pairs.similarity_no_chirality <= LOW_SIMILARITY) & (pairs.abs_score_difference_kcal_mol <= CLOSE_SCORE_DIFFERENCE)]
    high_close = pairs[(pairs.pair_type == "intra-cluster") & (pairs.similarity_no_chirality >= HIGH_SIMILARITY) & (pairs.abs_score_difference_kcal_mol <= CLOSE_SCORE_DIFFERENCE)]
    stereo_pairs = pairs[pairs.stereo_difference_same_connectivity]

    def rep_line(category: str) -> str:
        sub = reps[reps.category == category]
        if len(sub) == 0:
            return "- None selected."
        out = []
        for _, r in sub.iterrows():
            if pd.isna(r.get("compound_id")):
                out.append(f"- Cluster {int(r.cluster_id)}: none available.")
            else:
                out.append(f"- `{r.compound_id}` (Cluster {int(r.cluster_id)}, row {int(r.row_index)}, score {fmt(r.vina_score_kcal_mol)} kcal/mol).")
        return "\n".join(out)

    cluster_246 = top[top.cluster_id == 246].sort_values("vina_score_kcal_mol")
    rho = summary["structure_score_association"]["spearman_descriptive_only"]
    report = f"""# Step 4.1: Docking Score and Molecular Structure Descriptive Analysis

## Scope and data provenance

本步骤只分析 Step 3.2 已完成的 24 个 `docking_ready` 化合物，每个化合物只保留其 Top-ranked pose 的 Vina score。没有重新运行 docking，没有调用 Vina `score()` 或 `optimize()`，也没有把同一化合物的其他 poses 当作独立样本。23 个 `review_required`/未确认立体化学的记录未纳入本分析。原始 PDB、受体、配体 PDBQT、docking box、Excel 和 Step 1–3 结果均未修改。

输入文件：

- `results/docking_2AM9/docking_top_poses.csv`
- `results/docking_2AM9/docking_qc.csv`
- `results/ligand_preparation/docking_ready.csv`
- `results/ligand_preparation/ligand_manifest.csv`
- `data/androgens_clusters_teacher_profile.xlsx`（只读来源检查）

Step 3.2 protocol context: Vina 1.2.7, receptor `data/prepared/2AM9/receptor_repaired.pdbqt`, box `data/prepared/2AM9/vina_box_repaired.txt`, exhaustiveness 16, seed 42. Testosterone native-ligand redocking mean score `-10.905 kcal/mol` is shown only as a reference line, not as an activity threshold.

QC summary: 24/24 Top Pose records passed the recorded Step 3.2 QC; no Top Pose was outside the docking box, no severe receptor clash (<2 Å) was recorded, and no top-pose structure mismatch was recorded.

## 1. Docking score distribution

{chr(10).join(score_lines)}

Cluster 246 has the widest score distribution by IQR ({fmt(cluster_246.vina_score_kcal_mol.quantile(0.75) - cluster_246.vina_score_kcal_mol.quantile(0.25))} kcal/mol) and range ({fmt(cluster_246.vina_score_kcal_mol.max() - cluster_246.vina_score_kcal_mol.min())} kcal/mol). Cluster 176 has the smallest IQR among clusters with more than two compounds ({fmt(score_stats[score_stats.cluster_id == 176].iloc[0].iqr)} kcal/mol). Cluster 416 has only two compounds, so its narrow spread is not evidence of strong within-cluster consistency. The four cluster means are descriptive and overlap in magnitude; they should not be interpreted as experimental affinity differences.

Figure 1 includes individual compounds, boxplots, and the native TES redocking reference line. More negative Vina scores are more favorable within the scoring model, not higher experimentally validated affinity.

## 2. Molecular physicochemical descriptors

RDKit descriptors were calculated from the original explicit-stereochemistry SMILES for each docking-ready compound: molecular weight, heavy atom count, cLogP, TPSA, HBD, HBA, rotatable bonds, fraction Csp3, formal charge, ring count, and chiral-center count. The complete compound-level table retains `compound_id`, `cluster_id`, `row_index`, `split`, original SMILES, and the original Androgens/PA-term annotation.

The descriptive Spearman coefficients with the Top Vina score were: molecular weight {fmt(rho['molecular_weight']['spearman_rho'])}, heavy atom count {fmt(rho['heavy_atom_count']['spearman_rho'])}, cLogP {fmt(rho['clogp']['spearman_rho'])}, TPSA {fmt(rho['tpsa']['spearman_rho'])}, HBD {fmt(rho['hbd']['spearman_rho'])}, HBA {fmt(rho['hba']['spearman_rho'])}, rotatable bonds {fmt(rho['rotatable_bonds']['spearman_rho'])}, fraction Csp3 {fmt(rho['fraction_csp3']['spearman_rho'])}, and ring count {fmt(rho['ring_count']['spearman_rho'])}. These are exploratory, small-sample associations and are not causal tests. Formal charge was constant across the 24 molecules, so its correlation is undefined.

Figure 2 shows the four requested descriptor–score panels with the same cluster color mapping. The positive rank association for molecular size and rotatable bonds is compatible with a possible scoring/descriptor confounding effect, but it is not sufficient to assign a pharmacological mechanism.

## 3. Molecular structure similarity

Morgan fingerprints used radius 2 and 2048 bits. Both `includeChirality=False` and `includeChirality=True` versions were calculated. The matrices are ordered by Cluster 3, 176, 246, and 416; white lines mark cluster boundaries.

{chr(10).join(sim_lines)}

For all 24 compounds there are {len(pairs)} unordered pairs: {int((pairs.pair_type == 'intra-cluster').sum())} intra-cluster and {int((pairs.pair_type == 'inter-cluster').sum())} inter-cluster pairs. There are {len(stereo_pairs)} same-connectivity/different-stereochemistry pairs under the canonical RDKit keys. These pairs are explicitly marked in `pairwise_similarity_score.csv`; they must not be treated as independent evidence of a cluster effect.

Figures 3 and 3-chiral provide the no-chirality and chirality-aware heatmaps. The current docking-ready subset does not reproduce the previously reported all-record cluster similarity values, because those values were calculated on the original cluster membership rather than this 24-compound subset.

## 4. Structure similarity versus score difference

For each unordered pair, `S_ij` is the Morgan Tanimoto similarity and `D_ij = abs(score_i - score_j)` uses Top-ranked Vina scores. Pairwise observations share compounds and are therefore dependent. The scatter and any descriptive Spearman-style trends must not be presented as 276 independent samples or as formal p-values.

Exploratory thresholds used only to identify cases are: high similarity ≥ {HIGH_SIMILARITY:.2f}, low similarity ≤ {LOW_SIMILARITY:.2f}, large score difference ≥ {LARGE_SCORE_DIFFERENCE:.1f} kcal/mol, and close score difference ≤ {CLOSE_SCORE_DIFFERENCE:.2f} kcal/mol.

High similarity with large score difference: {len(high_div)} pair(s).

{chr(10).join(example_lines(high_div))}

Low similarity with close scores: {len(low_close)} pair(s). The presence of such pairs indicates that similar docking scores can occur for structurally dissimilar ligands under this fixed protocol.

{chr(10).join(example_lines(low_close, limit=5))}

Within-cluster high similarity with close scores: {len(high_close)} pair(s).

{chr(10).join(example_lines(high_close))}

Figure 4 displays the real pairwise points, distinguishes intra- from inter-cluster pairs, and outlines same-connectivity/different-stereo pairs when present. It is a descriptive structure–score relationship, not a causal model.

## 5. Representative compounds for possible Step 4.2

The following categories were selected to cover score extremes, score-central examples, annotated Androgens references, and explicit structure–score cases. Selection is not based only on the most negative score.

### Best-scoring example in each cluster

{rep_line('best_score_by_cluster')}

### Score-central example in each cluster

{rep_line('score_central_by_cluster')}

### Androgens-annotated reference in each cluster

{rep_line('androgens_annotated_reference')}

### Same-cluster, similar-structure and similar-score example

{rep_line('same_cluster_similar_score_consistent')}

### Low-similarity, close-score example

{rep_line('low_similarity_score_convergence')}

Cluster 246 spans Top scores from {fmt(cluster_246.vina_score_kcal_mol.min())} to {fmt(cluster_246.vina_score_kcal_mol.max())} kcal/mol, so it is the clearest candidate for a later representative comparison of more- and less-favorable model scores. This does not establish that Cluster 246 is biologically more heterogeneous.

## 6. Limitations

- Vina score is a model score, not experimental binding free energy or AR pharmacological activity.
- The analysis uses 24 compounds and Cluster 416 contains only two compounds.
- Only one repaired AR receptor structure and one Step 3.2 seed were used for the candidate batch.
- Pairwise observations are non-independent because each compound appears in multiple pairs.
- Teacher-derived bioactivity clustering is not a direct experimental AR activity label.
- The previous stereochemistry filtering and candidate selection define the current subset and may affect both chemical diversity and interpretation.
- Descriptors and scores are confounded by molecular size, flexibility, and other scoring-function preferences. No post hoc outlier deletion was performed.
- No protein–ligand interaction analysis or new docking was performed in Step 4.1.

## 7. Outputs and next step

Tables are in `results/docking_analysis/`. Figures are in `figures/docking_analysis/` as 300 dpi PNG plus SVG and PDF. The reproducible analysis and independent validation scripts are `scripts/05_analyze_docking_scores.py` and `scripts/05_validate_score_analysis.py`.

The outputs support choosing a small, explicitly documented set of representative and contrasting compounds for a possible Step 4.2 pose/interaction analysis. Step 4.2 was not run automatically.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def choose_representatives(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(record: pd.Series, category: str, reason: str, related: str = "") -> None:
        rows.append({
            "category": category,
            "selection_reason": reason,
            "related_compounds": related,
            **{"compound_id": record.get("compound_id", record.name), **{k: record.get(k) for k in ["cluster_id", "row_index", "split", "original_smiles", "vina_score_kcal_mol", "molecular_weight", "heavy_atom_count", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "fraction_csp3", "formal_charge", "ring_count", "has_androgens_annotation", "pa_terms_list", "connectivity_key"]}},
        })

    for cluster in CLUSTERS:
        sub = df[df.cluster_id == cluster].copy()
        add(sub.loc[sub.vina_score_kcal_mol.idxmin()], "best_score_by_cluster", "Most negative top-ranked Vina score within cluster; retained as a score-extreme example.")
        median_score = sub.vina_score_kcal_mol.median()
        central = sub.iloc[(sub.vina_score_kcal_mol - median_score).abs().argsort().iloc[0]]
        add(central, "score_central_by_cluster", "Top score closest to the cluster median; represents a typical score rather than an extreme.")
        if sub.has_androgens_annotation.astype(bool).any():
            andr = sub[sub.has_androgens_annotation.astype(bool)].sort_values("vina_score_kcal_mol").iloc[0]
            add(andr, "androgens_annotated_reference", "Best-scoring docking-ready compound in this cluster with the original Androgens annotation.")
        else:
            rows.append({"category": "androgens_annotated_reference", "cluster_id": cluster, "selection_reason": "No Androgens-annotated docking-ready compound in this cluster.", "compound_id": None})

    high_close = pairs[(pairs.pair_type == "intra-cluster") & (pairs.similarity_no_chirality >= HIGH_SIMILARITY) & (pairs.abs_score_difference_kcal_mol <= CLOSE_SCORE_DIFFERENCE)].sort_values(["similarity_no_chirality", "abs_score_difference_kcal_mol"], ascending=[False, True])
    if len(high_close):
        for _, pair in high_close.head(2).iterrows():
            for cid in [pair.compound_id_1, pair.compound_id_2]:
                add(df.set_index("compound_id").loc[cid], "same_cluster_similar_score_consistent", f"Intra-cluster pair with similarity ≥ {HIGH_SIMILARITY:.2f} and score difference ≤ {CLOSE_SCORE_DIFFERENCE:.2f} kcal/mol.", f"{pair.compound_id_1}; {pair.compound_id_2}")

    high_div = pairs[(pairs.similarity_no_chirality >= HIGH_SIMILARITY) & (pairs.abs_score_difference_kcal_mol >= LARGE_SCORE_DIFFERENCE)].sort_values(["abs_score_difference_kcal_mol", "similarity_no_chirality"], ascending=[False, False])
    if len(high_div):
        for _, pair in high_div.head(2).iterrows():
            for cid in [pair.compound_id_1, pair.compound_id_2]:
                add(df.set_index("compound_id").loc[cid], "high_similarity_score_divergence", f"High structural similarity (≥ {HIGH_SIMILARITY:.2f}) but large score difference (≥ {LARGE_SCORE_DIFFERENCE:.1f} kcal/mol); useful for Step 4.2 comparison.", f"{pair.compound_id_1}; {pair.compound_id_2}")

    low_close = pairs[(pairs.similarity_no_chirality <= LOW_SIMILARITY) & (pairs.abs_score_difference_kcal_mol <= CLOSE_SCORE_DIFFERENCE)].sort_values(["similarity_no_chirality", "abs_score_difference_kcal_mol"], ascending=[True, True])
    if len(low_close):
        for _, pair in low_close.head(2).iterrows():
            for cid in [pair.compound_id_1, pair.compound_id_2]:
                add(df.set_index("compound_id").loc[cid], "low_similarity_score_convergence", f"Low structural similarity (≤ {LOW_SIMILARITY:.2f}) but close score difference (≤ {CLOSE_SCORE_DIFFERENCE:.2f} kcal/mol); tests whether similar scores can arise from different structures.", f"{pair.compound_id_1}; {pair.compound_id_2}")

    return pd.DataFrame(rows).drop_duplicates(subset=["category", "compound_id", "cluster_id", "related_compounds"], keep="first")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    top = pd.read_csv(TOP)
    ready = pd.read_csv(READY)
    qc = pd.read_csv(QC)
    manifest = pd.read_csv(MANIFEST)

    required_top = {"compound_id", "cluster_id", "row_index", "split", "original_smiles", "vina_score_kcal_mol", "docking_status", "pose_rank"}
    missing = required_top - set(top.columns)
    if missing:
        raise ValueError(f"Missing top-pose columns: {sorted(missing)}")
    top = top[top.pose_rank == 1].copy()
    ready = ready[ready.docking_status == "docking_ready"].copy()
    if len(top) != 24 or len(ready) != 24 or top.compound_id.nunique() != 24:
        raise ValueError(f"Expected 24 unique top poses and 24 ready records, got {len(top)}, {len(ready)}")
    if set(top.compound_id) != set(ready.compound_id):
        raise ValueError("Top-pose and docking-ready compound sets do not match")
    if not (top.docking_status == "success").all():
        raise ValueError("At least one top pose is not marked successful")

    top = top.merge(ready[["compound_id", "canonical_isomeric_smiles", "canonical_nonisomeric_smiles", "connectivity_id", "resolution_status", "step3_1_status"]], on="compound_id", how="left", validate="one_to_one")
    top = top.merge(manifest[["compound_id", "has_androgens_annotation", "PA_Terms_List", "stereochemistry_status"]], on="compound_id", how="left", validate="one_to_one")
    top["mol"] = top.original_smiles.map(rdkit_mol)
    top["isomeric_key"] = top.mol.map(lambda m: Chem.MolToSmiles(m, isomericSmiles=True))
    top["connectivity_key"] = top.mol.map(nonisomeric_key)
    top["pa_terms_list"] = top["PA_Terms_List"].fillna("")
    top["has_androgens_annotation"] = top["has_androgens_annotation"].fillna(0).astype(bool)

    top["molecular_weight"] = top.mol.map(Descriptors.MolWt)
    top["heavy_atom_count"] = top.mol.map(rdMolDescriptors.CalcNumHeavyAtoms)
    top["clogp"] = top.mol.map(Crippen.MolLogP)
    top["tpsa"] = top.mol.map(rdMolDescriptors.CalcTPSA)
    top["hbd"] = top.mol.map(Lipinski.NumHDonors)
    top["hba"] = top.mol.map(Lipinski.NumHAcceptors)
    top["rotatable_bonds"] = top.mol.map(Lipinski.NumRotatableBonds)
    top["fraction_csp3"] = top.mol.map(rdMolDescriptors.CalcFractionCSP3)
    top["formal_charge"] = top.mol.map(Chem.GetFormalCharge)
    top["ring_count"] = top.mol.map(rdMolDescriptors.CalcNumRings)
    top["chiral_center_count"] = top.mol.map(lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)))
    top["source_smiles"] = top.original_smiles
    top = top.sort_values(["cluster_id", "row_index"]).reset_index(drop=True)

    # Basic source checks, including a read-only check against the original workbook.
    excel_check = {"read": False, "header_row_zero_based": None, "columns": [], "smiles_column": None, "row_index_column": None, "rows_found": None, "smiles_match_count": None, "androgens_annotation_match_count": None, "error": None}
    try:
        raw_source = pd.read_excel(SOURCE_XLSX, header=None)
        header_candidates = [i for i, row in raw_source.iterrows() if any(str(v).strip().lower() == "smiles" for v in row.tolist())]
        header_row = header_candidates[0] if header_candidates else 0
        source = pd.read_excel(SOURCE_XLSX, header=header_row)
        excel_check["read"] = True
        excel_check["header_row_zero_based"] = int(header_row)
        excel_check["columns"] = [str(c) for c in source.columns]
        smiles_col = next((c for c in source.columns if "smiles" in str(c).lower()), None)
        row_col = next((c for c in source.columns if str(c).strip().lower() == "row_index"), None)
        excel_check["smiles_column"] = str(smiles_col) if smiles_col is not None else None
        excel_check["row_index_column"] = str(row_col) if row_col is not None else None
        if smiles_col is not None and row_col is not None:
            source = source.copy()
            source[row_col] = pd.to_numeric(source[row_col], errors="coerce")
            joined = top[["row_index", "original_smiles"]].merge(source[[row_col, smiles_col]], left_on="row_index", right_on=row_col, how="left", validate="one_to_one")
            source_smiles = joined[smiles_col].astype(str).tolist()
            excel_check["rows_found"] = int(joined[smiles_col].notna().sum())
            excel_check["smiles_match_count"] = int(sum(a == b for a, b in zip(source_smiles, top.original_smiles.astype(str))))
            if "has_androgens_annotation" in source.columns:
                ann = top[["row_index", "has_androgens_annotation"]].merge(source[[row_col, "has_androgens_annotation"]], left_on="row_index", right_on=row_col, how="left", validate="one_to_one")
                excel_check["androgens_annotation_match_count"] = int((ann["has_androgens_annotation_x"].astype(bool) == ann["has_androgens_annotation_y"].astype(bool)).sum())
    except Exception as exc:
        excel_check["error"] = str(exc)

    descriptor_cols = ["compound_id", "cluster_id", "row_index", "split", "source_smiles", "original_smiles", "canonical_isomeric_smiles", "canonical_nonisomeric_smiles", "connectivity_id", "resolution_status", "step3_1_status", "vina_score_kcal_mol", "molecular_weight", "heavy_atom_count", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "fraction_csp3", "formal_charge", "ring_count", "chiral_center_count", "has_androgens_annotation", "pa_terms_list", "stereochemistry_status", "isomeric_key", "connectivity_key"]
    descriptors = top[descriptor_cols].copy()
    write_csv(descriptors, OUT / "compound_descriptors.csv")

    score_stats = []
    for cluster in CLUSTERS:
        stats = summary_stats(top.loc[top.cluster_id == cluster, "vina_score_kcal_mol"])
        stats["cluster_id"] = cluster
        score_stats.append(stats)
    overall = summary_stats(top.vina_score_kcal_mol)
    overall["cluster_id"] = "overall"
    score_stats.append(overall)
    score_stats_df = pd.DataFrame(score_stats)[["cluster_id", "n", "mean", "median", "sd_sample", "minimum", "maximum", "q1", "q3", "iqr", "range"]]
    write_csv(score_stats_df, OUT / "cluster_score_statistics.csv")

    no_chiral = fp_matrix(top.mol.tolist(), include_chirality=False)
    chiral = fp_matrix(top.mol.tolist(), include_chirality=True)
    pd.DataFrame(no_chiral, index=top.compound_id, columns=top.compound_id).to_csv(OUT / "similarity_matrix_no_chirality.csv", encoding="utf-8-sig")
    pd.DataFrame(chiral, index=top.compound_id, columns=top.compound_id).to_csv(OUT / "similarity_matrix_with_chirality.csv", encoding="utf-8-sig")

    pairs = pairwise_records(top, no_chiral, chiral)
    write_csv(pairs, OUT / "pairwise_similarity_score.csv")
    cluster_sim = pd.concat([cluster_similarity_stats(pairs, "similarity_no_chirality"), cluster_similarity_stats(pairs, "similarity_with_chirality")], ignore_index=True)
    write_csv(cluster_sim, OUT / "cluster_similarity_statistics.csv")

    make_score_figure(top)
    make_descriptor_figure(top)
    make_heatmap(no_chiral, top, "figure3_similarity_heatmap", "Molecular Similarity Heatmap of Docking-ready Compounds (Morgan, no chirality)")
    make_heatmap(chiral, top, "figure3_similarity_heatmap_chiral", "Molecular Similarity Heatmap of Docking-ready Compounds (Morgan, with chirality)")
    make_pairwise_figure(pairs)

    reps = choose_representatives(top, pairs)
    write_csv(reps, OUT / "representative_compounds.csv")

    correlations = {}
    for col in ["molecular_weight", "heavy_atom_count", "clogp", "tpsa", "hbd", "hba", "rotatable_bonds", "fraction_csp3", "formal_charge", "ring_count"]:
        if top[col].nunique(dropna=True) < 2:
            correlations[col] = {"spearman_rho": None, "n": int(len(top)), "note": "constant descriptor"}
            continue
        rho, _ = spearmanr(top[col], top.vina_score_kcal_mol)
        correlations[col] = {"spearman_rho": float(rho), "n": int(len(top))}

    same_stereo = pairs[pairs.stereo_difference_same_connectivity]
    high_div = pairs[(pairs.similarity_no_chirality >= HIGH_SIMILARITY) & (pairs.abs_score_difference_kcal_mol >= LARGE_SCORE_DIFFERENCE)]
    low_close = pairs[(pairs.similarity_no_chirality <= LOW_SIMILARITY) & (pairs.abs_score_difference_kcal_mol <= CLOSE_SCORE_DIFFERENCE)]
    high_close = pairs[(pairs.pair_type == "intra-cluster") & (pairs.similarity_no_chirality >= HIGH_SIMILARITY) & (pairs.abs_score_difference_kcal_mol <= CLOSE_SCORE_DIFFERENCE)]

    summary = {
        "analysis_scope": {"docking_ready_count": int(len(top)), "cluster_counts": {str(c): int((top.cluster_id == c).sum()) for c in CLUSTERS}, "excluded_review_required_count": int(len(pd.read_csv(ROOT / "results" / "ligand_preparation" / "ligand_manifest.csv")) - len(top)), "only_top_pose_per_compound": True},
        "inputs": {"top_poses": str(TOP.relative_to(ROOT)), "docking_ready": str(READY.relative_to(ROOT)), "ligand_manifest": str(MANIFEST.relative_to(ROOT)), "source_excel": str(SOURCE_XLSX.relative_to(ROOT)), "qc": str(QC.relative_to(ROOT))},
        "versions": {"python": sys.version.split()[0], "rdkit": rdBase.rdkitVersion, "pandas": pd.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
        "protocol_context": {"docking_protocol_not_rerun": True, "step3_2_seed": 42, "native_testosterone_redocking_reference_score_kcal_mol": NATIVE_REFERENCE_SCORE},
        "excel_source_check": excel_check,
        "score_statistics": {str(r.cluster_id): {k: jsonable(v) for k, v in r.items() if k != "cluster_id"} for _, r in score_stats_df.iterrows()},
        "similarity": {"fingerprint": {"type": "Morgan", "radius": 2, "fp_size": 2048, "no_chirality": True, "with_chirality": True}, "pair_count_total": int(len(pairs)), "pair_count_intra": int((pairs.pair_type == "intra-cluster").sum()), "pair_count_inter": int((pairs.pair_type == "inter-cluster").sum()), "same_connectivity_different_stereo_pair_count": int(len(same_stereo)), "cluster_statistics": jsonable(cluster_sim.to_dict(orient="records"))},
        "structure_score_association": {"spearman_descriptive_only": correlations, "pairwise_dependence_warning": True, "thresholds": {"high_similarity": HIGH_SIMILARITY, "low_similarity": LOW_SIMILARITY, "large_score_difference_kcal_mol": LARGE_SCORE_DIFFERENCE, "close_score_difference_kcal_mol": CLOSE_SCORE_DIFFERENCE}, "high_similarity_large_difference_pair_count": int(len(high_div)), "low_similarity_close_score_pair_count": int(len(low_close)), "intra_cluster_high_similarity_close_score_pair_count": int(len(high_close))},
        "qc_summary": {"top_pose_qc_rows": int(len(qc)), "top_pose_qc_all_pass": bool(((qc.input_ligand_vina_load_ok == True) & (qc.top_pose_outside_box_atoms == 0) & (qc.receptor_heavy_clashes_lt_2A == 0) & (qc.top_pose_stereo_connectivity_match == True)).all())},
        "representative_categories": reps.category.value_counts(dropna=False).to_dict(),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(top, score_stats_df, cluster_sim, pairs, reps, summary)
    print(json.dumps({"status": "success", "n": len(top), "pair_count": len(pairs), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
