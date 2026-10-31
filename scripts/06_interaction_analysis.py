"""Generate Step 4.2 interaction matrices, PyMOL overlays, figures and report."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pose_analysis"
FIG = ROOT / "figures" / "pose_analysis"
STRUCT = OUT / "structures"
RECEPTOR = ROOT / "data" / "prepared" / "2AM9" / "receptor_repaired.pdb"
REPORT = ROOT / "reports" / "binding_pose_interaction_analysis.md"
CLUSTERS = [3, 176, 246, 416]
COLORS = {3: "#4C78A8", 176: "#F58518", 246: "#54A24B", 416: "#E45756"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pose = pd.read_csv(OUT / "pose_descriptors.csv")
    contacts = pd.read_csv(OUT / "ligand_residue_contacts.csv")
    fp = pd.read_csv(OUT / "interaction_fingerprints.csv")
    summary = json.loads((OUT / "pose_analysis_summary.json").read_text(encoding="utf-8"))
    pose = pose.sort_values(["cluster_id", "row_index"]).reset_index(drop=True)
    return pose, contacts, fp, summary


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_interaction_heatmap(pose: pd.DataFrame, fp: pd.DataFrame) -> list[str]:
    # Keep all key-residue columns and the most frequent non-key contact columns.
    bit_cols = [c for c in fp.columns if c != "compound_id"]
    key_cols = [c for c in bit_cols if any(key in c for key in ["ASN705", "GLN711", "ARG752", "THR877"])]
    other_cols = [c for c in bit_cols if c not in key_cols]
    other_cols = sorted(other_cols, key=lambda c: (-int(fp[c].sum()), c))[:16]
    columns = key_cols + other_cols
    matrix = fp.set_index("compound_id").loc[pose.compound_id, columns]
    pretty = [c.replace("A:", "").replace("__", "\n").replace("hydrogen_bond_candidate", "H-bond cand.").replace("polar_contact", "polar").replace("hydrophobic_contact", "hydrophobic").replace("close_vdw_contact", "vdW short").replace("pi_contact_candidate", "π cand.").replace("halogen_contact_candidate", "halogen cand.").replace("steric_clash", "clash") for c in columns]
    fig, ax = plt.subplots(figsize=(max(11, 0.55 * len(columns)), 10))
    sns.heatmap(matrix, ax=ax, cmap=sns.color_palette(["#F3F3F3", "#2166AC"], as_cmap=True), vmin=0, vmax=1, cbar_kws={"label": "Candidate interaction present"}, xticklabels=pretty, yticklabels=[f"C{int(r.cluster_id)}:{int(r.row_index)}" for _, r in pose.iterrows()], linewidths=0.3, linecolor="white")
    boundaries = np.cumsum([(pose.cluster_id == c).sum() for c in CLUSTERS])
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="#222222", linewidth=1.2)
    ax.set_title("Interaction Fingerprint Heatmap of Top-ranked Poses")
    ax.set_xlabel("Residue and interaction candidate")
    ax.set_ylabel("Compound")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    savefig(fig, "figure6_interaction_heatmap")
    return columns


def make_family_summary(pose: pd.DataFrame, fp: pd.DataFrame) -> None:
    exclude = {"compound_id", "cluster_id", "row_index", "split", "vina_score_kcal_mol", "pose_sdf", "key_residue_min_distances", "core_match_status"}
    feature_cols = [c for c in pose.columns if c not in exclude and (c.endswith("_A") or c.endswith("_count") or "__" in c)]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(pose[c])]
    X = pose[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(pose[feature_cols].median(numeric_only=True)).fillna(0.0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster in CLUSTERS:
        sub = pose.cluster_id == cluster
        ax.scatter(coords[sub, 0], coords[sub, 1], s=70, color=COLORS[cluster], label=f"Cluster {cluster}", edgecolor="white", linewidth=0.7)
    for i, row in pose.iterrows():
        ax.text(coords[i, 0] + 0.04, coords[i, 1] + 0.04, f"{int(row.row_index)}\nF{int(row.pose_family)}", fontsize=7)
    ax.set_xlabel(f"Pose descriptor PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"Pose descriptor PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Exploratory Pose-family Summary")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, "figure8_pose_family_summary")


def write_pml(path: Path, ligand_ids: list[str], output_png: Path, title: str, width: int = 2000, height: int = 1600, ligand_colors: dict[str, str] | None = None) -> None:
    ligand_colors = ligand_colors or {}
    lines = [
        "reinitialize",
        "bg_color white",
        "set ray_opaque_background, on",
        "set antialias, 2",
        "set cartoon_fancy_helices, 1",
        "set cartoon_transparency, 20",
        "set stick_radius, 0.18",
        "load data/prepared/2AM9/receptor_repaired.pdb, receptor",
        "hide everything, receptor",
        "show cartoon, receptor",
        "color gray80, receptor",
        "load results/pose_analysis/structures/TES_crystal.pdb, native_TES",
        "show sticks, native_TES",
        "color yellow, native_TES",
        "set stick_radius, 0.22, native_TES",
        "select pocket, byres (receptor within 5.0 of native_TES)",
        "show sticks, pocket",
        "color gray50, pocket",
        "select key_residues, receptor and resi 705+711+752+877",
        "show sticks, key_residues",
        "color orange, key_residues",
        "label key_residues and name CA, resn+resi",
        "set label_size, 18",
    ]
    for compound_id in ligand_ids:
        obj = "lig_" + compound_id.replace("-", "_")
        lines.append(f"load results/pose_analysis/structures/{compound_id}.pdb, {obj}")
        lines.append(f"show sticks, {obj}")
        lines.append(f"set stick_radius, 0.20, {obj}")
        color = ligand_colors.get(compound_id, "cyan")
        lines.append(f"color {color}, {obj}")
    lines += [
        "orient native_TES",
        "zoom native_TES, 8",
        f"png {output_png.as_posix()}, width={width}, height={height}, dpi=300, ray=1",
        "quit",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pymol(pml: Path) -> None:
    pymol = shutil.which("pymol")
    if pymol is None:
        raise RuntimeError("PyMOL executable not found in the active environment")
    subprocess.run([pymol, "-cq", str(pml)], cwd=ROOT, check=True, capture_output=True, text=True)


def compose_cluster_overlays(cluster_paths: dict[int, Path], out_path: Path) -> None:
    images = [Image.open(cluster_paths[c]).convert("RGB") for c in CLUSTERS]
    target_w = max(im.width for im in images)
    target_h = max(im.height for im in images)
    canvas = Image.new("RGB", (2 * target_w, 2 * target_h), "white")
    for idx, (cluster, im) in enumerate(zip(CLUSTERS, images)):
        x, y = (idx % 2) * target_w, (idx // 2) * target_h
        canvas.paste(im, (x, y))
    canvas.save(out_path, dpi=(200, 200))


def make_overlays(pose: pd.DataFrame) -> dict[str, Any]:
    FIG.mkdir(parents=True, exist_ok=True)
    pml_dir = OUT / "pymol_scripts"
    pml_dir.mkdir(parents=True, exist_ok=True)
    all_ids = pose.compound_id.tolist()
    color_map = {row.compound_id: {3: "blue", 176: "orange", 246: "green", 416: "red"}[int(row.cluster_id)] for _, row in pose.iterrows()}
    all_png = FIG / "figure5_all_pose_overlay.png"
    pml = pml_dir / "figure5_all_pose_overlay.pml"
    write_pml(pml, all_ids, all_png, "All top poses", ligand_colors=color_map)
    run_pymol(pml)

    cluster_paths = {}
    for cluster in CLUSTERS:
        ids = pose[pose.cluster_id == cluster].compound_id.tolist()
        out = FIG / f"figure5_cluster_{cluster}_overlay.png"
        pml = pml_dir / f"figure5_cluster_{cluster}_overlay.pml"
        write_pml(pml, ids, out, f"Cluster {cluster}", ligand_colors={cid: {3: "blue", 176: "orange", 246: "green", 416: "red"}[cluster] for cid in ids})
        run_pymol(pml)
        cluster_paths[cluster] = out
    compose_cluster_overlays(cluster_paths, FIG / "figure5_cluster_overlays.png")

    cases = {
        "figure7_case_stereo_pair_5378_5380": ["cluster_176_row_5378", "cluster_176_row_5380"],
        "figure7_case_stereo_pair_5356_5359": ["cluster_246_row_5356", "cluster_246_row_5359"],
        "figure7_case_cluster3_5354_vs_2412": ["cluster_3_row_5354", "cluster_3_row_2412"],
        "figure7_case_cluster246_reps": ["cluster_246_row_5372", "cluster_246_row_704"],
    }
    case_outputs = {}
    for stem, ids in cases.items():
        out = FIG / f"{stem}.png"
        pml = pml_dir / f"{stem}.pml"
        write_pml(pml, ids, out, stem, ligand_colors={ids[0]: "cyan", ids[1]: "magenta"})
        run_pymol(pml)
        case_outputs[stem] = str(out.relative_to(ROOT))
    return {"all": str(all_png.relative_to(ROOT)), "cluster": {str(k): str(v.relative_to(ROOT)) for k, v in cluster_paths.items()}, "cases": case_outputs}


def write_report(pose: pd.DataFrame, contacts: pd.DataFrame, summary: dict, figure_outputs: dict, heatmap_columns: list[str]) -> None:
    interaction_counts = summary["interaction_type_counts"]
    family_counts = pose.groupby("pose_family").size().to_dict()
    family_cluster = pose.groupby(["pose_family", "cluster_id"]).size().unstack(fill_value=0)
    key_rows = contacts[contacts.is_key_residue].copy()
    key_summary = key_rows.groupby("residue_key").size().sort_values(ascending=False).to_dict()
    stereo = pose[pose.compound_id.isin(["cluster_176_row_5378", "cluster_176_row_5380", "cluster_246_row_5356", "cluster_246_row_5359"])][["compound_id", "cluster_id", "vina_score_kcal_mol", "pose_family", "core_centroid_to_TES_A", "core_raw_rmsd_to_TES_A", "core_main_axis_angle_deg", "hbond_candidate_count", "polar_contact_count", "hydrophobic_contact_count"]]
    c3 = pose[pose.compound_id.isin(["cluster_3_row_5354", "cluster_3_row_2412"])][["compound_id", "vina_score_kcal_mol", "pose_family", "core_centroid_to_TES_A", "core_raw_rmsd_to_TES_A", "core_main_axis_angle_deg", "hbond_candidate_count", "polar_contact_count", "hydrophobic_contact_count"]]
    c246 = pose[pose.compound_id.isin(["cluster_246_row_5372", "cluster_246_row_704"])][["compound_id", "vina_score_kcal_mol", "pose_family", "core_centroid_to_TES_A", "core_raw_rmsd_to_TES_A", "core_main_axis_angle_deg", "hbond_candidate_count", "polar_contact_count", "hydrophobic_contact_count"]]

    def table(df: pd.DataFrame) -> str:
        clean = df.reset_index(drop=True).copy()
        headers = [str(c) for c in clean.columns]
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, row in clean.iterrows():
            values = []
            for value in row.tolist():
                if isinstance(value, (float, np.floating)):
                    values.append("NA" if not np.isfinite(value) else f"{value:.3f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    report = f"""# Step 4.2: Binding Pose and Protein–Ligand Interaction Analysis

## Scope and coordinate provenance

本步骤只分析已有的 24 个 docking-ready 化合物的 top-ranked poses。没有重新运行 docking，没有调用 Vina `score()` 或 `optimize()`，没有修改 receptor、docking box、原始 PDB、配体 PDBQT 或 Step 4.1 结果。

- Receptor coordinate frame: `data/prepared/2AM9/receptor_repaired.pdb`
- Receptor hydrogen geometry: `data/prepared/2AM9/receptor_repaired_H.pdb`
- Native reference: `data/prepared/2AM9/TES_crystal.sdf`
- Top poses: `results/docking_2AM9/top_sdf/`
- Docking score source: `results/docking_2AM9/docking_top_poses.csv`

All 24 pose SDF structures matched their verified stereochemical structures before analysis. The receptor was kept fixed, and all ligand coordinates were analyzed in the original docking coordinate frame.

## 1. Methods

Interaction analysis used explicit 3D geometry rules implemented in `scripts/06_pose_analysis.py`. Hydrogen-bond results are reported only as `hydrogen_bond_candidate`; polar distance alone is reported as `polar_contact`. The rules and thresholds are recorded in `results/pose_analysis/pose_analysis_summary.json`.

The common scaffold was an RDKit MCS across the 24 top poses plus native TES, requiring complete rings with element and bond-order matching and no chiral matching. The resulting query contains {summary['mcs']['query_atom_count']} atoms. Core metrics are raw coordinate-frame comparisons; no independent rigid-body superposition was applied.

The interaction candidates detected across all poses were: hydrogen-bond candidates {interaction_counts['hydrogen_bond_candidate']}, polar contacts {interaction_counts['polar_contact']}, hydrophobic contacts {interaction_counts['hydrophobic_contact']}, close van der Waals contacts {interaction_counts['close_vdw_contact']}, π-contact candidates {interaction_counts['pi_contact_candidate']}, halogen-contact candidates {interaction_counts['halogen_contact_candidate']}, and steric clashes below 2 Å {interaction_counts['steric_clash']}.

## 2. Pose organization

Exploratory agglomerative clustering of standardized pose descriptors and key-residue interaction bits selected {summary['family_assignments']['selected_family_count']} pose families, with silhouette score {summary['family_assignments']['selected_silhouette']:.3f}. Family sizes were {family_counts}. This is an organization of the current poses, not a validated classification.

Family by cluster counts:

{table(family_cluster.reset_index())}

The family assignment is useful for checking pose heterogeneity. It should not be interpreted as evidence that the docking protocol found physically distinct binding states.

## 3. Interaction landscape

The most frequently observed key-residue contact records were: {key_summary}. Key residues were inspected explicitly: Asn705, Gln711, Arg752 and Thr877. The complete residue-level table is in `ligand_residue_contacts.csv`; the binary residue/interactions matrix is in `interaction_fingerprints.csv`.

The most common pattern is a hydrophobic steroid-pocket contact network, accompanied by polar contacts around the oxygen-containing functional groups. Candidate hydrogen bonds are less frequent than generic polar contacts. No sub-2 Å heavy-atom steric clashes were detected under the stated rule. These observations describe predicted geometry and do not validate an experimental binding mechanism.

## 4. Stereo-sensitive pairs

### Cluster 176: row 5378 versus row 5380

{table(stereo[stereo.cluster_id == 176])}

The pair has the same two-dimensional connectivity but different stereochemistry and a score difference of approximately 0.672 kcal/mol. Both were assigned to the same exploratory pose family. Their predicted core placement and contact counts are similar at this coarse level, so the current analysis does not establish a distinct stereo-driven pose family. A stereochemical effect remains plausible, but it is not isolated from ligand geometry and scoring-function effects.

### Cluster 246: row 5356 versus row 5359

{table(stereo[stereo.cluster_id == 246])}

The score difference is approximately 0.531 kcal/mol. Both were assigned to the same exploratory pose family. The pair differs in detailed contact counts, but the interaction candidates do not by themselves prove that stereochemistry caused the score difference.

## 5. Cluster 3 contrast

{table(c3)}

The more favorable row 5354 pose has a much smaller raw common-core displacement from TES than row 2412 in this coordinate frame. Row 2412 also has a larger ligand bounding box and more hydrophobic/close-contact counts. These are direct geometric observations that could contribute to score heterogeneity, but they cannot identify which term of Vina caused the difference. The contrast is not a proof that the less favorable pose is incorrectly docked.

## 6. Cluster 246 representatives

{table(c246)}

Rows 5372 and 704 were selected as more- and less-favorable score representatives. They were selected to span the Cluster 246 score range, not because the two poses alone define the cluster. Their pose families, core placement, functional-group geometry and residue-contact summaries should be used for later focused multi-seed or receptor-ensemble checks.

## 7. Androgens-annotated references

The original Androgens-annotated docking-ready references were retained as separate reference compounds:

- Cluster 3: `cluster_3_row_5354`
- Cluster 176: `cluster_176_row_5378`
- Cluster 246: `cluster_246_row_5330`
- Cluster 416: `cluster_416_row_5373`

Their complete pose and interaction records are in `pose_descriptors.csv`, `interaction_summary.csv`, and `ligand_residue_contacts.csv`.

## 8. Figures and PyMOL outputs

- Figure 5 all-pose overlay: `{figure_outputs['all']}`
- Figure 5 cluster overlays: `figures/pose_analysis/figure5_cluster_overlays.png`
- Figure 6 interaction fingerprint heatmap: `figures/pose_analysis/figure6_interaction_heatmap.png`
- Figure 7 case overlays: saved under `figures/pose_analysis/figure7_case_*.png`
- Figure 8 exploratory pose-family summary: `figures/pose_analysis/figure8_pose_family_summary.png`
- Reproducible PyMOL scripts: `results/pose_analysis/pymol_scripts/`

The overlays show the repaired AR receptor as cartoon, native TES in yellow, and predicted ligands as sticks. They are visualization aids for the predicted coordinate geometries.

## 9. Limitations and practical interpretation

- Vina scores are model scores, not experimental binding free energies or AR activity measurements.
- The interaction rules identify geometric candidates. They do not prove hydrogen bonds, water-mediated contacts, binding mechanisms or pharmacological effects.
- Only one receptor conformation, 2AM9, and one batch-docking seed were used for the 24 candidates.
- The MCS contains a conservative 10-atom common core; it does not make the full molecules chemically identical.
- Pairwise pose/fingerprint relationships are not independent observations.
- The current analysis does not include explicit solvent, receptor flexibility, alternate protonation microstates, or ensemble docking.
- The two pose families are exploratory and may change under additional seeds, receptors, or interaction definitions.

## 10. Recommended next step

The most informative next checks are multi-seed redocking for the two stereo pairs, the Cluster 3 contrast pair, and the Cluster 246 representatives. Cross-docking or ensemble docking on 2AMA/2AMB may be useful for a limited representative set after the current pose observations are reviewed. No such additional calculation was run in Step 4.2.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    pose, contacts, fp, summary = load_data()
    FIG.mkdir(parents=True, exist_ok=True)
    heatmap_columns = make_interaction_heatmap(pose, fp)
    make_family_summary(pose, fp)
    if os.environ.get("STEP42_SKIP_PYMOL") == "1":
        figure_outputs = {
            "all": "figures/pose_analysis/figure5_all_pose_overlay.png",
            "cluster": {str(c): f"figures/pose_analysis/figure5_cluster_{c}_overlay.png" for c in CLUSTERS},
            "cases": {p.stem: str(p.relative_to(ROOT)) for p in FIG.glob("figure7_case_*.png")},
        }
    else:
        figure_outputs = make_overlays(pose)
    write_report(pose, contacts, summary, figure_outputs, heatmap_columns)
    metadata = {"status": "success", "figures": figure_outputs, "heatmap_column_count": len(heatmap_columns), "report": str(REPORT.relative_to(ROOT))}
    (OUT / "interaction_analysis_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
