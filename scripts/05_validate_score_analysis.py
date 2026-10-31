"""Independent QC for Step 4.1 score/structure analysis outputs.

This validator recomputes the key joins, descriptor values, pair counts and
matrix properties from the raw Step 3.1/3.2 files.  It does not run docking and
does not import AutoDock Vina.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv"
QC = ROOT / "results" / "docking_2AM9" / "docking_qc.csv"
READY = ROOT / "results" / "ligand_preparation" / "docking_ready.csv"
MANIFEST = ROOT / "results" / "ligand_preparation" / "ligand_manifest.csv"
OUT = ROOT / "results" / "docking_analysis"
FIG = ROOT / "figures" / "docking_analysis"
REPORT = ROOT / "reports" / "docking_score_structure_analysis.md"


def mol_from_smiles(s: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {s}")
    return mol


def canonical_nonisomeric(mol: Chem.Mol) -> str:
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def fail(checks: dict, name: str, detail: str) -> None:
    checks[name] = {"pass": False, "detail": detail}


def ok(checks: dict, name: str, detail: str) -> None:
    checks[name] = {"pass": True, "detail": detail}


def main() -> None:
    checks: dict[str, dict] = {}
    errors: list[str] = []
    top = pd.read_csv(TOP)
    ready_all = pd.read_csv(READY)
    ready = ready_all[ready_all.docking_status == "docking_ready"].copy()
    manifest = pd.read_csv(MANIFEST)
    qc = pd.read_csv(QC)
    desc = pd.read_csv(OUT / "compound_descriptors.csv")
    score_stats = pd.read_csv(OUT / "cluster_score_statistics.csv")
    pairs = pd.read_csv(OUT / "pairwise_similarity_score.csv")
    mat_no = pd.read_csv(OUT / "similarity_matrix_no_chirality.csv", index_col=0)
    mat_chiral = pd.read_csv(OUT / "similarity_matrix_with_chirality.csv", index_col=0)
    summary = json.loads((OUT / "analysis_summary.json").read_text(encoding="utf-8"))

    ids = set(ready.compound_id)
    top_ids = set(top.compound_id)
    if len(ready) == 24 and len(top[top.pose_rank == 1]) == 24 and top.pose_rank.eq(1).sum() == top.compound_id.nunique():
        ok(checks, "analysis_contains_24_unique_top_poses", "24 docking-ready compounds and one rank-1 pose per compound")
    else:
        fail(checks, "analysis_contains_24_unique_top_poses", f"ready={len(ready)}, rank1={int(top.pose_rank.eq(1).sum())}, unique={top.compound_id.nunique()}")
    if top_ids == ids:
        ok(checks, "compound_set_matches_docking_ready", "No extra or missing compound IDs")
    else:
        fail(checks, "compound_set_matches_docking_ready", f"extra={sorted(top_ids - ids)}, missing={sorted(ids - top_ids)}")
    if not set(desc.compound_id).intersection(set(ready_all[ready_all.docking_status != "docking_ready"].compound_id)):
        ok(checks, "review_required_excluded", "No review_required record appears in analysis descriptors")
    else:
        fail(checks, "review_required_excluded", "A non-docking-ready compound appears in descriptors")

    expected_counts = {3: 5, 176: 6, 246: 11, 416: 2}
    actual_counts = ready.groupby("cluster_id").size().to_dict()
    if actual_counts == expected_counts:
        ok(checks, "cluster_counts", str(actual_counts))
    else:
        fail(checks, "cluster_counts", f"expected={expected_counts}, actual={actual_counts}")

    top_scores = top[top.pose_rank == 1].set_index("compound_id").vina_score_kcal_mol.sort_index()
    qc_scores = qc.set_index("compound_id").top_score_kcal_mol.sort_index()
    desc_scores = desc.set_index("compound_id").vina_score_kcal_mol.sort_index()
    if top_scores.index.equals(qc_scores.index) and np.allclose(top_scores.values, qc_scores.values, atol=1e-9) and top_scores.index.equals(desc_scores.index) and np.allclose(top_scores.values, desc_scores.values, atol=1e-9):
        ok(checks, "top_score_reproduces_step3_2", "Top-pose scores agree across docking_top_poses, docking_qc and descriptor table")
    else:
        fail(checks, "top_score_reproduces_step3_2", "Score or compound-ID mismatch across source and analysis tables")

    if len(qc) == 24 and bool(qc.input_ligand_vina_load_ok.all()) and bool((qc.top_pose_outside_box_atoms == 0).all()) and bool((qc.receptor_heavy_clashes_lt_2A == 0).all()) and bool(qc.top_pose_stereo_connectivity_match.all()):
        ok(checks, "all_top_pose_qc_pass", "All recorded Step 3.2 QC flags pass")
    else:
        fail(checks, "all_top_pose_qc_pass", "At least one Step 3.2 QC flag failed")

    # Independent descriptor recomputation from the verified explicit SMILES.
    ready_by_id = ready.set_index("compound_id")
    desc_by_id = desc.set_index("compound_id")
    descriptor_fields = {
        "molecular_weight": lambda m: Descriptors.MolWt(m),
        "heavy_atom_count": lambda m: rdMolDescriptors.CalcNumHeavyAtoms(m),
        "clogp": lambda m: Crippen.MolLogP(m),
        "tpsa": lambda m: rdMolDescriptors.CalcTPSA(m),
        "hbd": lambda m: Lipinski.NumHDonors(m),
        "hba": lambda m: Lipinski.NumHAcceptors(m),
        "rotatable_bonds": lambda m: Lipinski.NumRotatableBonds(m),
        "fraction_csp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
        "formal_charge": lambda m: Chem.GetFormalCharge(m),
        "ring_count": lambda m: rdMolDescriptors.CalcNumRings(m),
    }
    descriptor_errors = 0
    for cid, row in ready_by_id.iterrows():
        mol = mol_from_smiles(row.original_smiles)
        for col, fn in descriptor_fields.items():
            if not np.isclose(float(desc_by_id.loc[cid, col]), float(fn(mol)), atol=1e-8):
                descriptor_errors += 1
    if descriptor_errors == 0 and len(desc) == 24:
        ok(checks, "rdkit_descriptors_recomputed", "All requested descriptors independently reproduce")
    else:
        fail(checks, "rdkit_descriptors_recomputed", f"descriptor mismatches={descriptor_errors}, rows={len(desc)}")

    # Matrix properties and compound ordering.
    for name, matrix in [("no_chirality", mat_no), ("with_chirality", mat_chiral)]:
        arr = matrix.to_numpy(dtype=float)
        if arr.shape == (24, 24) and list(matrix.index) == list(matrix.columns) and np.allclose(arr, arr.T, atol=1e-12) and np.allclose(np.diag(arr), 1.0, atol=1e-12) and np.all((arr >= -1e-12) & (arr <= 1 + 1e-12)) and set(matrix.index) == ids:
            ok(checks, f"similarity_matrix_{name}", "24x24, symmetric, diagonal 1, values in [0,1], IDs match")
        else:
            fail(checks, f"similarity_matrix_{name}", f"shape={arr.shape}, IDs={len(set(matrix.index) & ids)}")

    expected_pairs = {tuple(sorted((a, b))) for a, b in combinations(sorted(ids), 2)}
    actual_pairs = {tuple(sorted((r.compound_id_1, r.compound_id_2))) for _, r in pairs.iterrows()}
    duplicate_pair_rows = len(pairs) - len(actual_pairs)
    self_pairs = int((pairs.compound_id_1 == pairs.compound_id_2).sum())
    if len(pairs) == 276 and actual_pairs == expected_pairs and duplicate_pair_rows == 0 and self_pairs == 0:
        ok(checks, "pairwise_completeness", "276 unique unordered pairs, no self-pairs")
    else:
        fail(checks, "pairwise_completeness", f"rows={len(pairs)}, unique={len(actual_pairs)}, duplicate_rows={duplicate_pair_rows}, self_pairs={self_pairs}")

    pair_value_errors = 0
    score_map = top_scores.to_dict()
    for _, row in pairs.iterrows():
        a, b = row.compound_id_1, row.compound_id_2
        if a not in ids or b not in ids:
            pair_value_errors += 1
            continue
        if not np.isclose(row.abs_score_difference_kcal_mol, abs(score_map[a] - score_map[b]), atol=1e-9):
            pair_value_errors += 1
        if not np.isclose(row.similarity_no_chirality, mat_no.loc[a, b], atol=1e-12):
            pair_value_errors += 1
        if not np.isclose(row.similarity_with_chirality, mat_chiral.loc[a, b], atol=1e-12):
            pair_value_errors += 1
    if pair_value_errors == 0:
        ok(checks, "pairwise_alignment", "Pairwise similarity and score differences align with IDs and matrices")
    else:
        fail(checks, "pairwise_alignment", f"mismatched pair values={pair_value_errors}")

    if len(score_stats) == 5 and set(score_stats.cluster_id.astype(str)) == {"3", "176", "246", "416", "overall"}:
        ok(checks, "score_statistics_complete", "Four cluster rows plus overall row")
    else:
        fail(checks, "score_statistics_complete", f"rows={len(score_stats)}, clusters={score_stats.cluster_id.tolist()}")

    figure_stems = ["figure1_score_distribution", "figure2_descriptor_score", "figure3_similarity_heatmap", "figure3_similarity_heatmap_chiral", "figure4_similarity_score_difference"]
    missing_figures = [str(FIG / f"{stem}{ext}") for stem in figure_stems for ext in [".png", ".svg", ".pdf"] if not (FIG / f"{stem}{ext}").exists() or (FIG / f"{stem}{ext}").stat().st_size == 0]
    if not missing_figures:
        ok(checks, "figures_present", "Five figure variants present as non-empty PNG/SVG/PDF files")
    else:
        fail(checks, "figures_present", f"missing or empty: {missing_figures}")

    report_text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    if report_text and "{fmt(" not in report_text and "Step 4.1" in report_text:
        ok(checks, "report_present", "Chinese Step 4.1 report exists without unresolved template placeholders")
    else:
        fail(checks, "report_present", "Report missing or contains unresolved placeholders")
    if summary.get("protocol_context", {}).get("docking_protocol_not_rerun") is True:
        ok(checks, "analysis_only_scope", "Summary records that docking was not rerun")
    else:
        fail(checks, "analysis_only_scope", "Analysis-only scope flag missing")

    failures = [name for name, result in checks.items() if not result["pass"]]
    output = {"status": "PASS" if not failures else "FAIL", "checks": checks, "failures": failures, "versions": {"python": sys.version.split()[0], "rdkit": rdBase.rdkitVersion, "pandas": pd.__version__, "numpy": np.__version__}}
    (OUT / "analysis_validation.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": output["status"], "failure_count": len(failures), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
