#!/usr/bin/env python3
"""Independent validation for Step 4.3 generated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "c17_analysis"
FIG = ROOT / "figures" / "c17_analysis"
TARGETS = ["cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]
PAIRS = ["A", "B", "C"]


def main():
    failures = []
    required = [
        OUT / "matched_pair_structure_validation.csv", OUT / "common_core_mapping.csv", OUT / "full_mcs_mapping.csv", OUT / "pairwise_pose_geometry.csv", OUT / "c17_local_contacts.csv", OUT / "comparison_summary.csv", OUT / "c17_analysis_manifest.json", ROOT / "reports" / "c17_substitution_analysis.md",
        FIG / "c17_comparison_2d.png", FIG / "c17_comparison_2d.pdf",
    ]
    for p in required:
        if not p.exists() or p.stat().st_size == 0:
            failures.append(f"missing_or_empty:{p.relative_to(ROOT)}")

    val = pd.read_csv(OUT / "matched_pair_structure_validation.csv")
    mapping = pd.read_csv(OUT / "common_core_mapping.csv")
    geom = pd.read_csv(OUT / "pairwise_pose_geometry.csv")
    local = pd.read_csv(OUT / "c17_local_contacts.csv")
    summary = pd.read_csv(OUT / "comparison_summary.csv")
    descriptors = pd.read_csv(ROOT / "results" / "docking_analysis" / "compound_descriptors.csv")
    top = pd.read_csv(ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv")
    ready = pd.read_csv(ROOT / "results" / "ligand_preparation" / "docking_ready.csv")

    if set(val.comparison) != set(PAIRS) or len(val) != 3:
        failures.append("comparison_table_must_have_A_B_C")
    if set(summary.compound_id) != set(TARGETS):
        failures.append("summary_compounds_do_not_match_five_requested_targets")
    if set(summary.comparison) != set(PAIRS) or len(summary) != 6:
        failures.append("summary_pair_rows_invalid")
    if set(TARGETS) - set(descriptors.compound_id):
        failures.append("target_missing_from_step41_descriptors")
    if set(TARGETS) - set(top.compound_id):
        failures.append("target_missing_from_step32_top_scores")
    if set(TARGETS) - set(ready.compound_id):
        failures.append("target_missing_from_docking_ready_manifest")

    score_map = top.set_index("compound_id").vina_score_kcal_mol.to_dict()
    for _, row in summary.iterrows():
        if not np.isclose(float(row.vina_score), float(score_map[row.compound_id]), atol=1e-8):
            failures.append(f"score_mismatch:{row.compound_id}")
        sdf = ROOT / "results" / "docking_2AM9" / "top_sdf" / f"{row.compound_id}.sdf"
        mol = Chem.MolFromMolFile(str(sdf), removeHs=False, sanitize=True) if sdf.exists() else None
        if mol is None or mol.GetNumConformers() == 0:
            failures.append(f"invalid_or_missing_top_sdf:{row.compound_id}")

    if not all(val.steroid_core_mapped_atoms == 17):
        failures.append("steroid_core_mapping_not_17_atoms_for_all_pairs")
    if not all(val.steroid_core_coverage_left == 1.0) or not all(val.steroid_core_coverage_right == 1.0):
        failures.append("steroid_core_mapping_does_not_cover_largest_ring_system")
    if not all(summary.c17_atom_index >= 0):
        failures.append("c17_atom_not_identified")
    for col in ["raw_common_core_rmsd_A", "best_fit_common_core_rmsd_A", "core_centroid_displacement_A", "core_orientation_difference_deg", "c17_distance_A"]:
        if not np.isfinite(geom[col].to_numpy(dtype=float)).all():
            failures.append(f"nonfinite_geometry:{col}")
    if not all(geom.raw_common_core_rmsd_A >= geom.best_fit_common_core_rmsd_A - 1e-6):
        failures.append("raw_rmsd_should_not_be_less_than_best_fit_rmsd")
    if set(local.compound_id) != set(TARGETS):
        failures.append("local_contact_table_does_not_cover_all_targets")
    if not all(local.distance_A > 0):
        failures.append("nonpositive_local_distance")

    for pid in PAIRS:
        for suffix in ["overlay", "c17_closeup"]:
            for ext in ["png"]:
                p = FIG / f"comparison_{pid}_{suffix}.{ext}"
                if not p.exists() or p.stat().st_size == 0:
                    failures.append(f"missing_or_empty_figure:{p.name}")
            pml = OUT / "pymol_scripts" / f"comparison_{pid}_{suffix}.pml"
            pse = OUT / "pymol_scripts" / f"comparison_{pid}_{suffix}.pse"
            for p in [pml, pse]:
                if not p.exists() or p.stat().st_size == 0:
                    failures.append(f"missing_or_empty_pymol_artifact:{p.name}")

    manifest = json.loads((OUT / "c17_analysis_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("docking_rerun") is not False:
        failures.append("manifest_docking_rerun_not_false")
    for pid, result in manifest.get("pymol", {}).items():
        for kind, info in result.items():
            if info.get("returncode") != 0:
                failures.append(f"pymol_failed:{pid}:{kind}")
    report = (ROOT / "reports" / "c17_substitution_analysis.md").read_text(encoding="utf-8")
    for marker in ["Comparison A", "Comparison B", "Comparison C", "raw RMSD", "氢键", "真实"]:
        if marker not in report:
            failures.append(f"report_missing_marker:{marker}")

    result = {"status": "PASS" if not failures else "FAIL", "failure_count": len(failures), "failures": failures, "target_count": len(TARGETS), "pair_count": len(PAIRS), "core_atoms": sorted(set(int(x) for x in val.steroid_core_mapped_atoms)), "local_contact_rows": len(local), "pymol_artifacts": 6}
    (OUT / "c17_analysis_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
