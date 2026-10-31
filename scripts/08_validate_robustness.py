#!/usr/bin/env python3
"""Independent validation for Step 4.4 orientation/multi-seed outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
ROB = ROOT / "results" / "orientation_robustness"
FIG = ROOT / "figures" / "orientation_robustness"
TARGETS = ["cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]
SEEDS = [42, 123, 2026, 7, 99, 11, 17, 73, 101, 314]


def main():
    failures = []
    metadata_path = ROB / "multiseed_run_metadata.json"
    if not metadata_path.exists():
        failures.append("missing_multiseed_metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    protocol = metadata.get("protocol", {})
    if protocol.get("method") != "Vina.dock (full search); optimize not used":
        failures.append("protocol_does_not_prove_full_dock")
    if protocol.get("seeds") != SEEDS or protocol.get("requested_poses") != 10 or protocol.get("exhaustiveness") != 16:
        failures.append("protocol_parameters_mismatch")
    if metadata.get("failure_count") != 0 or len(metadata.get("runs", [])) != 50:
        failures.append("not_50_successful_runs")
    if set(metadata.get("target_compounds", [])) != set(TARGETS):
        failures.append("target_compound_set_mismatch")

    top = pd.read_csv(ROB / "multi_seed_top_poses.csv")
    allp = pd.read_csv(ROB / "all_pose_orientation.csv")
    mapping = pd.read_csv(ROB / "atom_mapping" / "c3_c17_atom_mapping.csv")
    stability = pd.read_csv(ROB / "orientation_stability.csv")
    if len(top) != 50:
        failures.append(f"top_pose_count:{len(top)}")
    if len(allp) < 490 or len(allp) > 500:
        failures.append(f"unexpected_all_pose_count:{len(allp)}")
    if set(top.compound_id) != set(TARGETS) or set(allp.compound_id) != set(TARGETS):
        failures.append("compound_coverage_mismatch")
    counts = top.groupby(["compound_id", "seed"]).size()
    if len(counts) != 50 or not all(counts == 1):
        failures.append("top_pose_seed_coverage_invalid")
    if set(top.seed) != set(SEEDS):
        failures.append("seed_coverage_invalid")
    if not all(top.pose_rank == 1):
        failures.append("top_table_contains_nonrank1_pose")
    if set(mapping.mapping_status) != {"validated"} or not all(mapping.tes_candidate_core_mapped_atoms == 17):
        failures.append("c3_c17_mapping_not_validated")
    if not all(np.isfinite(top.angle_to_TES_deg)) or not all(np.isfinite(top.D_same_TES_A)) or not all(np.isfinite(top.D_swap_TES_A)):
        failures.append("nonfinite_orientation_metric")
    allowed = {"consistent_with_native_orientation", "head_to_tail_reversal_supported", "alternative_orientation", "ambiguous"}
    if set(top.orientation_class) - allowed or set(allp.orientation_class) - allowed:
        failures.append("unknown_orientation_class")

    score_src = pd.read_csv(ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv").set_index("compound_id").vina_score_kcal_mol.to_dict()
    seed42 = top[top.seed == 42]
    for _, row in seed42.iterrows():
        if not np.isclose(float(row.vina_score_kcal_mol), float(score_src[row.compound_id]), atol=1e-3):
            failures.append(f"seed42_score_mismatch:{row.compound_id}")
    for (cid, seed), sub in allp.groupby(["compound_id", "seed"]):
        if not np.isclose(float(sub.iloc[0].vina_score_kcal_mol), float(sub.vina_score_kcal_mol.min()), atol=1e-8):
            failures.append(f"rank1_not_best_score:{cid}:{seed}")
        for rel in sub.sdf:
            p = ROOT / rel
            mol = Chem.MolFromMolFile(str(p), removeHs=False, sanitize=True) if p.exists() else None
            if mol is None or mol.GetNumConformers() == 0:
                failures.append(f"invalid_pose_sdf:{rel}")
                break

    for stem in ["c3_c17_orientation", "representative_3d_orientation"]:
        for ext in ["png", "svg", "pdf"]:
            p = FIG / f"{stem}.{ext}"
            if not p.exists() or p.stat().st_size == 0:
                failures.append(f"missing_3d_figure:{p.name}")
        for ext in ["pml", "pse"]:
            p = ROB / "pymol_scripts" / f"{stem}.{ext}"
            if not p.exists() or p.stat().st_size == 0:
                failures.append(f"missing_pymol_artifact:{p.name}")
    for stem in ["multi_seed_orientation", "multi_seed_scores", "orientation_vs_score"]:
        for ext in ["png", "svg", "pdf"]:
            p = FIG / f"{stem}.{ext}"
            if not p.exists() or p.stat().st_size == 0:
                failures.append(f"missing_plot:{p.name}")
    report_path = ROOT / "reports" / "orientation_robustness_analysis.md"
    if not report_path.exists() or report_path.stat().st_size == 0:
        failures.append("missing_report")
    else:
        report = report_path.read_text(encoding="utf-8")
        for marker in ["Vina.dock", "C3", "C17", "head_to_tail_reversal_supported", "2412", "5379", "2AMA", "2AMB"]:
            if marker not in report:
                failures.append(f"report_missing_marker:{marker}")
    analysis_summary = ROB / "analysis_summary.json"
    if not analysis_summary.exists():
        failures.append("missing_analysis_summary")
    result = {"status": "PASS" if not failures else "FAIL", "failure_count": len(failures), "failures": failures, "top_pose_rows": len(top), "all_pose_rows": len(allp), "run_count": len(metadata.get("runs", [])), "mapping_statuses": sorted(set(mapping.mapping_status))}
    (ROB / "robustness_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
