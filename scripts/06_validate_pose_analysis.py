"""Independent QC for Step 4.2 pose and interaction outputs."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv"
READY = ROOT / "results" / "ligand_preparation" / "docking_ready.csv"
QC = ROOT / "results" / "docking_2AM9" / "docking_qc.csv"
TOP_SDF = ROOT / "results" / "docking_2AM9" / "top_sdf"
OUT = ROOT / "results" / "pose_analysis"
FIG = ROOT / "figures" / "pose_analysis"
REPORT = ROOT / "reports" / "binding_pose_interaction_analysis.md"


def main() -> None:
    failures: list[str] = []
    top = pd.read_csv(TOP)
    top = top[(top.pose_rank == 1) & (top.docking_status == "success")]
    ready = pd.read_csv(READY)
    ready = ready[ready.docking_status == "docking_ready"]
    qc = pd.read_csv(QC)
    pose = pd.read_csv(OUT / "pose_descriptors.csv")
    contacts = pd.read_csv(OUT / "ligand_residue_contacts.csv")
    fp = pd.read_csv(OUT / "interaction_fingerprints.csv")
    summary = json.loads((OUT / "pose_analysis_summary.json").read_text(encoding="utf-8"))

    expected_ids = set(ready.compound_id)
    if len(top) != 24 or len(pose) != 24 or pose.compound_id.nunique() != 24 or set(pose.compound_id) != expected_ids:
        failures.append("pose table does not contain exactly the 24 docking-ready top poses")
    if pose.cluster_id.value_counts().to_dict() != {246: 11, 176: 6, 3: 5, 416: 2}:
        failures.append("pose cluster counts are incorrect")
    if pose.compound_id.duplicated().any():
        failures.append("duplicate compound in pose descriptors")

    top_score = top.set_index("compound_id").vina_score_kcal_mol.sort_index()
    pose_score = pose.set_index("compound_id").vina_score_kcal_mol.sort_index()
    if not top_score.index.equals(pose_score.index) or not np.allclose(top_score.values, pose_score.values, atol=1e-9):
        failures.append("pose score values do not match docking_top_poses.csv")
    qc_index = qc.set_index("compound_id").loc[sorted(expected_ids)]
    if not bool(qc_index.input_ligand_vina_load_ok.all()) or not bool(qc_index.top_pose_stereo_connectivity_match.all()):
        failures.append("recorded Step 3.2 QC not passed for all poses")

    sdf_failures = []
    for compound_id in sorted(expected_ids):
        path = TOP_SDF / f"{compound_id}.sdf"
        if not path.exists() or path.stat().st_size == 0:
            sdf_failures.append(compound_id)
            continue
        mol = Chem.SDMolSupplier(str(path), removeHs=False)[0]
        if mol is None or mol.GetNumConformers() == 0:
            sdf_failures.append(compound_id)
    if sdf_failures:
        failures.append(f"invalid or missing 3D top SDF: {sdf_failures}")

    if set(contacts.compound_id) != expected_ids or len(contacts) == 0:
        failures.append("residue contact table does not cover all compounds")
    if set(fp.compound_id) != expected_ids or len(fp.columns) <= 1:
        failures.append("interaction fingerprint table is incomplete")
    for key in ["ASN705", "GLN711", "ARG752", "THR877"]:
        if not any(key in c for c in fp.columns):
            failures.append(f"key residue {key} absent from fingerprint matrix")
    if not (pose.steric_clash_count == 0).all():
        failures.append("at least one pose has a recorded heavy-atom clash below 2 A")
    if not pose.pose_family.notna().all():
        failures.append("pose family assignment missing")
    if summary.get("scope", {}).get("docking_rerun") is not False:
        failures.append("summary does not record docking_rerun=false")
    if summary.get("mcs", {}).get("query_atom_count", 0) < 6:
        failures.append("common MCS is unexpectedly small")

    required_figures = [
        "figure5_all_pose_overlay.png",
        "figure5_cluster_overlays.png",
        "figure6_interaction_heatmap.png",
        "figure7_case_stereo_pair_5378_5380.png",
        "figure7_case_stereo_pair_5356_5359.png",
        "figure7_case_cluster3_5354_vs_2412.png",
        "figure7_case_cluster246_reps.png",
        "figure8_pose_family_summary.png",
    ]
    for name in required_figures:
        path = FIG / name
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing figure: {name}")
        elif Image.open(path).size[0] < 500 or Image.open(path).size[1] < 400:
            failures.append(f"figure unexpectedly small: {name}")
    pml_count = len(list((OUT / "pymol_scripts").glob("*.pml")))
    if pml_count < 9:
        failures.append(f"expected reproducible PyMOL scripts, found {pml_count}")
    if not REPORT.exists() or "hydrogen_bond_candidate" not in REPORT.read_text(encoding="utf-8") or "Vina" not in REPORT.read_text(encoding="utf-8"):
        failures.append("Step 4.2 report missing or incomplete")

    result = {"status": "PASS" if not failures else "FAIL", "failure_count": len(failures), "failures": failures, "compound_count": int(len(pose)), "contact_rows": int(len(contacts)), "fingerprint_columns": int(len(fp.columns) - 1), "pymol_script_count": pml_count}
    (OUT / "pose_analysis_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
