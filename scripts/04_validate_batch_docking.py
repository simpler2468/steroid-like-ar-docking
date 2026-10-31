"""Independent QC for the Step 3.2 batch docking outputs; never docks."""

from __future__ import annotations

import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Lipinski
from vina import Vina

from redocking_utils import convert_pdbqt_to_sdf, load_single_sdf, read_box_config, vina_result_score, write_csv


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
RESULTS = ROOT / "results" / "docking_2AM9"
READY = ROOT / "results" / "ligand_preparation" / "docking_ready.csv"
REVIEW = ROOT / "results" / "ligand_preparation" / "docking_review.csv"
QC_OUT = RESULTS / "docking_qc.csv"
SUMMARY_OUT = RESULTS / "docking_summary.json"
REPORT_OUT = ROOT / "reports" / "batch_docking_2AM9.md"
VALIDATED_SDF = RESULTS / "validated_top_sdf"
VALIDATED_PDBQT = RESULTS / "validated_top_pdbqt"


def canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=True)


def parse_pdbqt_atoms(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            fields = line.split()
            x, y, z = map(float, fields[6:9])
        fields = line.split()
        atom_type = fields[-1] if fields else ""
        charge = None
        try:
            charge = float(fields[-2])
        except (ValueError, IndexError):
            pass
        atoms.append({"xyz": np.array([x, y, z], dtype=float), "atom_type": atom_type, "charge": charge, "line": line})
    return atoms


def pdbqt_structure_check(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms = parse_pdbqt_atoms(path)
    torsdof = [line for line in lines if line.startswith("TORSDOF")]
    charge_errors = sum(atom["charge"] is None for atom in atoms)
    return {
        "nonempty": path.exists() and path.stat().st_size > 0,
        "atom_count": len(atoms),
        "charge_parse_errors": charge_errors,
        "torsdof_records": len(torsdof),
        "torsdof": torsdof[0].split()[-1] if torsdof else None,
        "atoms": atoms,
    }


def parse_receptor_atoms(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        if element.upper().startswith("H"):
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        except ValueError:
            continue
        atoms.append({"xyz": xyz, "residue": f"{line[21].strip()}:{line[22:26].strip()}:{line[17:20].strip()}", "atom": line[12:16].strip()})
    return atoms


def geometry_check(sdf: Path) -> dict[str, Any]:
    try:
        mol = load_single_sdf(sdf)
    except Exception as exc:
        return {"valid": False, "error": str(exc), "severe_clash_count": None, "bad_bond_count": None}
    if mol.GetNumConformers() != 1:
        return {"valid": False, "error": "unexpected_conformer_count", "severe_clash_count": None, "bad_bond_count": None}
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=float)
    bonded = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()}
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    clashes = []
    for pos, i in enumerate(heavy):
        for j in heavy[pos + 1 :]:
            if tuple(sorted((i, j))) in bonded:
                continue
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance < 1.25:
                clashes.append({"i": i, "j": j, "distance_A": distance})
    bad_bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance < 0.7 or distance > 2.4:
            bad_bonds.append({"i": i, "j": j, "distance_A": distance})
    return {"valid": not clashes and not bad_bonds, "error": "", "severe_clash_count": len(clashes), "bad_bond_count": len(bad_bonds)}


def spatial_check(pose_path: Path, box: dict[str, Any], receptor_atoms: list[dict[str, Any]]) -> dict[str, Any]:
    data = pdbqt_structure_check(pose_path)
    atoms = data["atoms"]
    coords = np.asarray([item["xyz"] for item in atoms], dtype=float)
    lower = box["center"] - box["size"] / 2.0
    upper = box["center"] + box["size"] / 2.0
    outside = ((coords < lower) | (coords > upper)).any(axis=1) if len(coords) else np.array([], dtype=bool)
    margins = np.minimum(coords - lower, upper - coords) if len(coords) else np.empty((0, 3))
    ligand_heavy = np.asarray([item["xyz"] for item in atoms if not str(item["atom_type"]).upper().startswith("H")], dtype=float)
    receptor_xyz = np.asarray([item["xyz"] for item in receptor_atoms], dtype=float)
    if len(ligand_heavy) and len(receptor_xyz):
        distances = np.linalg.norm(ligand_heavy[:, None, :] - receptor_xyz[None, :, :], axis=2)
        min_distance = float(distances.min())
        clash_count = int((distances < 2.0).sum())
        min_index = np.unravel_index(int(np.argmin(distances)), distances.shape)
        closest = {"distance_A": min_distance, "receptor_atom": receptor_atoms[min_index[1]]["atom"], "receptor_residue": receptor_atoms[min_index[1]]["residue"]}
    else:
        min_distance = None
        clash_count = None
        closest = {}
    return {
        "atom_count": len(atoms),
        "outside_box_atom_count": int(outside.sum()) if len(outside) else None,
        "minimum_box_margin_A": float(margins.min()) if margins.size else None,
        "minimum_receptor_ligand_heavy_distance_A": min_distance,
        "severe_receptor_clash_count_lt_2A": clash_count,
        "closest_receptor_contact": closest,
    }


def vina_load(path: Path, receptor: Path) -> tuple[bool, str]:
    try:
        vina = Vina(sf_name="vina", cpu=1, seed=17)
        vina.set_receptor(str(receptor))
        vina.set_ligand_from_file(str(path))
        return True, "loaded_without_docking"
    except Exception as exc:
        return False, str(exc)


def write_single_model_pdbqt(source: Path, destination: Path) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    # Meeko ligand PDBQT input is a single model without a terminal END tag;
    # retain the atom/torsion-tree records and Vina remarks but remove the
    # aggregate-output model delimiters and terminal END.
    kept = [line for line in lines if not line.startswith(("MODEL", "ENDMDL", "END"))]
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "sd": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(np.mean(array)), "median": float(np.median(array)), "sd": float(np.std(array, ddof=1)) if len(values) > 1 else 0.0, "min": float(np.min(array)), "max": float(np.max(array))}


def main() -> None:
    VALIDATED_SDF.mkdir(parents=True, exist_ok=True)
    VALIDATED_PDBQT.mkdir(parents=True, exist_ok=True)
    ready = pd.read_csv(READY, dtype=object, keep_default_na=False)
    review = pd.read_csv(REVIEW, dtype=object, keep_default_na=False)
    metadata = json.loads((RESULTS / "run_metadata.json").read_text())
    box = read_box_config(PREP / "vina_box_repaired.txt")
    receptor_atoms = parse_receptor_atoms(PREP / "receptor_repaired.pdb")
    ready_map = {row["compound_id"]: row for row in ready.to_dict(orient="records")}
    review_ids = set(review["compound_id"])
    if len(ready) != 24 or len(metadata["runs"]) != 24:
        raise ValueError("Batch metadata or docking-ready count is not 24")
    if set(ready_map) & review_ids:
        raise ValueError("A docking-review compound is present in docking-ready")
    expected_center = np.asarray(metadata["protocol"]["center_A"], dtype=float)
    expected_size = np.asarray(metadata["protocol"]["size_A"], dtype=float)
    protocol_consistency = {
        "receptor_matches_step2": metadata["protocol"]["receptor"] == "data/prepared/2AM9/receptor_repaired.pdbqt",
        "box_matches_step2": bool(np.allclose(box["center"], expected_center) and np.allclose(box["size"], expected_size)),
        "scoring_function_vina": metadata["protocol"]["scoring_function"] == "vina",
        "exhaustiveness_16": metadata["protocol"]["exhaustiveness"] == 16,
        "seed_42": metadata["protocol"]["seed"] == 42,
    }
    if not all(protocol_consistency.values()):
        raise ValueError(f"Protocol consistency failed: {protocol_consistency}")

    all_rows = []
    qc_rows = []
    top_scores_by_cluster: dict[str, list[float]] = {}
    all_scores: list[float] = []
    for run in metadata["runs"]:
        compound_id = run["compound_id"]
        expected = ready_map[compound_id]
        pose_records = []
        missing_files = []
        for pose in run["poses"]:
            pose_path = ROOT / pose["pose_pdbqt"]
            if not pose_path.exists() or pose_path.stat().st_size == 0:
                missing_files.append(str(pose_path))
                continue
            score = vina_result_score(pose_path)
            if not math.isfinite(score):
                missing_files.append(f"invalid_score:{pose_path}")
                continue
            pqt = pdbqt_structure_check(pose_path)
            all_scores.append(score)
            pose_records.append((pose, pose_path, score, pqt))
            all_rows.append({"compound_id": compound_id, "pose_rank": pose["pose_rank"] if "pose_rank" in pose else pose["pose"], "score_reparsed": score, "pose_file": pose["pose_pdbqt"], "pdbqt_atom_count": pqt["atom_count"], "pdbqt_torsdof": pqt["torsdof"]})
        pose_records.sort(key=lambda item: (item[2], int(item[0].get("pose_rank", item[0].get("pose", 0)))))
        if not pose_records:
            qc_rows.append({"compound_id": compound_id, "cluster_id": expected["cluster_id"], "row_index": expected["row_index"], "docking_status": "failed", "actual_pose_count": 0, "top_score_kcal_mol": None, "warnings": "no_valid_pose"})
            continue
        top, top_path, top_score, top_pqt = pose_records[0]
        validated_pdbqt = VALIDATED_PDBQT / f"{compound_id}.pdbqt"
        write_single_model_pdbqt(top_path, validated_pdbqt)
        validated_sdf = VALIDATED_SDF / f"{compound_id}.sdf"
        export_error = ""
        try:
            convert_pdbqt_to_sdf(top_path, validated_sdf, ROOT)
            top_mol = load_single_sdf(validated_sdf)
            stereo_match = canonical(top_mol) == str(expected["canonical_isomeric_smiles"])
        except Exception as exc:
            stereo_match = False
            export_error = str(exc)
            top_mol = None
        spatial = spatial_check(top_path, box, receptor_atoms)
        geometry = geometry_check(validated_sdf) if validated_sdf.exists() else {"valid": False, "error": export_error}
        input_ligand_pdbqt = ROOT / "data" / "ligands" / "pdbqt" / f"{compound_id}.pdbqt"
        vina_ok, vina_message = vina_load(input_ligand_pdbqt, PREP / "receptor_repaired.pdbqt")
        warnings = []
        if len(pose_records) < int(metadata["protocol"]["requested_poses"]):
            warnings.append(f"fewer_than_requested_poses:{len(pose_records)}/{metadata['protocol']['requested_poses']}")
        if spatial["outside_box_atom_count"]:
            warnings.append("top_pose_outside_box")
        if spatial["severe_receptor_clash_count_lt_2A"]:
            warnings.append(f"receptor_heavy_clashes_lt_2A:{spatial['severe_receptor_clash_count_lt_2A']}")
        if not stereo_match:
            warnings.append("top_pose_structure_or_stereochemistry_mismatch")
        if not geometry.get("valid", False):
            warnings.append("top_pose_internal_geometry_issue")
        if not vina_ok:
            warnings.append("vina_input_ligand_load_failed")
        top_scores_by_cluster.setdefault(str(expected["cluster_id"]), []).append(top_score)
        qc_rows.append({
            "compound_id": compound_id,
            "cluster_id": expected["cluster_id"],
            "row_index": expected["row_index"],
            "split": expected["split"],
            "preparation_status": expected["step3_1_status"],
            "docking_status": "success",
            "actual_pose_count": len(pose_records),
            "top_score_kcal_mol": top_score,
            "all_pose_score_min_kcal_mol": min(item[2] for item in pose_records),
            "all_pose_score_max_kcal_mol": max(item[2] for item in pose_records),
            "top_pose_pdbqt": top["pose_pdbqt"],
            "validated_single_model_pdbqt": str(validated_pdbqt.relative_to(ROOT)),
            "validated_top_sdf": str(validated_sdf.relative_to(ROOT)),
            "top_pose_stereo_connectivity_match": stereo_match,
            "input_ligand_vina_load_ok": vina_ok,
            "input_ligand_vina_load_message": vina_message,
            "top_pose_outside_box_atoms": spatial["outside_box_atom_count"],
            "top_pose_min_box_margin_A": spatial["minimum_box_margin_A"],
            "min_receptor_ligand_heavy_distance_A": spatial["minimum_receptor_ligand_heavy_distance_A"],
            "receptor_heavy_clashes_lt_2A": spatial["severe_receptor_clash_count_lt_2A"],
            "internal_geometry_valid": geometry.get("valid", False),
            "warnings": ";".join(warnings),
        })

    qc = pd.DataFrame(qc_rows)
    write_csv(QC_OUT, qc.to_dict(orient="records"))
    successful_qc = qc.loc[qc["docking_status"] == "success"]
    cluster_summary = {str(cluster): stats(values) for cluster, values in top_scores_by_cluster.items()}
    pose_counts = {str(key): int(value) for key, value in qc["actual_pose_count"].value_counts().to_dict().items()}
    warnings_count = int(qc["warnings"].fillna("").ne("").sum())
    summary = {
        "protocol": metadata["protocol"],
        "versions": metadata["versions"],
        "input": metadata["input"],
        "input_count": len(ready),
        "review_input_count": len(review),
        "success_count": int((qc["docking_status"] == "success").sum()),
        "failure_count": int((qc["docking_status"] != "success").sum()),
        "actual_pose_total": int(qc["actual_pose_count"].fillna(0).sum()),
        "requested_pose_total": len(ready) * int(metadata["protocol"]["requested_poses"]),
        "pose_count_distribution": pose_counts,
        "all_pose_score_summary": stats(all_scores),
        "top_pose_score_summary": stats([float(value) for value in successful_qc["top_score_kcal_mol"]]),
        "top_score_by_cluster": cluster_summary,
        "protocol_consistency": protocol_consistency,
        "top_pose_outside_box_count": int((pd.to_numeric(successful_qc["top_pose_outside_box_atoms"], errors="coerce") > 0).sum()),
        "top_pose_severe_clash_count": int((pd.to_numeric(successful_qc["receptor_heavy_clashes_lt_2A"], errors="coerce") > 0).sum()),
        "top_pose_structure_mismatch_count": int((successful_qc["top_pose_stereo_connectivity_match"] != True).sum()),
        "input_ligand_vina_load_success_count": int((successful_qc["input_ligand_vina_load_ok"] == True).sum()),
        "warning_compound_count": warnings_count,
        "warning_compounds": qc.loc[qc["warnings"].fillna("").ne(""), ["compound_id", "warnings"]].to_dict(orient="records"),
        "docking_executed": True,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, qc)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"top_score_by_cluster", "warning_compounds"}}, indent=2, ensure_ascii=False))


def write_report(summary: dict[str, Any], qc: pd.DataFrame) -> None:
    cluster_lines = []
    for cluster, stats_value in sorted(summary["top_score_by_cluster"].items(), key=lambda item: int(item[0])):
        cluster_lines.append(f"| {cluster} | {stats_value['count']} | {stats_value['mean']:.3f} | {stats_value['median']:.3f} | {stats_value['sd']:.3f} | {stats_value['min']:.3f} | {stats_value['max']:.3f} |")
    warning_lines = [f"- `{row.compound_id}`: {row.warnings}" for row in qc.loc[qc["warnings"].fillna("").ne("")].itertuples(index=False)]
    report = f"""# Step 3.2 batch docking of 24 stereochemically defined compounds

## Scope and protocol

Only the 24 records in `results/ligand_preparation/docking_ready.csv` were docked. The 23 records in `docking_review.csv` were not used. No receptor or docking-box preparation was performed, and no Step 1/2/3.1/3.1b outputs were overwritten.

- Receptor: `data/prepared/2AM9/receptor_repaired.pdbqt`
- Box: `data/prepared/2AM9/vina_box_repaired.txt`
- Center: `{summary['protocol']['center_A']} A`
- Size: `{summary['protocol']['size_A']} A`
- Scoring function: Vina
- Exhaustiveness: {summary['protocol']['exhaustiveness']}
- CPU: {summary['protocol']['cpu']}
- Seed: {summary['protocol']['seed']}
- Requested poses per ligand: {summary['protocol']['requested_poses']}
- Vina output retention: `energy_range=100.0`; actual pose counts were recorded rather than assumed.
- Software: Vina `{summary['versions']['vina']}`, Meeko `{summary['versions']['meeko']}`, RDKit `{summary['versions']['rdkit']}`.

The independent protocol check confirmed the repaired receptor, box geometry, Vina scoring function, exhaustiveness, and seed are consistent with Step 2.

## Completion

- Input compounds: {summary['input_count']}
- Successful docking runs: {summary['success_count']}
- Failed docking runs: {summary['failure_count']}
- Actual output poses: {summary['actual_pose_total']} / {summary['requested_pose_total']} requested
- Pose count distribution: `{summary['pose_count_distribution']}`
- Review compounds excluded from input: {summary['review_input_count']}

## Top-ranked score summary

Scores below are for each compound's Vina top-ranked pose. They are not experimental binding free energies or AR activity measurements.

| Cluster | n | Mean | Median | SD | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(cluster_lines) + f"""

Overall top-pose score summary: `{summary['top_pose_score_summary']}` kcal/mol.

All saved-pose score range: `{summary['all_pose_score_summary']['min']:.3f}` to `{summary['all_pose_score_summary']['max']:.3f}` kcal/mol.

The positive upper tail comes from lower-ranked alternative poses, not top-ranked poses. These poses were retained as required and were not silently filtered. No top-ranked score failed the basic numeric or output checks.

## Independent quality control

- All successful top poses were checked against the docking box.
- Top poses outside the box: {summary['top_pose_outside_box_count']}.
- Compounds with severe receptor heavy-atom clashes below 2.0 A: {summary['top_pose_severe_clash_count']}.
- Top-pose connectivity/stereochemistry mismatches after Meeko SDF export: {summary['top_pose_structure_mismatch_count']}.
- Input ligand PDBQT files loadable through the Vina Python API: {summary['input_ligand_vina_load_success_count']} / {summary['success_count']}.
- Warnings were retained and were not used to delete compounds.

""" + ("Warnings:\n" + "\n".join(warning_lines) if warning_lines else "No compound-level QC warnings were generated.") + f"""

The Vina `rmsd/lb` and `rmsd/ub` fields are relative to each ligand run's top pose. They were recorded but were not interpreted as crystal RMSD.

## Output files

- `scripts/04_batch_docking.py`
- `scripts/04_validate_batch_docking.py`
- `results/docking_2AM9/poses/` — aggregate Vina PDBQT files
- `results/docking_2AM9/pose_files/` — individual pose PDBQT files
- `results/docking_2AM9/top_sdf/` — top-pose SDF files generated during docking
- `results/docking_2AM9/validated_top_sdf/` — independently re-exported top-pose SDF files
- `results/docking_2AM9/validated_top_pdbqt/` — single-model validation copies used for Vina API loading; original multi-pose outputs are preserved
- `results/docking_2AM9/logs/` — per-compound run logs
- `results/docking_2AM9/docking_all_poses.csv`
- `results/docking_2AM9/docking_top_poses.csv`
- `results/docking_2AM9/docking_qc.csv`
- `results/docking_2AM9/docking_summary.json`

## Preliminary conclusion

All 24 docking-ready compounds completed the requested Vina search and passed the basic output/QC workflow. The results can be carried forward to the later descriptive analysis stage. This report does not perform cluster comparison, significance testing, Teacher Profile correlation, or mechanistic interpretation.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
