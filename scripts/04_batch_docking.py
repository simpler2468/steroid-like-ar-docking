"""Run the Step 3.2 batch docking for the 24 docking-ready compounds only."""

from __future__ import annotations

import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from vina import Vina

from redocking_utils import convert_pdbqt_to_sdf, read_box_config, split_vina_models, vina_result_fields, write_csv


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
READY = ROOT / "results" / "ligand_preparation" / "docking_ready.csv"
RESULTS = ROOT / "results" / "docking_2AM9"
POSES = RESULTS / "poses"
POSE_FILES = RESULTS / "pose_files"
TOP_SDF = RESULTS / "top_sdf"
LOGS = RESULTS / "logs"
RECEPTOR = PREP / "receptor_repaired.pdbqt"
BOX_FILE = PREP / "vina_box_repaired.txt"
SEED = 42
EXHAUSTIVENESS = 16
N_POSES = 10
CPU = 4


def load_preflight() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ready = pd.read_csv(READY, dtype=object, keep_default_na=False)
    if len(ready) != 24:
        raise ValueError(f"Expected 24 docking-ready rows, found {len(ready)}")
    if set(ready["docking_status"]) != {"docking_ready"}:
        raise ValueError("docking_ready.csv contains a non-docking_ready status")
    if len(set(ready["compound_id"])) != len(ready):
        raise ValueError("Duplicate compound_id in docking_ready.csv")
    expected = {3: 5, 176: 6, 246: 11, 416: 2}
    counts = {int(key): int(value) for key, value in ready.groupby("cluster_id").size().to_dict().items()}
    if counts != expected:
        raise ValueError(f"Unexpected docking-ready cluster counts: {counts}")
    box = read_box_config(BOX_FILE)
    if box["receptor"] != RECEPTOR.name:
        raise ValueError(f"Box receptor entry {box['receptor']} does not match {RECEPTOR.name}")
    step2_path = ROOT / "results" / "redocking_2AM9" / "run_metadata.json"
    step2 = json.loads(step2_path.read_text())
    step2_params = step2["parameters"]
    for key in ("scoring_function", "exhaustiveness", "cpu"):
        expected_value = {"scoring_function": "vina", "exhaustiveness": EXHAUSTIVENESS, "cpu": CPU}[key]
        if step2_params[key] != expected_value:
            raise ValueError(f"Step 2 protocol mismatch for {key}: {step2_params[key]} != {expected_value}")
    if not np.allclose(box["center"], step2_params["center_A"]) or not np.allclose(box["size"], step2_params["size_A"]):
        raise ValueError("Current docking box geometry differs from Step 2")
    return ready, box, step2_params


def ligand_paths(compound_id: str) -> tuple[Path, Path]:
    return ROOT / "data" / "ligands" / "pdbqt" / f"{compound_id}.pdbqt", ROOT / "data" / "ligands" / "sdf" / f"{compound_id}.sdf"


def preflight_ligands(ready: pd.DataFrame) -> list[dict[str, Any]]:
    checks = []
    for row in ready.to_dict(orient="records"):
        compound_id = row["compound_id"]
        pdbqt, sdf = ligand_paths(compound_id)
        check: dict[str, Any] = {
            "compound_id": compound_id,
            "cluster_id": row["cluster_id"],
            "row_index": row["row_index"],
            "pdbqt": str(pdbqt.relative_to(ROOT)),
            "sdf": str(sdf.relative_to(ROOT)),
            "pdbqt_exists": pdbqt.exists() and pdbqt.stat().st_size > 0,
            "sdf_exists": sdf.exists() and sdf.stat().st_size > 0,
            "vina_load_ok": False,
            "vina_load_error": "",
            "heavy_atom_count": None,
            "rotatable_bond_count": None,
            "input_extent_A": None,
            "warnings": [],
        }
        if check["sdf_exists"]:
            supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
            mol = supplier[0] if len(supplier) else None
            if mol is not None:
                check["heavy_atom_count"] = int(Chem.RemoveHs(mol).GetNumHeavyAtoms())
                check["rotatable_bond_count"] = int(Lipinski.NumRotatableBonds(Chem.RemoveHs(mol)))
                conf = mol.GetConformer()
                heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
                if heavy:
                    coords = np.asarray([conf.GetAtomPosition(idx) for idx in heavy], dtype=float)
                    check["input_extent_A"] = (coords.max(axis=0) - coords.min(axis=0)).tolist()
        if check["pdbqt_exists"]:
            try:
                vina = Vina(sf_name="vina", cpu=1, seed=SEED)
                vina.set_ligand_from_file(str(pdbqt))
                check["vina_load_ok"] = True
            except Exception as exc:
                check["vina_load_error"] = str(exc)
        if not check["pdbqt_exists"] or not check["sdf_exists"] or not check["vina_load_ok"]:
            check["warnings"].append("input_not_ready")
        checks.append(check)
    return checks


def main() -> None:
    for directory in (RESULTS, POSES, POSE_FILES, TOP_SDF, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    ready, box, step2_params = load_preflight()
    preflight = preflight_ligands(ready)
    (RESULTS / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if any(not item["vina_load_ok"] or not item["pdbqt_exists"] for item in preflight):
        raise RuntimeError("Preflight failed; no docking was started")

    all_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for row in ready.to_dict(orient="records"):
        compound_id = row["compound_id"]
        input_pdbqt, input_sdf = ligand_paths(compound_id)
        aggregate = POSES / f"{compound_id}.pdbqt"
        log_path = LOGS / f"{compound_id}.json"
        start = time.time()
        run: dict[str, Any] = {
            "compound_id": compound_id,
            "cluster_id": row["cluster_id"],
            "row_index": row["row_index"],
            "split": row["split"],
            "original_smiles": row["original_smiles"],
            "seed": SEED,
            "exhaustiveness": EXHAUSTIVENESS,
            "requested_poses": N_POSES,
            "status": "failed",
            "input_pdbqt": str(input_pdbqt.relative_to(ROOT)),
            "input_sdf": str(input_sdf.relative_to(ROOT)),
            "aggregate_pdbqt": str(aggregate.relative_to(ROOT)),
            "pose_count": 0,
            "poses": [],
            "error": "",
        }
        try:
            vina = Vina(sf_name="vina", cpu=CPU, seed=SEED)
            vina.set_receptor(str(RECEPTOR))
            vina.set_ligand_from_file(str(input_pdbqt))
            vina.compute_vina_maps(center=box["center"], box_size=box["size"])
            vina.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=N_POSES)
            vina.write_poses(str(aggregate), n_poses=N_POSES, energy_range=100.0, overwrite=True)
            energy_components = np.asarray(vina.energies(n_poses=N_POSES, energy_range=100.0), dtype=float)
            individual = split_vina_models(aggregate, POSE_FILES, compound_id)
            run["pose_count"] = len(individual)
            for rank, pose_path in enumerate(individual, 1):
                score, rmsd_lb, rmsd_ub = vina_result_fields(pose_path)
                top_sdf = ""
                if rank == 1:
                    sdf_path = TOP_SDF / f"{compound_id}.sdf"
                    export_log = convert_pdbqt_to_sdf(pose_path, sdf_path, ROOT)
                    top_sdf = str(sdf_path.relative_to(ROOT))
                pose_record = {
                    "compound_id": compound_id,
                    "cluster_id": row["cluster_id"],
                    "row_index": row["row_index"],
                    "split": row["split"],
                    "original_smiles": row["original_smiles"],
                    "preparation_status": row.get("step3_1_status", "ready"),
                    "docking_status": "success",
                    "seed": SEED,
                    "exhaustiveness": EXHAUSTIVENESS,
                    "pose_rank": rank,
                    "vina_score_kcal_mol": score,
                    "vina_rmsd_lb_A": rmsd_lb,
                    "vina_rmsd_ub_A": rmsd_ub,
                    "actual_pose_count": len(individual),
                    "aggregate_pdbqt": str(aggregate.relative_to(ROOT)),
                    "pose_pdbqt": str(pose_path.relative_to(ROOT)),
                    "top_pose_sdf": top_sdf,
                    "energy_components": json.dumps(energy_components[rank - 1].tolist() if rank <= len(energy_components) else []),
                    "failure_or_warning_reason": "",
                }
                run["poses"].append(pose_record)
                all_rows.append(pose_record)
            run["status"] = "success" if len(individual) else "failed"
            if not individual:
                run["error"] = "Vina returned no output poses"
            else:
                top_rows.append(run["poses"][0])
        except Exception as exc:
            run["error"] = str(exc)
            all_rows.append({
                "compound_id": compound_id,
                "cluster_id": row["cluster_id"],
                "row_index": row["row_index"],
                "split": row["split"],
                "original_smiles": row["original_smiles"],
                "preparation_status": row.get("step3_1_status", "ready"),
                "docking_status": "failed",
                "seed": SEED,
                "exhaustiveness": EXHAUSTIVENESS,
                "pose_rank": "",
                "vina_score_kcal_mol": "",
                "vina_rmsd_lb_A": "",
                "vina_rmsd_ub_A": "",
                "actual_pose_count": 0,
                "aggregate_pdbqt": str(aggregate.relative_to(ROOT)),
                "pose_pdbqt": "",
                "top_pose_sdf": "",
                "energy_components": "",
                "failure_or_warning_reason": str(exc),
            })
        run["runtime_s"] = time.time() - start
        log_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
        run_records.append(run)

    write_csv(RESULTS / "docking_all_poses.csv", all_rows)
    write_csv(RESULTS / "docking_top_poses.csv", top_rows)
    metadata = {
        "protocol": {
            "scoring_function": "vina",
            "receptor": str(RECEPTOR.relative_to(ROOT)),
            "box_file": str(BOX_FILE.relative_to(ROOT)),
            "center_A": box["center"].tolist(),
            "size_A": box["size"].tolist(),
            "exhaustiveness": EXHAUSTIVENESS,
            "seed": SEED,
            "cpu": CPU,
            "requested_poses": N_POSES,
            "output_energy_range": 100.0,
        },
        "versions": {"vina": version("vina"), "meeko": version("meeko"), "rdkit": __import__("rdkit").rdBase.rdkitVersion, "pandas": version("pandas")},
        "input": str(READY.relative_to(ROOT)),
        "input_count": len(ready),
        "preflight": preflight,
        "runs": run_records,
        "success_count": sum(run["status"] == "success" for run in run_records),
        "failure_count": sum(run["status"] != "success" for run in run_records),
        "docking_executed": True,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"input_count": len(ready), "success_count": metadata["success_count"], "failure_count": metadata["failure_count"], "pose_rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
