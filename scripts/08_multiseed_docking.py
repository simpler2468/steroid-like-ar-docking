#!/usr/bin/env python3
"""Step 4.4 multi-seed docking for the five predefined compounds only.

Every run uses Vina.dock(), never optimize(), and writes to a new independent
results directory. Existing Step 3.2 outputs are read but never overwritten.
"""

from __future__ import annotations

import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from vina import Vina

from redocking_utils import convert_pdbqt_to_sdf, read_box_config, split_vina_models, vina_result_fields, write_csv

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
READY = ROOT / "results" / "ligand_preparation" / "docking_ready.csv"
OUT = ROOT / "results" / "orientation_robustness"
POSE_DIR = OUT / "docking_poses"
LOG_DIR = OUT / "docking_logs"

TARGETS = ["cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]
SEEDS = [42, 123, 2026, 7, 99, 11, 17, 73, 101, 314]
RECEPTOR = PREP / "receptor_repaired.pdbqt"
BOX = PREP / "vina_box_repaired.txt"
CPU = min(4, os.cpu_count() or 1)
EXHAUSTIVENESS = 16
N_POSES = 10


def main():
    for d in [OUT, POSE_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    ready = pd.read_csv(READY, dtype=object, keep_default_na=False)
    target_rows = ready[ready.compound_id.isin(TARGETS)].copy()
    if set(target_rows.compound_id) != set(TARGETS) or len(target_rows) != len(TARGETS):
        raise ValueError("The five requested compounds are not exactly present in docking_ready.csv")
    box = read_box_config(BOX)
    if Path(box["receptor"]).name != RECEPTOR.name:
        raise ValueError("Box receptor entry does not match repaired receptor")
    input_ligands = {}
    for cid in TARGETS:
        p = ROOT / "data" / "ligands" / "pdbqt" / f"{cid}.pdbqt"
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(p)
        input_ligands[cid] = p

    all_rows = []
    run_records = []
    for cid in TARGETS:
        compound_dir = POSE_DIR / cid
        compound_dir.mkdir(parents=True, exist_ok=True)
        row = target_rows.loc[target_rows.compound_id == cid].iloc[0].to_dict()
        for seed in SEEDS:
            start = time.time()
            seed_dir = compound_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            aggregate = seed_dir / "aggregate.pdbqt"
            log_path = LOG_DIR / f"{cid}__seed_{seed}.json"
            run = {"compound_id": cid, "cluster_id": int(row["cluster_id"]), "row_index": int(row["row_index"]), "seed": seed, "method": "Vina.dock", "status": "failed", "receptor": str(RECEPTOR.relative_to(ROOT)), "ligand": str(input_ligands[cid].relative_to(ROOT)), "box_file": str(BOX.relative_to(ROOT)), "exhaustiveness": EXHAUSTIVENESS, "cpu": CPU, "requested_poses": N_POSES, "aggregate_pdbqt": str(aggregate.relative_to(ROOT)), "poses": [], "error": ""}
            try:
                vina = Vina(sf_name="vina", cpu=CPU, seed=seed)
                vina.set_receptor(str(RECEPTOR))
                vina.set_ligand_from_file(str(input_ligands[cid]))
                vina.compute_vina_maps(center=box["center"], box_size=box["size"])
                # Full global search is intentional; no optimize()/local search.
                vina.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=N_POSES)
                vina.write_poses(str(aggregate), n_poses=N_POSES, energy_range=100.0, overwrite=True)
                energies = np.asarray(vina.energies(n_poses=N_POSES, energy_range=100.0), dtype=float)
                pose_paths = split_vina_models(aggregate, seed_dir, f"{cid}__seed_{seed}")
                for rank, pose_path in enumerate(pose_paths, 1):
                    sdf_path = seed_dir / f"pose_{rank:02d}.sdf"
                    export_log = convert_pdbqt_to_sdf(pose_path, sdf_path, ROOT)
                    score, rmsd_lb, rmsd_ub = vina_result_fields(pose_path)
                    rec = {"compound_id": cid, "cluster_id": int(row["cluster_id"]), "row_index": int(row["row_index"]), "seed": seed, "pose_rank": rank, "vina_score_kcal_mol": score, "vina_rmsd_lb_A": rmsd_lb, "vina_rmsd_ub_A": rmsd_ub, "actual_pose_count": len(pose_paths), "aggregate_pdbqt": str(aggregate.relative_to(ROOT)), "pose_pdbqt": str(pose_path.relative_to(ROOT)), "pose_sdf": str(sdf_path.relative_to(ROOT)), "energy_components": json.dumps(energies[rank - 1].tolist() if rank <= len(energies) else []), "export_log_tail": export_log[-300:]}
                    run["poses"].append(rec)
                    all_rows.append(rec)
                run["status"] = "success" if pose_paths else "failed"
                if not pose_paths:
                    run["error"] = "Vina returned no poses"
            except Exception as exc:
                run["error"] = repr(exc)
            run["runtime_s"] = time.time() - start
            log_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
            run_records.append(run)

    write_csv(OUT / "multi_seed_all_poses.csv", all_rows)
    metadata = {"protocol": {"scoring_function": "vina", "receptor": str(RECEPTOR.relative_to(ROOT)), "box_file": str(BOX.relative_to(ROOT)), "center_A": box["center"].tolist(), "size_A": box["size"].tolist(), "exhaustiveness": EXHAUSTIVENESS, "cpu": CPU, "requested_poses": N_POSES, "seeds": SEEDS, "method": "Vina.dock (full search); optimize not used"}, "versions": {p: version(p) for p in ["vina", "meeko", "rdkit"]}, "target_compounds": TARGETS, "runs": run_records, "success_count": sum(r["status"] == "success" for r in run_records), "failure_count": sum(r["status"] != "success" for r in run_records), "docking_executed": True}
    (OUT / "multiseed_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "success" if metadata["failure_count"] == 0 else "partial", "target_count": len(TARGETS), "seed_count": len(SEEDS), "run_count": len(run_records), "success_count": metadata["success_count"], "failure_count": metadata["failure_count"], "pose_rows": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
