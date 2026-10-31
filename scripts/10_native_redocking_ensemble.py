"""Step 5.2: native-ligand redocking for receptors passing Gate 1."""

from __future__ import annotations

import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from vina import Vina

from ensemble_utils import ROOT, crystal_rmsd, package_versions, write_csv
from redocking_utils import convert_pdbqt_to_sdf, read_box_config, split_vina_models, vina_result_fields


SEEDS = [42, 123, 2026, 7, 99]


def eligible_targets() -> list[str]:
    out = []
    for pdb_id in ("2AMA", "2AMB"):
        p = ROOT / "data" / "prepared" / pdb_id
        if (p / "preparation_summary.json").exists() and (p / "receptor.pdbqt").exists():
            out.append(pdb_id)
    return out


def run_one(pdb_id: str) -> dict:
    prep = ROOT / "data" / "prepared" / pdb_id
    result = ROOT / "results" / "receptor_ensemble" / "native_redocking" / pdb_id
    poses = result / "poses"
    poses.mkdir(parents=True, exist_ok=True)
    receptor = prep / "receptor.pdbqt"
    ligand = prep / "native_aligned.pdbqt"
    crystal = prep / "native_crystal_aligned.sdf"
    box = read_box_config(prep / "vina_box_ensemble.txt")
    ref = __import__("ensemble_utils").load_single_sdf(crystal)
    cpu = min(4, os.cpu_count() or 1)
    rows: list[dict] = []
    runs: list[dict] = []
    for seed in SEEDS:
        started = time.time()
        vina = Vina(sf_name="vina", cpu=cpu, seed=seed)
        vina.set_receptor(str(receptor))
        vina.set_ligand_from_file(str(ligand))
        vina.compute_vina_maps(center=box["center"], box_size=box["size"])
        # This is the full global search API call. No reference pose is
        # supplied and no optimize()/local-only step is used.
        vina.dock(exhaustiveness=16, n_poses=10)
        aggregate = poses / f"seed_{seed}.pdbqt"
        vina.write_poses(str(aggregate), n_poses=10, energy_range=100.0, overwrite=True)
        pose_files = split_vina_models(aggregate, poses, seed)
        run_rows = []
        for idx, pose_path in enumerate(pose_files, 1):
            sdf = poses / f"seed_{seed}_pose_{idx:02d}.sdf"
            export_log = convert_pdbqt_to_sdf(pose_path, sdf, ROOT)
            pred = __import__("ensemble_utils").load_single_sdf(sdf)
            rmsd = crystal_rmsd(ref, pred)
            score, rmsd_lb, rmsd_ub = vina_result_fields(pose_path)
            row = {"receptor": pdb_id, "seed": seed, "pose": idx, "vina_score_kcal_mol": score, "vina_rmsd_lb_A": rmsd_lb, "vina_rmsd_ub_A": rmsd_ub, "crystal_rmsd_A": rmsd["rmsd_A"], "mapping_count": rmsd["mapping_count"], "mapping_chirality_used": rmsd["mapping_chirality_used"], "pdbqt": str(pose_path.relative_to(ROOT)), "sdf": str(sdf.relative_to(ROOT)), "export_log_tail": export_log[-300:]}
            rows.append(row)
            run_rows.append(row)
        runs.append({"seed": seed, "runtime_s": time.time() - started, "top_score_kcal_mol": run_rows[0]["vina_score_kcal_mol"], "top_crystal_rmsd_A": run_rows[0]["crystal_rmsd_A"], "min_crystal_rmsd_A": min(x["crystal_rmsd_A"] for x in run_rows), "poses": run_rows})
    write_csv(result / "native_redocking_all_poses.csv", rows)
    summary = {"receptor": pdb_id, "protocol": {"scoring_function": "vina", "exhaustiveness": 16, "n_poses": 10, "seeds": SEEDS, "cpu": cpu, "receptor": str(receptor.relative_to(ROOT)), "ligand": str(ligand.relative_to(ROOT)), "crystal_reference": str(crystal.relative_to(ROOT)), "box": str((prep / "vina_box_ensemble.txt").relative_to(ROOT)), "rmsd_method": "direct coordinate RMSD in aligned 2AM9 frame; graph-based chemical mapping; no fitting"}, "versions": package_versions(), "runs": runs, "status": "completed"}
    (result / "native_redocking_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    results = {}
    for target in eligible_targets():
        results[target] = run_one(target)
        print(json.dumps({"receptor": target, "runs": len(results[target]["runs"])}, indent=2))
    out = ROOT / "results" / "receptor_ensemble" / "native_redocking_summary.json"
    out.write_text(json.dumps({"targets": results, "skipped": [x for x in ("2AMA", "2AMB") if x not in results]}, indent=2))


if __name__ == "__main__":
    main()

