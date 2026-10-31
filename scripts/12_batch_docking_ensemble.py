"""Step 5.3: batch dock the existing 24 docking-ready ligands on Gate-2 pass receptors."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from vina import Vina

from ensemble_utils import ROOT, package_versions, read_csv_rows, write_csv
from redocking_utils import convert_pdbqt_to_sdf, read_box_config, split_vina_models, vina_result_fields


def passed_targets() -> list[str]:
    path = ROOT / "results" / "receptor_ensemble" / "native_redocking" / "gate2_summary.json"
    data = json.loads(path.read_text())
    return list(data.get("passed_targets", []))


def main() -> None:
    ready = read_csv_rows(ROOT / "results" / "ligand_preparation" / "docking_ready.csv")
    if len(ready) != 24:
        raise ValueError(f"Expected exactly 24 docking-ready compounds, found {len(ready)}")
    targets = passed_targets()
    all_summary = {"protocol": {"seed": 42, "exhaustiveness": 16, "n_poses": 10, "cpu": min(4, os.cpu_count() or 1), "docking_ready_count": len(ready)}, "versions": package_versions(), "targets": {}}
    for pdb_id in targets:
        prep = ROOT / "data" / "prepared" / pdb_id
        out = ROOT / "results" / "receptor_ensemble" / "batch_docking" / pdb_id
        poses = out / "poses"
        pose_files = out / "pose_files"
        sdfs = out / "top_sdf"
        poses.mkdir(parents=True, exist_ok=True); pose_files.mkdir(exist_ok=True); sdfs.mkdir(exist_ok=True)
        box = read_box_config(prep / "vina_box_ensemble.txt")
        receptor = prep / "receptor.pdbqt"
        cpu = min(4, os.cpu_count() or 1)
        top_rows, all_rows, failures = [], [], []
        for idx, item in enumerate(ready, 1):
            cid = item["compound_id"]
            ligand = ROOT / "data" / "ligands" / "pdbqt" / f"{cid}.pdbqt"
            started = time.time()
            try:
                vina = Vina(sf_name="vina", cpu=cpu, seed=42)
                vina.set_receptor(str(receptor)); vina.set_ligand_from_file(str(ligand))
                vina.compute_vina_maps(center=box["center"], box_size=box["size"])
                vina.dock(exhaustiveness=16, n_poses=10)
                aggregate = poses / f"{cid}.pdbqt"
                vina.write_poses(str(aggregate), n_poses=10, energy_range=100.0, overwrite=True)
                split = split_vina_models(aggregate, pose_files, 42)
                # split_vina_models names are seed_42_pose_XX; move to a
                # compound-specific directory to avoid collisions.
                compound_pose_dir = pose_files / cid
                compound_pose_dir.mkdir(exist_ok=True)
                renamed = []
                for pose_num, path in enumerate(split, 1):
                    new = compound_pose_dir / f"pose_{pose_num:02d}.pdbqt"
                    new.write_text(path.read_text()); path.unlink(); renamed.append(new)
                energies = np.asarray(vina.energies(n_poses=10, energy_range=100.0), dtype=float)
                for pose_num, path in enumerate(renamed, 1):
                    score, lb, ub = vina_result_fields(path)
                    row = {"receptor": pdb_id, "compound_id": cid, "cluster_id": item["cluster_id"], "row_index": item["row_index"], "seed": 42, "pose_rank": pose_num, "vina_score_kcal_mol": score, "vina_rmsd_lb_A": lb, "vina_rmsd_ub_A": ub, "pdbqt": str(path.relative_to(ROOT)), "energy_components": json.dumps(energies[pose_num - 1].tolist() if pose_num <= len(energies) else [])}
                    all_rows.append(row)
                    if pose_num == 1:
                        sdf = sdfs / f"{cid}.sdf"
                        row["sdf"] = str(sdf.relative_to(ROOT)); row["export_log"] = convert_pdbqt_to_sdf(path, sdf, ROOT)
                        top_rows.append(row.copy())
                print(f"{pdb_id}: {idx}/{len(ready)} {cid} {top_rows[-1]['vina_score_kcal_mol']:.3f}")
            except Exception as exc:
                failures.append({"compound_id": cid, "error": str(exc), "runtime_s": time.time() - started})
                print(f"{pdb_id}: FAILED {cid}: {exc}")
        write_csv(out / "docking_top_poses.csv", top_rows)
        write_csv(out / "docking_all_poses.csv", all_rows)
        (out / "docking_failures.json").write_text(json.dumps(failures, indent=2))
        target_summary = {"receptor": pdb_id, "compound_count": len(top_rows), "expected_count": len(ready), "failure_count": len(failures), "status": "gate3_candidate" if len(top_rows) == len(ready) and not failures else "gate3_failed", "top_poses": top_rows}
        (out / "batch_summary.json").write_text(json.dumps(target_summary, indent=2))
        all_summary["targets"][pdb_id] = target_summary
    (ROOT / "results" / "receptor_ensemble" / "batch_summary.json").write_text(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()

