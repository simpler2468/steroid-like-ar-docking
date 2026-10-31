"""Run Testosterone redocking against the repaired 2AM9 receptor."""

from __future__ import annotations

import json
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from vina import Vina

from prepare_utils import build_tes_mol, read_tes_from_pdb
from redocking_utils import convert_pdbqt_to_sdf, read_box_config, split_vina_models, vina_result_fields, write_csv


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
RESULTS = ROOT / "results" / "redocking_2AM9"
POSES = RESULTS / "poses"
SEEDS = [42, 123, 2026, 7, 99]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    POSES.mkdir(parents=True, exist_ok=True)
    paths = {
        "receptor": PREP / "receptor_repaired.pdbqt",
        "ligand": PREP / "TES.pdbqt",
        "crystal": PREP / "TES_crystal.sdf",
        "box": PREP / "vina_box_repaired.txt",
        "receptor_pdb": PREP / "receptor_repaired.pdb",
    }
    if not all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        raise FileNotFoundError(f"Missing Step 1 input: {paths}")
    box = read_box_config(paths["box"])
    if Path(box["receptor"]).name != paths["receptor"].name:
        raise ValueError(f"Box receptor entry does not match repaired receptor: {box['receptor']}")

    # Parse/validate the input reference without rewriting it.
    _, tes_coords = read_tes_from_pdb(ROOT / "data" / "PDB" / "2AM9.pdb")
    crystal_mol, tes_meta = build_tes_mol(ROOT / "data" / "PDB" / "2AM9.pdb", PREP / "TES.cif")
    if len(tes_coords) != 21 or tes_meta["formula"] != "C19H28O2":
        raise ValueError("TES reference validation failed before docking")

    cpu = min(4, os.cpu_count() or 1)
    parameters = {
        "scoring_function": "vina",
        "exhaustiveness": 16,
        "n_poses": 10,
        "output_energy_range": 100.0,
        "seeds": SEEDS,
        "cpu": cpu,
        "center_A": box["center"].tolist(),
        "size_A": box["size"].tolist(),
        "box_file": str(paths["box"].relative_to(ROOT)),
        "receptor": str(paths["receptor"].relative_to(ROOT)),
        "ligand": str(paths["ligand"].relative_to(ROOT)),
    }
    metadata = {
        "parameters": parameters,
        "versions": {package: version(package) for package in ["vina", "meeko", "rdkit", "gemmi"]},
        "runs": [],
    }
    all_score_rows = []
    for seed in SEEDS:
        start = time.time()
        vina = Vina(sf_name="vina", cpu=cpu, seed=seed)
        vina.set_receptor(str(paths["receptor"]))
        vina.set_ligand_from_file(str(paths["ligand"]))
        vina.compute_vina_maps(center=box["center"], box_size=box["size"])
        vina.dock(exhaustiveness=16, n_poses=10)
        aggregate = POSES / f"seed_{seed}.pdbqt"
        # Vina 1.2.7 defaults to energy_range=3.0, which can silently write
        # fewer than n_poses even when docking generated ten modes. Use a
        # wide range so every requested ranked pose is retained.
        vina.write_poses(str(aggregate), n_poses=10, energy_range=100.0, overwrite=True)
        energies = np.asarray(vina.energies(n_poses=10, energy_range=100.0), dtype=float)
        pose_pdbqts = split_vina_models(aggregate, POSES, seed)
        pose_rows = []
        for pose_index, pose_path in enumerate(pose_pdbqts, 1):
            sdf_path = POSES / f"seed_{seed}_pose_{pose_index:02d}.sdf"
            export_log = convert_pdbqt_to_sdf(pose_path, sdf_path, ROOT)
            score, rmsd_lb, rmsd_ub = vina_result_fields(pose_path)
            energy_components = energies[pose_index - 1].tolist() if pose_index <= len(energies) else []
            pose_rows.append({
                "seed": seed,
                "pose": pose_index,
                "vina_score_kcal_mol": score,
                "rmsd_lb_A": rmsd_lb,
                "rmsd_ub_A": rmsd_ub,
                "energy_components": energy_components,
                "pdbqt": str(pose_path.relative_to(ROOT)),
                "sdf": str(sdf_path.relative_to(ROOT)),
                "export_log": export_log[-500:],
            })
            all_score_rows.append(pose_rows[-1])
        run = {
            "seed": seed,
            "runtime_s": time.time() - start,
            "aggregate_pdbqt": str(aggregate.relative_to(ROOT)),
            "poses": pose_rows,
        }
        metadata["runs"].append(run)
        (RESULTS / f"seed_{seed}.json").write_text(json.dumps(run, indent=2))

    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_csv(RESULTS / "vina_scores.csv", all_score_rows)
    # The independent validator will create redocking_results.csv and the report.
    print(json.dumps({"results": str(RESULTS), "seeds": SEEDS, "cpu": cpu, "poses": len(all_score_rows)}, indent=2))


if __name__ == "__main__":
    main()

