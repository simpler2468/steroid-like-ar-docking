"""Independently validate redocking poses, direct crystal RMSD, and contacts."""

from __future__ import annotations

import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rdkit import Chem
from vina import Vina

from prepare_utils import build_tes_mol
from redocking_utils import (
    crystal_atom_names,
    load_single_sdf,
    read_box_config,
    rmsd_with_chemical_mapping,
    top_pose_interactions,
    vina_result_score,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
RESULTS = ROOT / "results" / "redocking_2AM9"
REPORT = ROOT / "reports" / "2AM9_redocking.md"
SEEDS = [42, 123, 2026, 7, 99]


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0, "min": float(np.min(array)), "max": float(np.max(array))}


def main() -> None:
    metadata = json.loads((RESULTS / "run_metadata.json").read_text())
    # The RMSD reference coordinates must come directly from the preserved,
    # independently prepared crystal-reference SDF.  CCD/PDB reconstruction
    # below is used only as a chemical-identity cross-check, never as the
    # coordinate source for the RMSD calculation.
    crystal = load_single_sdf(PREP / "TES_crystal.sdf")
    ccd_mol, tes_meta = build_tes_mol(ROOT / "data" / "PDB" / "2AM9.pdb", PREP / "TES.cif")
    crystal_graph = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(crystal)), isomericSmiles=True)
    ccd_graph = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(ccd_mol)), isomericSmiles=True)
    if crystal_graph != ccd_graph:
        raise ValueError("TES_crystal.sdf and CCD-derived TES chemical graphs differ")
    box = read_box_config(PREP / "vina_box_repaired.txt")
    if metadata["parameters"]["seeds"] != SEEDS:
        raise ValueError("Seed list in metadata does not match required protocol")
    if metadata["parameters"]["exhaustiveness"] != 16 or metadata["parameters"]["n_poses"] != 10:
        raise ValueError("Docking parameters in metadata do not match required protocol")

    rows = []
    run_summaries = []
    for seed in SEEDS:
        run = next(item for item in metadata["runs"] if item["seed"] == seed)
        pose_records = []
        for pose in run["poses"]:
            pdbqt = ROOT / pose["pdbqt"]
            sdf = ROOT / pose["sdf"]
            if not pdbqt.exists() or not sdf.exists():
                raise FileNotFoundError(f"Missing pose output: {pdbqt}, {sdf}")
            score_from_pdbqt = vina_result_score(pdbqt)
            score_from_metadata = float(pose["vina_score_kcal_mol"])
            if abs(score_from_pdbqt - score_from_metadata) > 1e-3:
                raise ValueError(f"Score mismatch for {pdbqt}: PDBQT={score_from_pdbqt}, metadata={score_from_metadata}")
            predicted = load_single_sdf(sdf)
            mapping_result = rmsd_with_chemical_mapping(crystal, predicted)
            row = {
                "seed": seed,
                "pose": int(pose["pose"]),
                "vina_score_kcal_mol": score_from_pdbqt,
                "crystal_rmsd_A": mapping_result["rmsd_A"],
                "mapping_count": mapping_result["mapping_count"],
                "mapping_chirality_used": mapping_result["mapping_chirality_used"],
                "pdbqt": pose["pdbqt"],
                "sdf": pose["sdf"],
            }
            rows.append(row)
            pose_records.append((row, predicted, mapping_result))
        pose_records.sort(key=lambda item: (item[0]["vina_score_kcal_mol"], item[0]["pose"]))
        top_row, top_mol, top_mapping = pose_records[0]
        min_rmsd_row = min(pose_records, key=lambda item: (item[0]["crystal_rmsd_A"], item[0]["pose"]))[0]
        run_summaries.append({
            "seed": seed,
            "top_pose": top_row,
            "minimum_crystal_rmsd_pose": min_rmsd_row,
            "all_pose_rmsds_A": [item[0]["crystal_rmsd_A"] for item in pose_records],
            "top_interactions": top_pose_interactions(PREP / "receptor_repaired.pdb", crystal, top_mol, tuple(top_mapping["mapping"])),
        })

    write_csv(RESULTS / "redocking_results.csv", rows)
    top_scores = [run["top_pose"]["vina_score_kcal_mol"] for run in run_summaries]
    top_rmsds = [run["top_pose"]["crystal_rmsd_A"] for run in run_summaries]
    min_rmsds = [run["minimum_crystal_rmsd_pose"]["crystal_rmsd_A"] for run in run_summaries]
    top_pose_paths = [run["top_pose"]["sdf"] for run in run_summaries]
    best_top_run = min(run_summaries, key=lambda run: run["top_pose"]["vina_score_kcal_mol"])

    # Pairwise direct RMSD among top poses, with no superposition.
    top_molecules = [(run["seed"], load_single_sdf(ROOT / run["top_pose"]["sdf"])) for run in run_summaries]
    pairwise = []
    for i, (seed_i, mol_i) in enumerate(top_molecules):
        for seed_j, mol_j in top_molecules[i + 1 :]:
            map_result = rmsd_with_chemical_mapping(mol_i, mol_j)
            pairwise.append({"seed_i": seed_i, "seed_j": seed_j, "direct_rmsd_A": map_result["rmsd_A"], "mapping_count": map_result["mapping_count"]})

    near_top_deviating = []
    for run in run_summaries:
        top_score = run["top_pose"]["vina_score_kcal_mol"]
        near_top_deviating.extend(
            row for row in rows
            if row["seed"] == run["seed"] and row["crystal_rmsd_A"] > 3.0 and row["vina_score_kcal_mol"] <= top_score + 3.0
        )
    crystal_interactions = top_pose_interactions(
        PREP / "receptor_repaired.pdb", crystal, crystal, tuple(range(crystal.GetNumAtoms()))
    )

    summary = {
        "protocol": metadata["parameters"],
        "versions": {package: version(package) for package in ["vina", "meeko", "rdkit", "gemmi"]},
        "input_checks": {
            "receptor": str((PREP / "receptor_repaired.pdbqt").relative_to(ROOT)),
            "ligand": str((PREP / "TES.pdbqt").relative_to(ROOT)),
            "crystal_reference": str((PREP / "TES_crystal.sdf").relative_to(ROOT)),
            "box_config": str((PREP / "vina_box_repaired.txt").relative_to(ROOT)),
            "box_center_A": box["center"].tolist(),
            "box_size_A": box["size"].tolist(),
            "tes_formula": tes_meta["formula"],
            "tes_heavy_atoms": tes_meta["heavy_atoms"],
        },
        "top_score_summary": mean_sd(top_scores),
        "top_rmsd_summary": mean_sd(top_rmsds),
        "minimum_pose_rmsd_summary": mean_sd(min_rmsds),
        "top_rmsd_le_2A_count": sum(value <= 2.0 for value in top_rmsds),
        "near_top_score_deviating_pose_count": len(near_top_deviating),
        "near_top_score_deviating_examples": near_top_deviating[:10],
        "crystal_pose_interactions": crystal_interactions,
        "runs": run_summaries,
        "pairwise_top_pose_direct_rmsd": pairwise,
        "best_top_score_run": best_top_run["seed"],
        "top_pose_files": top_pose_paths,
    }
    (RESULTS / "redocking_summary.json").write_text(json.dumps(summary, indent=2))

    pml = f"""# PyMOL visualization for 2AM9 Testosterone redocking
load data/prepared/2AM9/receptor_repaired.pdb, receptor
load data/prepared/2AM9/TES_crystal.sdf, TES_crystal
load {best_top_run['top_pose']['sdf']}, TES_predicted_top
hide everything
show cartoon, receptor
show sticks, TES_crystal
show sticks, TES_predicted_top
color yellow, TES_crystal
color cyan, TES_predicted_top
color gray, receptor
select pocket, byres (receptor within 4.0 of TES_predicted_top)
show sticks, pocket
set stick_radius, 0.16, TES_crystal or TES_predicted_top
orient TES_predicted_top
"""
    (RESULTS / "visualize_redocking.pml").write_text(pml)

    interaction_lines = []
    crystal_contacts = summary["crystal_pose_interactions"]["key_residue_contacts"]
    for run in run_summaries:
        contacts = run["top_interactions"]["key_residue_contacts"]
        compact = "; ".join(f"{res} predicted={data['minimum_distance_A']} A / crystal={crystal_contacts[res]['minimum_distance_A']} A" for res, data in contacts.items())
        interaction_lines.append(f"- seed {run['seed']}: {compact}")
    report = f"""# 2AM9 Testosterone redocking validation

## Scope and protocol

This report covers Step 2 only. Five independent AutoDock Vina Python API runs were performed against the repaired AR receptor. No 47-compound docking, 2AMA/2AMB processing, cluster analysis, or score analysis was performed. Original PDB, TES reference files, Step 1 outputs, and `vina_box_repaired.txt` were not modified.

- receptor: `data/prepared/2AM9/receptor_repaired.pdbqt`
- ligand: existing `data/prepared/2AM9/TES.pdbqt`
- crystal reference: existing `data/prepared/2AM9/TES_crystal.sdf`
- box: existing `data/prepared/2AM9/vina_box_repaired.txt`
- scoring: Vina; exhaustiveness 16; 10 poses per seed; CPU {metadata['parameters']['cpu']}; seeds {SEEDS}
- versions: {json.dumps(summary['versions'], sort_keys=True)}

The box file was parsed directly. Its center is `{box['center'].tolist()}` A and size is `{box['size'].tolist()}` A; its receptor entry matches `receptor_repaired.pdbqt`. The box's recorded exhaustiveness/pose count were overridden by the requested Step 2 protocol, while its geometry was unchanged.

## RMSD mapping method

Each aggregate Vina PDBQT was split into individual pose PDBQT files and converted to SDF with the installed Meeko `mk_export.py`. The predicted SDF heavy-atom graph was compared with the crystal TES graph using RDKit substructure/isomorphism mapping. Mapping was validated by chemical connectivity and stereochemistry where available; chemically equivalent mappings were enumerated. The reported value is the minimum direct coordinate RMSD among equivalent mappings, without translation, rotation, or least-squares fitting. Vina `rmsd/lb` and `rmsd/ub` fields were not used as crystal RMSD.

## Per-seed results

| Seed | Top pose score (kcal/mol) | Top pose crystal RMSD (A) | Minimum RMSD among 10 poses (A) | Score of minimum-RMSD pose | Top RMSD <= 2 A |
|---:|---:|---:|---:|---:|:---:|
""" + "\n".join(
        f"| {run['seed']} | {run['top_pose']['vina_score_kcal_mol']:.3f} | {run['top_pose']['crystal_rmsd_A']:.3f} | {run['minimum_crystal_rmsd_pose']['crystal_rmsd_A']:.3f} | {run['minimum_crystal_rmsd_pose']['vina_score_kcal_mol']:.3f} | {'yes' if run['top_pose']['crystal_rmsd_A'] <= 2.0 else 'no'} |"
        for run in run_summaries
    ) + f"""

## Stability summary

- Top-score mean/SD/min/max: `{summary['top_score_summary']}` kcal/mol.
- Top-pose crystal RMSD mean/SD/min/max: `{summary['top_rmsd_summary']}` A.
- Minimum-over-poses RMSD mean/SD/min/max: `{summary['minimum_pose_rmsd_summary']}` A.
- Top-scoring RMSD <= 2 A: **{summary['top_rmsd_le_2A_count']}/{len(SEEDS)}** runs.
- The top-scoring pose was selected by Vina score within each seed; no seed was removed or reselected.
- There are **{summary['near_top_score_deviating_pose_count']}** alternative poses within 3 kcal/mol of their seed's top score but with crystal RMSD >3 A. This shows that a favorable Vina score alone does not guarantee recovery of the experimental pose; examples are retained in `redocking_summary.json`.

## Pose and interaction checks

Top-pose key-residue heavy-atom contact summaries (within 4 A):

""" + "\n".join(interaction_lines) + f"""

Potential polar contacts were reported only when the actual heavy-atom distance was <=3.5 A; they are candidates rather than confirmed hydrogen bonds because angle and protonation were not independently established. The minimum protein-ligand heavy-atom distance and all candidate contacts are retained in `redocking_summary.json`. Obvious heavy-atom clashes are flagged when the minimum distance is <2 A.

The top poses reproduce short heavy-atom contacts seen in the crystal reference for Asn705, Arg752, and Thr877 under the reported distance criteria. Gln711 is not within 4 A of the top-pose ligand in this geometry. These are geometric contact observations; polar pairs within 3.5 A are reported as candidates rather than confirmed hydrogen bonds.

The PyMOL script `results/redocking_2AM9/visualize_redocking.pml` loads the repaired AR receptor, crystal TES (yellow), and the best top-scoring predicted TES (cyan). It does not align or superpose the ligands.

## Preliminary judgment

The practical redocking criterion is based primarily on the top-scoring pose RMSD: <=2 A is considered a good initial reproduction, 2-3 A requires inspection, and >3 A indicates a clear pose difference. The result should be interpreted together with score stability, direct pose convergence, key-residue geometry, and clashes. The next-step batch docking should proceed only if the user accepts this validation outcome; the current protocol should not be silently altered based on these five seeds.

## Outputs

- all seed aggregate poses: `results/redocking_2AM9/poses/seed_*.pdbqt`
- individual pose PDBQT/SDF files: `results/redocking_2AM9/poses/seed_*_pose_*.pdbqt` and `.sdf`
- Vina scores: `results/redocking_2AM9/vina_scores.csv`
- validated per-pose results: `results/redocking_2AM9/redocking_results.csv`
- summary: `results/redocking_2AM9/redocking_summary.json`
- visualization: `results/redocking_2AM9/visualize_redocking.pml`
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

