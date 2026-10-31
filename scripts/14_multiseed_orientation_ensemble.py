"""Step 5.4: 10-seed C17 orientation robustness on the Gate-2 receptors."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from vina import Vina

from ensemble_utils import ROOT, crystal_rmsd, load_single_sdf, package_versions, write_csv
from redocking_utils import convert_pdbqt_to_sdf, read_box_config, split_vina_models, vina_result_fields


SEEDS = [42, 123, 2026, 7, 99, 11, 17, 73, 101, 314]
TARGETS = ["cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]
GROUPS = {"A_ethynyl": {"cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"}}


def load_step44():
    path = ROOT / "scripts" / "08_orientation_verification.py"
    spec = importlib.util.spec_from_file_location("step44_orientation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def heavy(mol):
    return Chem.RemoveHs(Chem.Mol(mol))


def group_for(cid: str) -> str:
    if cid in {"cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378"}:
        return "A_ethynyl"
    if cid in {"cluster_3_row_2412", "cluster_246_row_5379"}:
        return "B_long_ester"
    return "unknown"


def copy_seed42(out: Path, cid: str) -> list[Path]:
    source = ROOT / "results" / "receptor_ensemble" / "batch_docking" / "2AMA" / "pose_files" / cid
    target = out / "pose_files" / cid / "seed_42"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for src in sorted(source.glob("pose_*.pdbqt")):
        dst = target / src.name
        shutil.copy2(src, dst)
        paths.append(dst)
    return paths


def run_seed(out: Path, cid: str, seed: int, receptor: Path, ligand: Path, box: dict, cpu: int) -> list[Path]:
    target = out / "pose_files" / cid / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    vina = Vina(sf_name="vina", cpu=cpu, seed=seed)
    vina.set_receptor(str(receptor)); vina.set_ligand_from_file(str(ligand))
    vina.compute_vina_maps(center=box["center"], box_size=box["size"])
    vina.dock(exhaustiveness=16, n_poses=10)
    aggregate = target / "aggregate.pdbqt"
    vina.write_poses(str(aggregate), n_poses=10, energy_range=100.0, overwrite=True)
    split = split_vina_models(aggregate, target, seed)
    out_paths = []
    for i, src in enumerate(split, 1):
        dst = target / f"pose_{i:02d}.pdbqt"
        if src != dst:
            shutil.move(str(src), str(dst))
        out_paths.append(dst)
    return out_paths


def main() -> None:
    gate = json.loads((ROOT / "results" / "receptor_ensemble" / "native_redocking" / "gate2_summary.json").read_text())
    receptors = gate.get("passed_targets", [])
    out_root = ROOT / "results" / "receptor_ensemble" / "multiseed_orientation"
    out_root.mkdir(parents=True, exist_ok=True)
    step44 = load_step44()
    tes = heavy(load_single_sdf(ROOT / "data" / "prepared" / "2AM9" / "TES_crystal.sdf"))
    tes_anchor = step44.find_anchor_atoms(tes)
    tes_c3, tes_c17 = step44.xyz(tes, tes_anchor["C3"]), step44.xyz(tes, tes_anchor["C17"])
    tes_u = tes_c17 - tes_c3
    ref_5378 = heavy(load_single_sdf(ROOT / "results" / "docking_2AM9" / "top_sdf" / "cluster_176_row_5378.sdf"))
    ref_anchor = step44.find_anchor_atoms(ref_5378)
    receptor_all = {}
    for pdb_id in receptors:
        prep = ROOT / "data" / "prepared" / pdb_id
        out = out_root / pdb_id
        out.mkdir(parents=True, exist_ok=True)
        receptor = step44.parse_pdb(prep / "receptor_aligned.pdb")
        receptor_all[pdb_id] = receptor
        box = read_box_config(prep / "vina_box_ensemble.txt")
        cpu = min(4, os.cpu_count() or 1)
        all_rows, top_rows, run_meta = [], [], []
        anchors, mappings = {}, {}
        for cid in TARGETS:
            sample = heavy(load_single_sdf(ROOT / "results" / "docking_2AM9" / "top_sdf" / f"{cid}.sdf"))
            anchors[cid] = step44.find_anchor_atoms(sample)
            mcs, pairs = step44.mcs_map(tes, sample)
            mappings[cid] = {"pairs": pairs, "mcs_atoms": mcs.numAtoms, "status": "validated" if len(pairs) >= 17 and dict(pairs).get(tes_anchor["C3"]) == anchors[cid]["C3"] and dict(pairs).get(tes_anchor["C17"]) == anchors[cid]["C17"] else "ambiguous"}
            if mappings[cid]["status"] != "validated":
                raise RuntimeError(f"C3/C17 mapping failed for {cid}")
        for cid in TARGETS:
            ligand = ROOT / "data" / "ligands" / "pdbqt" / f"{cid}.pdbqt"
            for seed in SEEDS:
                started = time.time()
                if seed == 42:
                    pose_paths = copy_seed42(out, cid)
                    source = "reused_gate3_seed42_exact_protocol"
                else:
                    pose_paths = run_seed(out, cid, seed, prep / "receptor.pdbqt", ligand, box, cpu)
                    source = "new_full_dock"
                pose_rows = []
                for rank, pose in enumerate(pose_paths, 1):
                    sdf = pose.with_suffix(".sdf")
                    convert_pdbqt_to_sdf(pose, sdf, ROOT)
                    mol = heavy(load_single_sdf(sdf))
                    a = anchors[cid]
                    c3, c17 = step44.xyz(mol, a["C3"]), step44.xyz(mol, a["C17"])
                    d_same = float(np.linalg.norm(c3 - tes_c3) + np.linalg.norm(c17 - tes_c17))
                    d_swap = float(np.linalg.norm(c3 - tes_c17) + np.linalg.norm(c17 - tes_c3))
                    angle = step44.angle_deg(c17 - c3, tes_u)
                    # Same local pocket metric used in Step 4.4, with the
                    # aligned receptor as the actual coordinate source.
                    region = {}
                    for end, point in (("C3", c3), ("C17", c17)):
                        region[f"{end}_A_region_min_A"] = min((float(np.linalg.norm(point - aa["coord"])) for key in ["A:ARG752", "A:GLN711"] for aa in step44.key_atoms(receptor, key)), default=np.nan)
                        region[f"{end}_D_region_min_A"] = min((float(np.linalg.norm(point - aa["coord"])) for key in ["A:ASN705", "A:THR877"] for aa in step44.key_atoms(receptor, key)), default=np.nan)
                    native_region_score = (region["C3_D_region_min_A"] - region["C3_A_region_min_A"]) + (region["C17_A_region_min_A"] - region["C17_D_region_min_A"])
                    reversed_region_score = -native_region_score
                    classification = step44.classify(angle, d_same, d_swap, native_region_score, reversed_region_score)
                    score, lb, ub = vina_result_fields(pose)
                    row = {"receptor": pdb_id, "compound_id": cid, "row_index": int(cid.rsplit("_", 1)[1]), "group": group_for(cid), "seed": seed, "pose_rank": rank, "vina_score_kcal_mol": score, "vina_rmsd_lb_A": lb, "vina_rmsd_ub_A": ub, "c3_c17_angle_to_TES_deg": angle, "d_same_A": d_same, "d_swap_A": d_swap, "native_region_score": native_region_score, "reversed_region_score": reversed_region_score, "orientation_class": classification, "pdbqt": str(pose.relative_to(ROOT)), "sdf": str(sdf.relative_to(ROOT)), "seed_source": source}
                    all_rows.append(row); pose_rows.append(row)
                top_rows.append(min(pose_rows, key=lambda x: x["vina_score_kcal_mol"]))
                run_meta.append({"compound_id": cid, "seed": seed, "source": source, "pose_count": len(pose_rows), "runtime_s": time.time() - started})
        write_csv(out / "orientation_all_poses.csv", all_rows)
        write_csv(out / "orientation_top_poses.csv", top_rows)
        write_csv(out / "competitive_pose_summary.csv", competitive_rows(all_rows))
        # Compact stability table by compound and orientation class.
        stability = []
        for cid in TARGETS:
            rows = [r for r in top_rows if r["compound_id"] == cid]
            stability.append({"receptor": pdb_id, "compound_id": cid, "top_seed_count": len(rows), "native_like_count": sum(r["orientation_class"] == "consistent_with_native_orientation" for r in rows), "reversed_supported_count": sum(r["orientation_class"] == "head_to_tail_reversal_supported" for r in rows), "alternative_count": sum(r["orientation_class"] == "alternative_orientation" for r in rows), "ambiguous_count": sum(r["orientation_class"] == "ambiguous" for r in rows), "angle_mean_deg": float(np.mean([r["c3_c17_angle_to_TES_deg"] for r in rows])), "angle_sd_deg": float(np.std([r["c3_c17_angle_to_TES_deg"] for r in rows], ddof=1)), "score_mean": float(np.mean([r["vina_score_kcal_mol"] for r in rows])), "score_sd": float(np.std([r["vina_score_kcal_mol"] for r in rows], ddof=1))})
        write_csv(out / "orientation_stability.csv", stability)
        (out / "run_metadata.json").write_text(json.dumps({"receptor": pdb_id, "seeds": SEEDS, "targets": TARGETS, "protocol": {"exhaustiveness": 16, "n_poses": 10, "cpu": cpu, "rmsd_not_used_for_orientation": True}, "versions": package_versions(), "runs": run_meta, "mapping": mappings}, indent=2))
        print(json.dumps({"receptor": pdb_id, "top_rows": len(top_rows), "all_rows": len(all_rows)}, indent=2))


def competitive_rows(rows: list[dict]) -> list[dict]:
    result = []
    for cid in TARGETS:
        for seed in SEEDS:
            group = [r for r in rows if r["compound_id"] == cid and int(r["seed"]) == seed]
            native = [r for r in group if r["orientation_class"] == "consistent_with_native_orientation"]
            reverse = [r for r in group if r["orientation_class"] == "head_to_tail_reversal_supported"]
            result.append({"compound_id": cid, "seed": seed, "best_native_score": min((r["vina_score_kcal_mol"] for r in native), default=np.nan), "best_reversed_score": min((r["vina_score_kcal_mol"] for r in reverse), default=np.nan), "native_pose_count": len(native), "reversed_pose_count": len(reverse), "delta_reversed_minus_native": (min((r["vina_score_kcal_mol"] for r in reverse), default=np.nan) - min((r["vina_score_kcal_mol"] for r in native), default=np.nan)) if native and reverse else np.nan})
    return result


if __name__ == "__main__":
    main()

