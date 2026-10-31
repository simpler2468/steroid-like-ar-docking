"""Prepare the 47 cluster ligands for later docking, without docking."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "androgens_clusters_teacher_profile.xlsx"
SHEET = "Cluster compounds"
SDF_DIR = ROOT / "data" / "ligands" / "sdf"
PDBQT_DIR = ROOT / "data" / "ligands" / "pdbqt"
DIAG_DIR = ROOT / "data" / "ligands" / "diagnostics"
RESULTS = ROOT / "results" / "ligand_preparation"
MANIFEST = RESULTS / "ligand_manifest.csv"
BASE_SEED = 20250311
N_CONFS = 5
EXPECTED_COUNTS = {3: 10, 176: 11, 246: 22, 416: 4}


def scalar(value: Any) -> Any:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return int(value)
    return value


def stable_id(cluster_id: Any, row_index: Any) -> str:
    return f"cluster_{scalar(cluster_id)}_row_{scalar(row_index)}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def potential_stereo_issues(mol: Chem.Mol) -> tuple[list[str], list[str]]:
    unassigned: list[str] = []
    double_bond: list[str] = []
    for atom_idx, label in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    ):
        if label == "?":
            atom = mol.GetAtomWithIdx(atom_idx)
            unassigned.append(f"atom_{atom_idx}_{atom.GetSymbol()}")
    for info in Chem.FindPotentialStereo(mol):
        type_text = str(info.type)
        specified_text = str(info.specified)
        if "Bond_Double" in type_text and "Unspecified" in specified_text:
            double_bond.append(f"bond_{info.idx}")
    return unassigned, double_bond


def protonation_review(mol: Chem.Mol) -> list[str]:
    issues: list[str] = []
    if Chem.GetFormalCharge(mol) != 0:
        issues.append("explicit_formal_charge_preserved")
    patterns = {
        "carboxylic_acid": Chem.MolFromSmarts("C(=O)[O;H1;+0]"),
        "neutral_amine": Chem.MolFromSmarts("[N;H0,H1,H2;+0;!$(N-C=O)]"),
        "aromatic_nitrogen": Chem.MolFromSmarts("[n;+0]"),
        "thiol": Chem.MolFromSmarts("[S;H1;+0]"),
    }
    for name, pattern in patterns.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            issues.append(f"possible_multiple_protonation_states:{name}")
    return issues


def unusual_atoms(mol: Chem.Mol) -> list[str]:
    common = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
    return [
        f"atom_{atom.GetIdx()}_{atom.GetSymbol()}"
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in common
    ]


def geometry_diagnostics(mol: Chem.Mol, conformer_id: int) -> dict[str, Any]:
    conf = mol.GetConformer(conformer_id)
    coords = np.asarray(conf.GetPositions(), dtype=float)
    finite = bool(np.isfinite(coords).all())
    bad_bonds: list[dict[str, Any]] = []
    bonded = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded.add(tuple(sorted((i, j))))
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance < 0.7 or distance > 2.4:
            bad_bonds.append({"i": i, "j": j, "distance_A": distance})
    severe_clashes: list[dict[str, Any]] = []
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    for pos, i in enumerate(heavy):
        for j in heavy[pos + 1 :]:
            if tuple(sorted((i, j))) in bonded:
                continue
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance < 1.25:
                severe_clashes.append({"i": i, "j": j, "distance_A": distance})
    return {
        "finite_coordinates": finite,
        "bad_bond_count": len(bad_bonds),
        "bad_bonds": bad_bonds,
        "severe_nonbonded_clash_count": len(severe_clashes),
        "severe_nonbonded_clashes": severe_clashes,
    }


def optimize_conformers(mol: Chem.Mol, conformer_ids: list[int]) -> tuple[str, list[dict[str, Any]], str | None]:
    method = "MMFF94s"
    records: list[dict[str, Any]] = []
    mmff_available = bool(AllChem.MMFFHasAllMoleculeParams(mol))
    if not mmff_available:
        method = "UFF"
    for conf_id in conformer_ids:
        try:
            if mmff_available:
                forcefield = AllChem.MMFFGetMoleculeForceField(
                    mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s"), confId=conf_id
                )
            else:
                forcefield = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if forcefield is None:
                raise RuntimeError("force-field parameters unavailable")
            status = int(forcefield.Minimize(maxIts=300))
            energy = float(forcefield.CalcEnergy())
            records.append({"conformer_id": conf_id, "optimization_status": status, "energy": energy})
        except Exception as exc:  # preserve the conformer diagnostic
            records.append({"conformer_id": conf_id, "optimization_status": "failed", "energy": None, "error": str(exc)})
    finite = [item for item in records if isinstance(item.get("energy"), float) and math.isfinite(item["energy"])]
    if not finite:
        return method, records, "no_conformer_with_finite_optimized_energy"
    return method, records, None


def run_meeko(sdf_path: Path, pdbqt_path: Path, log_path: Path) -> tuple[bool, str]:
    env_executable = Path(sys.executable).resolve().parent / "mk_prepare_ligand.py"
    executable = str(env_executable) if env_executable.exists() else shutil.which("mk_prepare_ligand.py")
    if not executable:
        raise FileNotFoundError("mk_prepare_ligand.py was not found in the active Python environment or PATH")
    command = [executable, "-i", str(sdf_path), "-o", str(pdbqt_path)]
    result = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
    text = f"$ {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    log_path.write_text(text, encoding="utf-8")
    return result.returncode == 0 and pdbqt_path.exists() and pdbqt_path.stat().st_size > 0, text


def main() -> None:
    for directory in (SDF_DIR, PDBQT_DIR, DIAG_DIR, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_excel(INPUT, sheet_name=SHEET, header=2)
    required = {"cluster_id", "cluster_size", "row_index", "split", "SMILES"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rows = dataframe.to_dict(orient="records")
    if len(rows) != 47:
        raise ValueError(f"Expected 47 compounds, found {len(rows)}")

    teacher_columns = [column for column in dataframe.columns if str(column).startswith("teacher_probability__")]
    manifest_rows: list[dict[str, Any]] = []
    source_hash = sha256(INPUT)
    for position, raw in enumerate(rows, 1):
        cluster_id = scalar(raw["cluster_id"])
        row_index = scalar(raw["row_index"])
        compound_id = stable_id(cluster_id, row_index)
        original_smiles = str(raw["SMILES"]).strip()
        sdf_path = SDF_DIR / f"{compound_id}.sdf"
        pdbqt_path = PDBQT_DIR / f"{compound_id}.pdbqt"
        diagnostic_path = DIAG_DIR / f"{compound_id}.json"
        log_path = DIAG_DIR / f"{compound_id}_meeko.log"
        diagnostic: dict[str, Any] = {
            "compound_id": compound_id,
            "cluster_id": cluster_id,
            "row_index": row_index,
            "source_excel_sha256": source_hash,
            "original_smiles": original_smiles,
            "status": "failed",
            "warnings": [],
            "errors": [],
        }
        record: dict[str, Any] = {
            "compound_id": compound_id,
            "source_excel_sha256": source_hash,
            "cluster_id": cluster_id,
            "cluster_size": scalar(raw["cluster_size"]),
            "row_index": row_index,
            "split": scalar(raw["split"]),
            "original_smiles": original_smiles,
            "canonical_isomeric_smiles": "",
            "molecular_formula": "",
            "molecular_weight": None,
            "formal_charge": None,
            "heavy_atom_count": None,
            "rotatable_bond_count": None,
            "chiral_center_count": None,
            "unassigned_stereocenter_count": None,
            "unassigned_stereocenters": "",
            "double_bond_stereochemistry_issues": "",
            "protonation_review": "",
            "unusual_atoms": "",
            "stereochemistry_status": "unknown",
            "sdf_path": str(sdf_path.relative_to(ROOT)),
            "pdbqt_path": str(pdbqt_path.relative_to(ROOT)),
            "preparation_status": "failed",
            "failure_or_warning_reason": "",
            "PA_Terms_List": scalar(raw.get("PA_Terms_List", "")),
            "has_androgens_annotation": scalar(raw.get("has_androgens_annotation", "")),
            "AR_mean_probability": scalar(raw.get("AR_mean_probability", "")),
            "teacher_profile_json": json.dumps({column: scalar(raw.get(column, "")) for column in teacher_columns}, ensure_ascii=False),
        }
        try:
            mol = Chem.MolFromSmiles(original_smiles, sanitize=True)
            if mol is None:
                raise ValueError("RDKit MolFromSmiles returned None")
            Chem.SanitizeMol(mol)
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            unassigned, double_issues = potential_stereo_issues(mol)
            protonation = protonation_review(mol)
            unusual = unusual_atoms(mol)
            components = len(Chem.GetMolFrags(mol, asMols=False))
            warnings: list[str] = []
            if components > 1:
                warnings.append(f"multiple_components:{components}")
            if unassigned:
                warnings.append("unassigned_stereocenters")
            if double_issues:
                warnings.append("unspecified_double_bond_stereochemistry")
            if protonation:
                warnings.append("protonation_review_required")
            if unusual:
                warnings.append("unusual_atoms")
            if Chem.GetFormalCharge(mol) != 0:
                warnings.append("explicit_charge_retained")
            record.update(
                {
                    "canonical_isomeric_smiles": canonical,
                    "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
                    "molecular_weight": float(Descriptors.MolWt(mol)),
                    "formal_charge": int(Chem.GetFormalCharge(mol)),
                    "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
                    "rotatable_bond_count": int(Lipinski.NumRotatableBonds(mol)),
                    "chiral_center_count": int(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))),
                    "unassigned_stereocenter_count": int(len(unassigned)),
                    "unassigned_stereocenters": ";".join(unassigned),
                    "double_bond_stereochemistry_issues": ";".join(double_issues),
                    "protonation_review": ";".join(protonation),
                    "unusual_atoms": ";".join(unusual),
                    "stereochemistry_status": "stereo_review_required" if (unassigned or double_issues) else "specified_or_not_applicable",
                }
            )
            diagnostic.update({"canonical_isomeric_smiles": canonical, "warnings": warnings, "component_count": components})

            hmol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = BASE_SEED + position
            params.numThreads = 1
            params.enforceChirality = True
            params.useSmallRingTorsions = True
            params.useMacrocycleTorsions = True
            conformer_ids = list(AllChem.EmbedMultipleConfs(hmol, numConfs=N_CONFS, params=params))
            diagnostic["embedding"] = {"method": "ETKDGv3", "random_seed": params.randomSeed, "requested_conformers": N_CONFS, "generated_conformers": len(conformer_ids)}
            if not conformer_ids:
                raise RuntimeError("ETKDGv3 generated no conformers")
            method, optimizations, optimization_error = optimize_conformers(hmol, conformer_ids)
            diagnostic["optimization"] = {"method": method, "conformers": optimizations}
            if optimization_error:
                warnings.append(optimization_error)
            finite = [item for item in optimizations if isinstance(item.get("energy"), float) and math.isfinite(item["energy"])]
            selected = min(finite, key=lambda item: item["energy"]) if finite else {"conformer_id": conformer_ids[0], "energy": None}
            selected_id = int(selected["conformer_id"])
            geometry = geometry_diagnostics(hmol, selected_id)
            diagnostic["selected_conformer"] = {"conformer_id": selected_id, "energy": selected.get("energy"), "geometry": geometry}
            if not geometry["finite_coordinates"] or geometry["bad_bond_count"] or geometry["severe_nonbonded_clash_count"]:
                warnings.append("geometry_review_required")
            if Chem.MolToSmiles(Chem.RemoveHs(hmol), isomericSmiles=True) != canonical:
                raise ValueError("3D molecule connectivity/stereochemistry differs from original canonical SMILES")
            writer = Chem.SDWriter(str(sdf_path))
            hmol.SetProp("compound_id", compound_id)
            hmol.SetProp("original_smiles", original_smiles)
            hmol.SetProp("canonical_isomeric_smiles", canonical)
            hmol.SetProp("conformer_method", "ETKDGv3")
            hmol.SetProp("optimization_method", method)
            hmol.SetProp("selected_conformer_id", str(selected_id))
            hmol.SetProp("selected_energy", "" if selected.get("energy") is None else str(selected["energy"]))
            writer.write(hmol, confId=selected_id)
            writer.close()
            meeko_ok, _ = run_meeko(sdf_path, pdbqt_path, log_path)
            diagnostic["meeko"] = {"success": meeko_ok, "log": str(log_path.relative_to(ROOT))}
            if not meeko_ok:
                diagnostic["errors"].append("meeko_prepare_ligand_failed")
                record["failure_or_warning_reason"] = ";".join(warnings + ["meeko_prepare_ligand_failed"])
                record["preparation_status"] = "failed"
            else:
                record["failure_or_warning_reason"] = ";".join(warnings)
                record["preparation_status"] = "review_required" if warnings else "ready"
                diagnostic["status"] = record["preparation_status"]
        except Exception as exc:
            diagnostic["errors"].append(str(exc))
            record["failure_or_warning_reason"] = ";".join(diagnostic["warnings"] + diagnostic["errors"])
            record["preparation_status"] = "failed"
        diagnostic_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_rows.append(record)

    pd.DataFrame(manifest_rows).to_csv(MANIFEST, index=False)
    metadata = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": source_hash,
        "sheet": SHEET,
        "header_row_zero_based": 2,
        "expected_cluster_counts": EXPECTED_COUNTS,
        "n_compounds": len(manifest_rows),
        "embedding": {"method": "ETKDGv3", "requested_conformers": N_CONFS, "base_seed": BASE_SEED, "threads": 1},
        "optimization": "MMFF94s when parameters available, otherwise UFF",
        "meeko_command": "mk_prepare_ligand.py -i <sdf> -o <pdbqt>",
        "docking_executed": False,
    }
    (RESULTS / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "n_compounds": len(manifest_rows), "input_sha256": source_hash}, indent=2))


if __name__ == "__main__":
    main()
