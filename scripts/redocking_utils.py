"""Utilities for reproducible 2AM9 Testosterone redocking validation."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem


def read_box_config(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key == "receptor":
            values[key] = value
        elif key in {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}:
            values[key] = float(value)
        elif key in {"exhaustiveness", "num_modes"}:
            values[key] = int(float(value))
    required = {"receptor", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"Missing box parameters: {sorted(missing)}")
    values["center"] = np.array([values["center_x"], values["center_y"], values["center_z"]], dtype=float)
    values["size"] = np.array([values["size_x"], values["size_y"], values["size_z"]], dtype=float)
    return values


def split_vina_models(pdbqt_path: Path, output_dir: Path, seed: int) -> list[Path]:
    lines = pdbqt_path.read_text().splitlines()
    model_starts = [i for i, line in enumerate(lines) if line.startswith("MODEL")]
    if not model_starts:
        model_blocks = [lines]
        preamble: list[str] = []
    else:
        preamble = lines[: model_starts[0]]
        model_blocks = []
        for index, start in enumerate(model_starts):
            end = model_starts[index + 1] if index + 1 < len(model_starts) else len(lines)
            model_blocks.append(lines[start:end])
    paths = []
    for pose_index, block in enumerate(model_blocks, 1):
        block = [line for line in block if line.strip() != "END"]
        path = output_dir / f"seed_{seed}_pose_{pose_index:02d}.pdbqt"
        path.write_text("\n".join(preamble + block + ["END"]) + "\n")
        paths.append(path)
    return paths


def vina_result_score(pdbqt_path: Path) -> float:
    return vina_result_fields(pdbqt_path)[0]


def vina_result_fields(pdbqt_path: Path) -> tuple[float, float, float]:
    for line in pdbqt_path.read_text().splitlines():
        match = re.search(r"REMARK VINA RESULT:\s*([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)", line)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
    raise ValueError(f"No VINA RESULT score in {pdbqt_path}")


def convert_pdbqt_to_sdf(pdbqt_path: Path, sdf_path: Path, project_root: Path) -> str:
    env_executable = Path(sys.executable).resolve().parent / "mk_export.py"
    executable = str(env_executable) if env_executable.exists() else shutil.which("mk_export.py")
    if not executable:
        raise FileNotFoundError("mk_export.py was not found in the active Python environment or PATH")
    command = [executable, "-s", str(sdf_path), str(pdbqt_path)]
    result = subprocess.run(command, cwd=str(project_root), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"mk_export failed for {pdbqt_path}:\n{result.stdout}\n{result.stderr}")
    return (result.stdout + "\n" + result.stderr).strip()


def load_single_sdf(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"RDKit could not parse {path}")
    return mol


def graph_matches(reference: Chem.Mol, predicted: Chem.Mol) -> tuple[list[tuple[int, ...]], bool]:
    ref = Chem.RemoveHs(Chem.Mol(reference))
    pred = Chem.RemoveHs(Chem.Mol(predicted))
    ref_smiles = Chem.MolToSmiles(ref, isomericSmiles=True)
    pred_smiles = Chem.MolToSmiles(pred, isomericSmiles=True)
    if ref_smiles != pred_smiles:
        # A stereochemical export can differ in RDKit tag placement while
        # retaining the same connectivity. Verify connectivity separately.
        if Chem.MolToSmiles(ref, isomericSmiles=False) != Chem.MolToSmiles(pred, isomericSmiles=False):
            raise ValueError(f"Crystal/predicted chemical graph mismatch: {ref_smiles} vs {pred_smiles}")
        use_chirality = False
    else:
        use_chirality = True
    matches = list(pred.GetSubstructMatches(ref, useChirality=use_chirality, uniquify=False))
    if not matches:
        raise ValueError("No atom mapping between crystal and predicted TES")
    return matches, use_chirality


def direct_rmsd(reference: Chem.Mol, predicted: Chem.Mol, match: tuple[int, ...]) -> float:
    ref = Chem.RemoveHs(Chem.Mol(reference))
    pred = Chem.RemoveHs(Chem.Mol(predicted))
    ref_conf = ref.GetConformer()
    pred_conf = pred.GetConformer()
    squared = []
    for ref_index, pred_index in enumerate(match):
        r = np.array(ref_conf.GetAtomPosition(ref_index), dtype=float)
        p = np.array(pred_conf.GetAtomPosition(pred_index), dtype=float)
        squared.append(float(np.sum((p - r) ** 2)))
    return float(np.sqrt(np.mean(squared)))


def rmsd_with_chemical_mapping(reference: Chem.Mol, predicted: Chem.Mol) -> dict[str, Any]:
    matches, chirality_used = graph_matches(reference, predicted)
    rmsds = [direct_rmsd(reference, predicted, match) for match in matches]
    best = int(np.argmin(rmsds))
    return {
        "rmsd_A": rmsds[best],
        "mapping_count": len(matches),
        "mapping_chirality_used": chirality_used,
        "symmetry_mapping_rmsds_A": rmsds,
        "mapping": list(matches[best]),
    }


def crystal_atom_names(crystal: Chem.Mol) -> list[str]:
    names = []
    for atom in crystal.GetAtoms():
        if atom.GetAtomicNum() != 1:
            names.append(atom.GetProp("ccd_atom_id") if atom.HasProp("ccd_atom_id") else f"atom_{atom.GetIdx()+1}")
    return names


def pdb_heavy_atoms(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM  "):
            continue
        element = line[76:78].strip() or re.sub(r"[^A-Za-z]", "", line[12:16]).strip()[:1]
        if element.upper() == "H":
            continue
        atoms.append({
            "chain": line[21],
            "resname": line[17:20].strip(),
            "resseq": int(line[22:26]),
            "atom": line[12:16].strip(),
            "element": element.upper(),
            "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
        })
    return atoms


def top_pose_interactions(receptor_pdb: Path, crystal: Chem.Mol, predicted: Chem.Mol, match: tuple[int, ...]) -> dict[str, Any]:
    receptor_atoms = pdb_heavy_atoms(receptor_pdb)
    pred = Chem.RemoveHs(Chem.Mol(predicted))
    pred_conf = pred.GetConformer()
    ligand_names = crystal_atom_names(crystal)
    ligand_atoms = [
        {"atom": ligand_names[ref_index], "element": pred.GetAtomWithIdx(pred_index).GetSymbol(), "coord": np.array(pred_conf.GetAtomPosition(pred_index), dtype=float)}
        for ref_index, pred_index in enumerate(match)
    ]

    residue_contacts: dict[str, dict[str, Any]] = {}
    for residue in ["ASN705", "GLN711", "ARG752", "THR877"]:
        target = [atom for atom in receptor_atoms if f"{atom['resname']}{atom['resseq']}" == residue]
        pairs = []
        for ratom in target:
            for latom in ligand_atoms:
                distance = float(np.linalg.norm(ratom["coord"] - latom["coord"]))
                if distance <= 4.0:
                    pairs.append({"protein_atom": ratom["atom"], "ligand_atom": latom["atom"], "distance_A": round(distance, 3)})
        residue_contacts[residue] = {
            "contact_count_within_4A": len(pairs),
            "minimum_distance_A": min((pair["distance_A"] for pair in pairs), default=None),
            "contacts": sorted(pairs, key=lambda item: item["distance_A"]),
        }

    hbond_candidates = []
    donor_acceptor_atoms = {
        ("ASN705", "ND2"), ("ASN705", "OD1"),
        ("GLN711", "NE2"), ("GLN711", "OE1"),
        ("ARG752", "NH1"), ("ARG752", "NH2"),
        ("THR877", "OG1"),
    }
    ligand_polar = [atom for atom in ligand_atoms if atom["atom"] in {"O3", "O17"}]
    for ratom in receptor_atoms:
        if (f"{ratom['resname']}{ratom['resseq']}", ratom["atom"]) not in donor_acceptor_atoms:
            continue
        for latom in ligand_polar:
            distance = float(np.linalg.norm(ratom["coord"] - latom["coord"]))
            if distance <= 3.5:
                hbond_candidates.append({"protein_atom": f"{ratom['resname']}{ratom['resseq']}:{ratom['atom']}", "ligand_atom": latom["atom"], "heavy_atom_distance_A": round(distance, 3), "interpretation": "geometric H-bond candidate; angle/protonation not independently confirmed"})

    min_pair = None
    for ratom in receptor_atoms:
        for latom in ligand_atoms:
            distance = float(np.linalg.norm(ratom["coord"] - latom["coord"]))
            if min_pair is None or distance < min_pair["distance_A"]:
                min_pair = {"protein_atom": f"{ratom['resname']}{ratom['resseq']}:{ratom['atom']}", "ligand_atom": latom["atom"], "distance_A": round(distance, 3)}
    return {
        "key_residue_contacts": residue_contacts,
        "hbond_candidates": hbond_candidates,
        "minimum_protein_ligand_heavy_atom_distance": min_pair,
        "obvious_heavy_atom_clash_lt_2A": bool(min_pair and min_pair["distance_A"] < 2.0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

