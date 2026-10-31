"""Utilities shared by the Step 5 receptor-ensemble workflow.

The functions in this module deliberately keep the original coordinate files
immutable.  Receptor preparation and alignment are performed on new files in
``data/prepared/<pdb_id>`` and all docking calculations use those derived
files.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from prepare_utils import cif_rows
from redocking_utils import read_box_config
from redocking_utils import (
    convert_pdbqt_to_sdf,
    direct_rmsd,
    graph_matches,
    load_single_sdf,
    split_vina_models,
    vina_result_fields,
)


ROOT = Path(__file__).resolve().parents[1]
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE",
}


def xyz(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)


def parse_remark_missing(pdb: Path) -> dict[str, Any]:
    missing_residues: list[dict[str, Any]] = []
    missing_atoms: list[dict[str, Any]] = []
    for line in pdb.read_text().splitlines():
        m = re.match(r"REMARK 465\s+(\w{3})\s+(\S)\s+(\d+)", line)
        if m:
            missing_residues.append({"resname": m.group(1), "chain": m.group(2), "resseq": int(m.group(3))})
        m = re.match(r"REMARK 470\s+(\w{3})\s+(\S)\s+(\d+)\s+(.+)$", line)
        if m:
            missing_atoms.append({"resname": m.group(1), "chain": m.group(2), "resseq": int(m.group(3)), "atoms": m.group(4).split()})
    return {"missing_residues": missing_residues, "missing_atoms": missing_atoms}


def inspect_raw_pdb(pdb: Path, ligand_code: str, ligand_number: int) -> dict[str, Any]:
    structure = gemmi.read_structure(str(pdb))
    model = structure[0]
    chains: dict[str, Any] = {}
    for chain in model:
        residues = [res for res in chain if res.het_flag == "A"]
        chains[chain.name] = {
            "observed_protein_residues": len(residues),
            "first_residue": f"{residues[0].name}{residues[0].seqid.num}" if residues else None,
            "last_residue": f"{residues[-1].name}{residues[-1].seqid.num}" if residues else None,
        }
    het_instances = []
    seen: set[tuple[str, str, int, str]] = set()
    for chain in model:
        for res in chain:
            if res.het_flag != "A":
                key = (res.name, chain.name, res.seqid.num, res.seqid.icode)
                if key not in seen:
                    seen.add(key)
                    het_instances.append({"resname": res.name, "chain": chain.name, "resseq": res.seqid.num, "icode": res.seqid.icode, "atom_count": len(res)})
    altloc = defaultdict(int)
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.altloc not in ("", " ", "\x00"):
                    altloc[f"{chain.name}:{res.name}{res.seqid.num}"] += 1
    ligand = []
    for item in het_instances:
        if item["resname"] == ligand_code and item["chain"] == "A":
            ligand.append(item)
    out = {
        "pdb": pdb.name,
        "chains": chains,
        "hetatm_instances": het_instances,
        "ligand_code": ligand_code,
        "ligand_instances": ligand,
        "altloc_atom_counts": dict(altloc),
        **parse_remark_missing(pdb),
    }
    if not ligand or not any(x["resseq"] == ligand_number for x in ligand):
        raise ValueError(f"Expected {ligand_code} A:{ligand_number} in {pdb}")
    return out


def write_clean_chain_a(raw_pdb: Path, output: Path) -> dict[str, Any]:
    """Write observed chain-A protein atoms; remove all HETATM and altlocs."""
    kept: list[str] = []
    removed_altloc = []
    for line in raw_pdb.read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] == "A":
            if line[16] not in (" ", "A", "1", "\x00"):
                removed_altloc.append(line[12:16].strip())
                continue
            if line[16] in ("A", "1"):
                line = line[:16] + " " + line[17:]
            kept.append(line)
    if not kept:
        raise ValueError(f"No chain-A protein atoms in {raw_pdb}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("REMARK 900 Step 5 observed chain A protein; non-protein components removed\n" + "\n".join(kept) + "\nEND\n")
    return {"atom_count": len(kept), "removed_altloc_atom_names": removed_altloc}


def repair_with_pdbfixer(input_pdb: Path, output_pdb: Path) -> dict[str, Any]:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    missing_residues_before = {str(k): list(v) for k, v in fixer.missingResidues.items()}
    # The 654–669 unresolved segment is intentionally not rebuilt.
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    missing_atoms_before = {}
    for residue, atoms in fixer.missingAtoms.items():
        key = f"{residue.chain.id}:{residue.name}{residue.id}"
        missing_atoms_before[key] = sorted(atom.name for atom in atoms)
    fixer.addMissingAtoms()
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    return {
        "missing_residues_detected_but_not_built": missing_residues_before,
        "missing_atoms_before_repair": missing_atoms_before,
        "input_atom_count": sum(1 for line in input_pdb.read_text().splitlines() if line.startswith("ATOM")),
        "output_atom_count": sum(1 for line in output_pdb.read_text().splitlines() if line.startswith("ATOM")),
    }


def run_meeko_receptor(receptor_pdb: Path, basename: Path) -> dict[str, Any]:
    exe = Path(sys.executable).resolve().parent / "mk_prepare_receptor.py"
    if not exe.exists():
        exe = Path(shutil.which("mk_prepare_receptor.py") or "")
    if not exe.exists():
        raise FileNotFoundError("mk_prepare_receptor.py is unavailable")
    cmd = [str(exe), "--read_pdb", str(receptor_pdb), "--output_basename", str(basename), "--write_pdbqt", str(basename.with_suffix(".pdbqt")), "--write_json", str(basename.with_suffix(".json")), "--write_pdb", str(basename.with_name(basename.name + "_H.pdb")), "--compute_charges", "--charge_model", "gasteiger"]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    log = basename.with_name(basename.name + "_meeko.log")
    log.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Meeko receptor preparation failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return {"command": cmd, "log": str(log.relative_to(ROOT)), "stdout_tail": (result.stdout + result.stderr)[-2000:]}


def ligand_residue_coords(pdb: Path, code: str, number: int) -> dict[str, np.ndarray]:
    structure = gemmi.read_structure(str(pdb))
    chain = structure[0]["A"]
    matches = [res for res in chain if res.name == code and res.seqid.num == number]
    if len(matches) != 1:
        raise ValueError(f"Expected one {code} A:{number} in {pdb}, found {len(matches)}")
    return {atom.name.strip(): xyz(atom) for atom in matches[0] if atom.element.name != "H"}


def build_ccd_ligand(pdb: Path, ccd: Path, code: str, number: int, pdb_id: str) -> tuple[Chem.Mol, dict[str, Any]]:
    coords = ligand_residue_coords(pdb, code, number)
    block = gemmi.cif.read(str(ccd)).sole_block()
    atoms = cif_rows(block, "_chem_comp_atom")
    bonds = cif_rows(block, "_chem_comp_bond")
    heavy = [row for row in atoms if row["type_symbol"] != "H"]
    names = {row["atom_id"] for row in heavy}
    if names != set(coords):
        raise ValueError(f"{pdb_id}/{code} PDB/CCD atom mismatch: PDB-only={sorted(set(coords)-names)} CCD-only={sorted(names-set(coords))}")
    rw = Chem.RWMol()
    indices: dict[str, int] = {}
    for row in heavy:
        atom = Chem.Atom(row["type_symbol"])
        charge = row.get("charge", "0")
        try:
            atom.SetFormalCharge(int(float(charge)))
        except ValueError:
            atom.SetFormalCharge(0)
        atom.SetProp("ccd_atom_id", row["atom_id"])
        atom.SetProp("atom_name", row["atom_id"])
        indices[row["atom_id"]] = rw.AddAtom(atom)
    bond_types = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE, "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC}
    for row in bonds:
        a1, a2 = row["atom_id_1"], row["atom_id_2"]
        if a1 in names and a2 in names:
            rw.AddBond(indices[a1], indices[a2], bond_types[row["value_order"]])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for name, idx in indices.items():
        conf.SetAtomPosition(idx, tuple(float(x) for x in coords[name]))
    mol.AddConformer(conf, assignId=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    mol.SetProp("_Name", f"{code}_{pdb_id}_crystal")
    mol.SetProp("PDB_ID", pdb_id)
    mol.SetProp("RESIDUE", f"{code} A:{number}")
    mol.SetProp("CCD_ID", code)
    mol.SetProp("COORDINATE_SOURCE", f"{pdb_id} experimental HETATM coordinates")
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return mol, {"formula": formula, "heavy_atoms": mol.GetNumHeavyAtoms(), "atom_names": sorted(names), "ccd": code, "pdb_id": pdb_id}


def add_hs_preserving_coords(mol: Chem.Mol, name: str) -> Chem.Mol:
    out = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    out.SetProp("_Name", name)
    out.SetProp("HYDROGENS", "RDKit AddHs(addCoords=True); heavy atom coordinates retained")
    return out


def transform_mol(mol: Chem.Mol, rotation: np.ndarray, translation: np.ndarray, name: str) -> Chem.Mol:
    out = Chem.Mol(mol)
    for conf in out.GetConformers():
        for i in range(out.GetNumAtoms()):
            old = np.array(conf.GetAtomPosition(i), dtype=float)
            new = old @ rotation + translation
            conf.SetAtomPosition(i, tuple(float(x) for x in new))
    out.SetProp("_Name", name)
    return out


def write_sdf(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def kabsch_mobile_to_target(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cm = mobile.mean(axis=0)
    ct = target.mean(axis=0)
    xm, yt = mobile - cm, target - ct
    u, _, vt = np.linalg.svd(xm.T @ yt)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    t = ct - cm @ r
    fitted = mobile @ r + t
    return r, t, float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))


def protein_atoms(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    atoms = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[21], int(line[22:26]), line[12:16].strip())
        atoms[key] = {"resname": line[17:20].strip(), "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float), "line": line}
    return atoms


def align_receptor_to_reference(mobile_pdb: Path, reference_pdb: Path, output_pdb: Path, key_residues: Iterable[int]) -> dict[str, Any]:
    mobile, target = protein_atoms(mobile_pdb), protein_atoms(reference_pdb)
    keys = [key for key in sorted(set(mobile) & set(target)) if key[2] in {"N", "CA", "C", "O"}]
    if len(keys) < 100:
        raise ValueError(f"Too few common backbone atoms for alignment: {len(keys)}")
    x = np.array([mobile[k]["coord"] for k in keys])
    y = np.array([target[k]["coord"] for k in keys])
    r, t, rmsd = kabsch_mobile_to_target(x, y)
    lines = []
    for line in mobile_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            old = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
            new = old @ r + t
            line = line[:30] + f"{new[0]:8.3f}{new[1]:8.3f}{new[2]:8.3f}" + line[54:]
        lines.append(line)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(lines) + "\n")
    pocket_rows = []
    for resseq in key_residues:
        vals = []
        for atom_name in ("N", "CA", "C", "O"):
            key = ("A", resseq, atom_name)
            if key in mobile and key in target:
                transformed = mobile[key]["coord"] @ r + t
                vals.append(float(np.linalg.norm(transformed - target[key]["coord"])))
        if vals:
            pocket_rows.append({"resseq": resseq, "backbone_rmsd_A": float(np.sqrt(np.mean(np.square(vals)))), "n_atoms": len(vals)})
    return {"common_backbone_atom_count": len(keys), "backbone_rmsd_A": rmsd, "rotation": r.tolist(), "translation": t.tolist(), "key_residue_backbone_displacements": pocket_rows}


def write_box(path: Path, receptor_name: str, center: np.ndarray, size: np.ndarray, extra: dict[str, Any] | None = None) -> None:
    lines = [f"receptor = {receptor_name}", f"center_x = {center[0]:.3f}", f"center_y = {center[1]:.3f}", f"center_z = {center[2]:.3f}", f"size_x = {size[0]:.3f}", f"size_y = {size[1]:.3f}", f"size_z = {size[2]:.3f}", "exhaustiveness = 16", "num_modes = 10"]
    if extra:
        lines += [f"# {k} = {v}" for k, v in extra.items()]
    path.write_text("\n".join(lines) + "\n")


def ligand_box_check(mol: Chem.Mol, center: np.ndarray, size: np.ndarray) -> dict[str, Any]:
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1], dtype=float)
    lower, upper = center - size / 2, center + size / 2
    margin = np.minimum(coords - lower, upper - coords)
    return {"all_heavy_atoms_inside": bool(np.all(margin >= 0)), "minimum_margin_A": float(np.min(margin)), "min_coord": coords.min(axis=0).tolist(), "max_coord": coords.max(axis=0).tolist()}


def mol_heavy_atom_map(reference: Chem.Mol, predicted: Chem.Mol) -> tuple[list[tuple[int, ...]], bool]:
    return graph_matches(reference, predicted)


def crystal_rmsd(reference: Chem.Mol, predicted: Chem.Mol) -> dict[str, Any]:
    matches, chirality = mol_heavy_atom_map(reference, predicted)
    values = [direct_rmsd(reference, predicted, m) for m in matches]
    i = int(np.argmin(values))
    return {"rmsd_A": float(values[i]), "mapping_count": len(matches), "mapping_chirality_used": chirality, "mapping": list(matches[i]), "all_mapping_rmsds_A": [float(x) for x in values]}


def package_versions() -> dict[str, str]:
    out = {}
    for pkg in ("vina", "meeko", "rdkit", "gemmi", "numpy", "scipy", "pdbfixer", "openmm"):
        try:
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "unavailable"
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

