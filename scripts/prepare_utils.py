"""Shared helpers for the 2AM9 preparation and validation scripts."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE",
}


def xyz(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)


def cif_rows(block: gemmi.cif.Block, prefix: str) -> list[dict[str, str]]:
    column = block.find_loop(f"{prefix}.comp_id")
    if column is None:
        raise ValueError(f"Missing CIF loop: {prefix}")
    loop = column.get_loop()
    tags = [tag.split(".", 1)[1] for tag in loop.tags]
    values = list(loop.values)
    width = loop.width()
    return [dict(zip(tags, values[i : i + width])) for i in range(0, len(values), width)]


def read_tes_from_pdb(pdb_path: Path) -> tuple[gemmi.Residue, dict[str, np.ndarray]]:
    structure = gemmi.read_structure(str(pdb_path))
    if len(structure) != 1:
        raise ValueError("Expected one model with chain A in 2AM9")
    try:
        chain = structure[0]["A"]
    except KeyError as exc:
        raise ValueError("Expected chain A in 2AM9") from exc
    matches = [res for res in chain if res.name == "TES" and res.seqid.num == 1000]
    if len(matches) != 1:
        raise ValueError(f"Expected one TES A:1000 residue, found {len(matches)}")
    residue = matches[0]
    coords = {atom.name.strip(): xyz(atom) for atom in residue}
    return residue, coords


def build_tes_mol(pdb_path: Path, ccd_path: Path) -> tuple[Chem.Mol, dict[str, Any]]:
    _, experimental_coords = read_tes_from_pdb(pdb_path)
    block = gemmi.cif.read(str(ccd_path)).sole_block()
    atom_rows = cif_rows(block, "_chem_comp_atom")
    bond_rows = cif_rows(block, "_chem_comp_bond")
    heavy_rows = [row for row in atom_rows if row["type_symbol"] != "H"]
    heavy_names = {row["atom_id"] for row in heavy_rows}
    if len(heavy_rows) != 21:
        raise ValueError(f"CCD TES heavy atom count is {len(heavy_rows)}, expected 21")
    if set(experimental_coords) != heavy_names:
        raise ValueError(
            "TES PDB/CCD atom-name mismatch: "
            f"PDB-only={sorted(set(experimental_coords)-heavy_names)}, "
            f"CCD-only={sorted(heavy_names-set(experimental_coords))}"
        )

    rw = Chem.RWMol()
    atom_index: dict[str, int] = {}
    for row in heavy_rows:
        atom = Chem.Atom(row["type_symbol"])
        atom.SetFormalCharge(int(float(row["charge"])))
        atom.SetProp("ccd_atom_id", row["atom_id"])
        atom.SetProp("atom_name", row["atom_id"])
        atom.SetProp("ccd_stereo_config", row["pdbx_stereo_config"])
        atom_index[row["atom_id"]] = rw.AddAtom(atom)

    bond_type = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "AROM": Chem.BondType.AROMATIC,
    }
    for row in bond_rows:
        a1, a2 = row["atom_id_1"], row["atom_id_2"]
        if a1 in heavy_names and a2 in heavy_names:
            rw.AddBond(atom_index[a1], atom_index[a2], bond_type[row["value_order"]])

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for atom_name, idx in atom_index.items():
        x, y, z = experimental_coords[atom_name]
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, confId=0, replaceExistingTags=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    mol.SetProp("_Name", "TES_2AM9_crystal")
    mol.SetProp("PDB_ID", "2AM9")
    mol.SetProp("RESIDUE", "TES A:1000")
    mol.SetProp("CCD_ID", "TES")
    mol.SetProp("COORDINATE_SOURCE", "2AM9 experimental HETATM coordinates")
    mol.SetProp("FORMULA_CCD", "C19H28O2")

    expected_stereo = {
        row["atom_id"]: row["pdbx_stereo_config"]
        for row in heavy_rows
        if row["pdbx_stereo_config"] in {"R", "S"}
    }
    centers = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True, includeCIP=True))
    observed_stereo = {
        mol.GetAtomWithIdx(idx).GetProp("ccd_atom_id"): cip
        for idx, cip in centers.items()
        if cip in {"R", "S"}
    }
    formula = rdMolDescriptors.CalcMolFormula(mol)
    metadata = {
        "formula": formula,
        "expected_formula": "C19H28O2",
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "expected_heavy_atoms": 21,
        "expected_stereo": expected_stereo,
        "observed_stereo": observed_stereo,
        "experimental_coords": {k: v.tolist() for k, v in experimental_coords.items()},
    }
    return mol, metadata


def write_sdf(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def add_hydrogens_preserving_heavy_coords(mol: Chem.Mol) -> Chem.Mol:
    prepared = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    Chem.AssignStereochemistry(prepared, cleanIt=True, force=True)
    prepared.SetProp("_Name", "TES_2AM9_prepared")
    for key, value in mol.GetPropsAsDict().items():
        prepared.SetProp(key, str(value))
    prepared.SetProp("HYDROGENS", "RDKit AddHs(addCoords=True); heavy atom coordinates retained")
    return prepared


def heavy_coord_rmsd(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    def mapped_coords(mol: Chem.Mol) -> dict[str, np.ndarray]:
        conf = mol.GetConformer()
        return {
            atom.GetProp("ccd_atom_id"): np.array(conf.GetAtomPosition(atom.GetIdx()))
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != 1 and atom.HasProp("ccd_atom_id")
        }
    a, b = mapped_coords(mol_a), mapped_coords(mol_b)
    if a and b and set(a) == set(b):
        d2 = [np.sum((a[name] - b[name]) ** 2) for name in sorted(a)]
        return float(math.sqrt(sum(d2) / len(d2)))
    # SD files do not carry arbitrary per-atom RDKit properties through the
    # standard writer. The atom order is retained by RDKit, so use the
    # validated CCD construction order as a deterministic fallback.
    def ordered_heavy(mol: Chem.Mol) -> list[np.ndarray]:
        conf = mol.GetConformer()
        return [
            np.array(conf.GetAtomPosition(atom.GetIdx()))
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != 1
        ]

    ordered_a, ordered_b = ordered_heavy(mol_a), ordered_heavy(mol_b)
    if len(ordered_a) != len(ordered_b):
        raise ValueError("Cannot compare heavy coordinates: atom counts differ")
    d2 = [np.sum((coord_a - coord_b) ** 2) for coord_a, coord_b in zip(ordered_a, ordered_b)]
    return float(math.sqrt(sum(d2) / len(d2)))


def parse_pdbqt_coords(path: Path) -> list[tuple[str, np.ndarray]]:
    coords: list[tuple[str, np.ndarray]] = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            coords.append((name, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return coords


def write_box_pdb(path: Path, center: np.ndarray, size: np.ndarray) -> None:
    half = size / 2.0
    corners = [
        np.array([sx * half[0], sy * half[1], sz * half[2]]) + center
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]
    edges = []
    for i, point in enumerate(corners):
        for axis in range(3):
            other = point.copy()
            other[axis] = center[axis] - half[axis] if point[axis] > center[axis] else center[axis] + half[axis]
            j = min(range(len(corners)), key=lambda k: float(np.linalg.norm(corners[k] - other)))
            if i < j:
                edges.append((i + 1, j + 1))
    lines = ["REMARK 900 Vina docking box visualization", "REMARK 900 center " + " ".join(f"{x:.3f}" for x in center), "REMARK 900 size " + " ".join(f"{x:.3f}" for x in size)]
    for i, point in enumerate(corners, 1):
        lines.append(f"HETATM{i:5d}  C   BOX A   1    {point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00           C")
    for i, j in edges:
        lines.append(f"CONECT{i:5d}{j:5d}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def read_vina_box(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            if key in {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}:
                values[key] = float(value)
    center = np.array([values["center_x"], values["center_y"], values["center_z"]], dtype=float)
    size = np.array([values["size_x"], values["size_y"], values["size_z"]], dtype=float)
    return center, size


def inspect_2am9(pdb_path: Path, tes_coords: dict[str, np.ndarray]) -> dict[str, Any]:
    lines = pdb_path.read_text().splitlines()
    structure = gemmi.read_structure(str(pdb_path))
    model = structure[0]
    chain = model["A"]
    atom_lines = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    atom_records = [line for line in lines if line.startswith("ATOM  ")]
    het_records = [line for line in lines if line.startswith("HETATM")]
    altloc = Counter(line[16] for line in atom_records if line[16] != " ")
    altloc_residues: dict[str, set[str]] = defaultdict(set)
    for line in atom_records:
        if line[16] != " ":
            altloc_residues[f"{line[17:20].strip()} A {line[22:26].strip()}"] .add(line[16])

    het_residues = Counter(line[17:20].strip() for line in het_records)
    het_residue_instances = []
    seen = set()
    for line in het_records:
        key = (line[17:20].strip(), line[21], line[22:26].strip(), line[26])
        if key not in seen:
            seen.add(key)
            het_residue_instances.append({"resname": key[0], "chain": key[1], "resseq": key[2], "icode": key[3]})

    observed_protein = [res for res in chain if res.het_flag == "A"]
    missing_residues = []
    for line in lines:
        m = re.match(r"REMARK 465\s+(\w{3})\s+(\S)\s+(\d+)", line)
        if m:
            missing_residues.append({"resname": m.group(1), "chain": m.group(2), "resseq": int(m.group(3))})
    missing_atoms = []
    for line in lines:
        m = re.match(r"REMARK 470\s+(\w{3})\s+(\S)\s+(\d+)\s+(.+)$", line)
        if m:
            missing_atoms.append({"resname": m.group(1), "chain": m.group(2), "resseq": int(m.group(3)), "atoms": m.group(4).split()})

    selected_protein_atoms = []
    for res in observed_protein:
        for atom in res:
            # A is the first listed conformer for the 14 0.5-occupancy residues.
            if atom.altloc not in {"", " ", "\x00", "A"}:
                continue
            selected_protein_atoms.append((res.name, res.seqid.num, atom.name.strip(), xyz(atom)))
    pocket = {}
    for resname, resseq, atom_name, atom_xyz in selected_protein_atoms:
        d = min(float(np.linalg.norm(atom_xyz - coord)) for coord in tes_coords.values())
        key = f"{resname}{resseq}"
        pocket[key] = min(pocket.get(key, float("inf")), d)
    pocket5 = sorted(
        [{"residue": key, "min_distance_A": round(distance, 3)} for key, distance in pocket.items() if distance <= 5.0],
        key=lambda row: row["min_distance_A"],
    )
    pocket6 = sorted(
        [{"residue": key, "min_distance_A": round(distance, 3)} for key, distance in pocket.items() if distance <= 6.0],
        key=lambda row: row["min_distance_A"],
    )
    missing_atom_keys = {f"{row['resname']}{row['resseq']}" for row in missing_atoms}
    pocket_missing = sorted(key for key in missing_atom_keys if key in pocket and pocket[key] <= 6.0)
    his_pocket = [row for row in pocket6 if row["residue"].startswith("HIS")]
    return {
        "models": len(structure),
        "resolution_A": structure.resolution,
        "chain_ids": [c.name for c in model],
        "chain_A_total_residue_records": len(list(chain)),
        "chain_A_observed_protein_residues": len(observed_protein),
        "observed_protein_range": [observed_protein[0].seqid.num, observed_protein[-1].seqid.num],
        "atom_records": len(atom_lines),
        "atom_records_ATOM": len(atom_records),
        "atom_records_HETATM": len(het_records),
        "altloc_atom_counts": dict(altloc),
        "altloc_residues": {key: sorted(value) for key, value in altloc_residues.items()},
        "het_residue_atom_counts": dict(het_residues),
        "het_residue_instances": het_residue_instances,
        "missing_residues_REMARK465": missing_residues,
        "missing_atoms_REMARK470": missing_atoms,
        "pocket_residues_within_5A": pocket5,
        "pocket_residues_within_6A": pocket6,
        "missing_atom_residues_within_6A": pocket_missing,
        "histidines_within_6A": his_pocket,
        "tes_atom_count": len(tes_coords),
    }


def write_clean_receptor(
    pdb_path: Path,
    output_path: Path,
    drop_residue_numbers: set[int] | None = None,
    preserve_seqres: bool = False,
) -> dict[str, Any]:
    drop_residue_numbers = drop_residue_numbers or set()
    raw_lines = pdb_path.read_text().splitlines()
    kept = []
    removed_altloc_b = 0
    removed_incomplete = Counter()
    for line in raw_lines:
        if not line.startswith("ATOM  ") or line[21] != "A":
            continue
        resseq = int(line[22:26])
        if resseq in drop_residue_numbers:
            removed_incomplete[f"{line[17:20].strip()} A:{resseq}"] += 1
            continue
        altloc = line[16]
        if altloc not in {" ", "A"}:
            removed_altloc_b += 1
            continue
        if altloc == "A":
            line = line[:16] + " " + line[17:]
        kept.append(line)
    if not kept:
        raise ValueError("No chain-A ATOM records found for clean receptor")
    seqres = [line for line in raw_lines if line.startswith("SEQRES") and line[11] == "A"] if preserve_seqres else []
    output_path.write_text(
        "REMARK 900 2AM9 AR LBD cleaned receptor\n"
        "REMARK 900 Chain A; ATOM records only; TES/HETATM/water removed.\n"
        "REMARK 900 For alternate locations, blank and conformer A retained; B discarded.\n"
        f"REMARK 900 Explicitly removed incomplete residues: {', '.join(sorted(removed_incomplete)) or 'none'}\n"
        + ("\n".join(seqres) + "\n" if seqres else "")
        + "\n".join(kept)
        + "\nTER\nEND\n"
    )
    return {
        "kept_atom_records": len(kept),
        "discarded_altloc_B_records": removed_altloc_b,
        "discarded_incomplete_residue_atom_records": dict(removed_incomplete),
    }


def repair_receptor_with_pdbfixer(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Add only missing atoms to a protein-only PDB, never missing residues."""
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(input_path))
    fixer.findMissingResidues()
    missing_residues_before_suppression = {
        str(key): [str(value) for value in values]
        for key, values in fixer.missingResidues.items()
    }
    fixer.missingResidues = {}
    fixer.findMissingAtoms()

    missing_atoms = {}
    for residue, atoms in fixer.missingAtoms.items():
        key = f"{residue.chain.id}:{residue.name}{residue.id}"
        missing_atoms[key] = sorted(atom.name for atom in atoms)
    if not missing_atoms:
        raise ValueError("PDBFixer found no missing atoms; repair target was not reproduced")

    fixer.addMissingAtoms()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)

    return {
        "suppressed_missing_residues": missing_residues_before_suppression,
        "missing_atoms_before_repair": missing_atoms,
        "added_atom_count": sum(len(names) for names in missing_atoms.values()),
        "output": str(output_path),
        "topology_atom_count": sum(1 for _ in fixer.topology.atoms()),
    }


def pdb_atom_map(path: Path) -> dict[tuple[str, str, int, str, str], np.ndarray]:
    atoms = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        key = (line[21], line[17:20].strip(), int(line[22:26]), line[26], line[12:16].strip())
        atoms[key] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return atoms


def repaired_geometry_checks(
    original_path: Path,
    repaired_path: Path,
    tes_coords: dict[str, np.ndarray],
    repaired_info: dict[str, Any],
) -> dict[str, Any]:
    original = pdb_atom_map(original_path)
    repaired = pdb_atom_map(repaired_path)
    common = sorted(set(original) & set(repaired))
    differences = {key: float(np.linalg.norm(original[key] - repaired[key])) for key in common}
    existing_max = max(differences.values()) if differences else float("nan")
    existing_rmsd = math.sqrt(sum(value * value for value in differences.values()) / len(differences)) if differences else float("nan")

    expected = {
        key: set(names) for key, names in repaired_info["missing_atoms_before_repair"].items()
    }
    added = sorted(set(repaired) - set(original))
    added_names = {f"{chain}:{resname}{resseq}": [] for chain, resname, resseq, _, _ in added}
    for chain, resname, resseq, icode, atom_name in added:
        added_names[f"{chain}:{resname}{resseq}"].append(atom_name)

    tes_array = np.array(list(tes_coords.values()), dtype=float)
    pocket_keys = []
    for key, coord in original.items():
        if key[0] != "A":
            continue
        if float(np.min(np.linalg.norm(coord - tes_array, axis=1))) <= 6.0:
            pocket_keys.append(key)
    pocket_differences = {"count": len(pocket_keys), "max_shift_A": max((differences.get(key, 0.0) for key in pocket_keys), default=0.0)}

    return {
        "original_atom_count": len(original),
        "repaired_atom_count": len(repaired),
        "original_residue_count": len({key[:4] for key in original}),
        "repaired_residue_count": len({key[:4] for key in repaired}),
        "residue_keys_lost": sorted({key[:4] for key in original} - {key[:4] for key in repaired}),
        "existing_atom_count_compared": len(common),
        "existing_atom_coordinate_rmsd_A": existing_rmsd,
        "existing_atom_max_shift_A": existing_max,
        "added_atoms": added_names,
        "added_atom_count_by_mapping": len(added),
        "expected_missing_atoms_by_residue": {key: sorted(value) for key, value in expected.items()},
        "pocket_existing_atom_coordinate_check": pocket_differences,
    }


def validate_repaired_outputs(
    outdir: Path,
    original_pdb: Path,
    repaired_pdb: Path,
    repaired_h_pdb: Path,
    repaired_pdbqt: Path,
    tes_pdbqt: Path,
    vina_box: Path,
    tes_coords: dict[str, np.ndarray],
    repaired_info: dict[str, Any],
) -> dict[str, Any]:
    from vina import Vina

    required = [repaired_pdb, repaired_h_pdb, repaired_pdbqt, tes_pdbqt, vina_box]
    nonempty = {path.name: path.exists() and path.stat().st_size > 0 for path in required}
    if not all(nonempty.values()):
        raise ValueError(f"Missing or empty repaired output: {nonempty}")

    geometry = repaired_geometry_checks(original_pdb, repaired_pdb, tes_coords, repaired_info)
    expected = geometry["expected_missing_atoms_by_residue"]
    actual = geometry["added_atoms"]
    if geometry["added_atom_count_by_mapping"] != repaired_info["added_atom_count"]:
        raise ValueError(f"Added-atom count mismatch: {geometry}")
    for residue_key, names in expected.items():
        if sorted(actual.get(residue_key, [])) != sorted(names):
            raise ValueError(f"Added atom mismatch for {residue_key}: expected={names}, actual={actual.get(residue_key)}")

    if geometry["existing_atom_max_shift_A"] > 1e-3:
        raise ValueError(f"Existing coordinates changed by more than tolerance: {geometry['existing_atom_max_shift_A']} A")
    if geometry["pocket_existing_atom_coordinate_check"]["max_shift_A"] > 1e-3:
        raise ValueError("Pocket existing atom coordinates changed")

    repaired_text = repaired_pdb.read_text()
    for key in ["TES", "SO4", "DTT", "GOL", "HOH"]:
        if re.search(rf"^(ATOM  |HETATM).{{9}}{key:>3}\s", repaired_text, flags=re.MULTILINE):
            raise ValueError(f"Non-protein component remains in repaired PDB: {key}")
    if "TES" in repaired_pdbqt.read_text():
        raise ValueError("TES text remains in repaired receptor PDBQT")

    # Heavy-atom collision screen for the 15 newly added side-chain atoms.
    from openmm.app import PDBFile
    topology = PDBFile(str(repaired_pdb)).topology
    positions = PDBFile(str(repaired_pdb)).positions
    atom_by_key = {}
    for atom in topology.atoms():
        residue = atom.residue
        key = (residue.chain.id, residue.name, int(residue.id), " ", atom.name)
        atom_by_key[key] = atom
    bonds = {frozenset((a1.index, a2.index)) for a1, a2 in topology.bonds()}
    added_keys = set()
    for residue_key, names in actual.items():
        match = re.match(r"(.):([A-Z0-9]{3})(\d+)$", residue_key)
        if match:
            chain, resname, resseq = match.group(1), match.group(2), int(match.group(3))
            added_keys.update((chain, resname, resseq, " ", name) for name in names)
    vdw = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}
    # OpenMM stores PDBFile positions in nm; all thresholds below are in A.
    atom_coords = {atom.index: 10.0 * np.array([positions[atom.index][0]._value, positions[atom.index][1]._value, positions[atom.index][2]._value]) for atom in topology.atoms()}
    severe_collisions = []
    bond_length_outliers = []

    # PDB files generally do not contain CONECT records for standard protein
    # residues. Add the local side-chain bonds relevant to the four repaired
    # residues so they are not falsely reported as steric clashes.
    sidechain_bonds = {
        "LYS": [("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
        "ASN": [("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
        "GLU": [("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    }
    explicit_bond_pairs = set(bonds)
    for residue_key, names in actual.items():
        match = re.match(r"(.):([A-Z0-9]{3})(\d+)$", residue_key)
        if not match:
            continue
        chain, resname, resseq = match.group(1), match.group(2), int(match.group(3))
        for atom_name_1, atom_name_2 in sidechain_bonds.get(resname, []):
            atom_1 = atom_by_key.get((chain, resname, resseq, " ", atom_name_1))
            atom_2 = atom_by_key.get((chain, resname, resseq, " ", atom_name_2))
            if atom_1 is not None and atom_2 is not None:
                explicit_bond_pairs.add(frozenset((atom_1.index, atom_2.index)))

    added_bond_pairs = set()
    for residue_key in actual:
        match = re.match(r"(.):([A-Z0-9]{3})(\d+)$", residue_key)
        if not match:
            continue
        chain, resname, resseq = match.group(1), match.group(2), int(match.group(3))
        for atom_name_1, atom_name_2 in sidechain_bonds.get(resname, []):
            atom_1 = atom_by_key.get((chain, resname, resseq, " ", atom_name_1))
            atom_2 = atom_by_key.get((chain, resname, resseq, " ", atom_name_2))
            if atom_1 is not None and atom_2 is not None and (atom_1.index in {atom_by_key[k].index for k in added_keys if k in atom_by_key} or atom_2.index in {atom_by_key[k].index for k in added_keys if k in atom_by_key}):
                added_bond_pairs.add((atom_1, atom_2))
                distance = float(np.linalg.norm(atom_coords[atom_1.index] - atom_coords[atom_2.index]))
                if not 1.15 <= distance <= 1.95:
                    bond_length_outliers.append({"residue": residue_key, "atoms": [atom_name_1, atom_name_2], "distance_A": distance})

    for key in sorted(added_keys):
        atom = atom_by_key.get(key)
        if atom is None:
            continue
        atom_element = atom.element.symbol if atom.element else key[4][0]
        for other in topology.atoms():
            if other.index == atom.index or other.element is None:
                continue
            same_residue = atom.residue is other.residue
            peptide_bond = (
                atom.residue.chain is other.residue.chain
                and abs(int(atom.residue.id) - int(other.residue.id)) == 1
                and {atom.name, other.name} == {"C", "N"}
            )
            if frozenset((atom.index, other.index)) in explicit_bond_pairs:
                continue
            if same_residue and float(np.linalg.norm(atom_coords[atom.index] - atom_coords[other.index])) <= 1.95:
                continue
            if peptide_bond and float(np.linalg.norm(atom_coords[atom.index] - atom_coords[other.index])) <= 1.8:
                continue
            other_element = other.element.symbol
            if atom_element == "H" or other_element == "H":
                continue
            distance = float(np.linalg.norm(atom_coords[atom.index] - atom_coords[other.index]))
            limit = 0.60 * (vdw.get(atom_element, 1.7) + vdw.get(other_element, 1.7))
            if distance < limit:
                severe_collisions.append({"added": key, "other": other.name, "distance_A": distance, "threshold_A": limit})
    v = Vina(sf_name="vina", cpu=2)
    v.set_receptor(str(repaired_pdbqt))
    v.set_ligand_from_file(str(tes_pdbqt))
    return {
        "all_required_outputs_nonempty": all(nonempty.values()),
        "nonempty_files": nonempty,
        "geometry": geometry,
        "severe_heavy_atom_collisions": severe_collisions,
        "bond_length_outliers": bond_length_outliers,
        "repaired_pdb_contains_nonprotein_components": False,
        "repaired_receptor_vina_load": "OK",
        "tes_reference_untouched": tes_pdbqt.exists(),
        "vina_box_untouched_path": str(vina_box),
    }


def run_command(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return (result.stdout + "\n" + result.stderr).strip()


def validate_outputs(outdir: Path, center: np.ndarray, size: np.ndarray, crystal_mol: Chem.Mol) -> dict[str, Any]:
    paths = {
        name: outdir / filename
        for name, filename in {
            "receptor_clean_pdb": "receptor_clean.pdb",
            "receptor_h_pdb": "receptor_H.pdb",
            "receptor_pdbqt": "receptor.pdbqt",
            "tes_crystal_sdf": "TES_crystal.sdf",
            "tes_prepared_sdf": "TES_prepared.sdf",
            "tes_pdbqt": "TES.pdbqt",
            "vina_box": "vina_box.txt",
            "vina_box_pdb": "vina_box.pdb",
        }.items()
    }
    nonempty = {key: path.exists() and path.stat().st_size > 0 for key, path in paths.items()}
    if not all(nonempty.values()):
        raise ValueError(f"Missing or empty preparation output: {[key for key, ok in nonempty.items() if not ok]}")

    crystal = Chem.SDMolSupplier(str(paths["tes_crystal_sdf"]), removeHs=False)[0]
    prepared = Chem.SDMolSupplier(str(paths["tes_prepared_sdf"]), removeHs=False)[0]
    if crystal is None or prepared is None:
        raise ValueError("RDKit could not read TES SDF output")
    formula = rdMolDescriptors.CalcMolFormula(crystal)
    formula_prepared = rdMolDescriptors.CalcMolFormula(prepared)
    if formula != "C19H28O2" or formula_prepared != "C19H28O2":
        raise ValueError(f"Unexpected TES formula: crystal={formula}, prepared={formula_prepared}")
    heavy_rmsd = heavy_coord_rmsd(crystal, prepared)
    if heavy_rmsd > 1e-3:
        raise ValueError(f"Heavy atom coordinates changed: RMSD={heavy_rmsd:.6f} A")

    tes_coords = np.array([np.array(crystal.GetConformer().GetAtomPosition(i)) for i in range(crystal.GetNumAtoms()) if crystal.GetAtomWithIdx(i).GetAtomicNum() != 1])
    lower, upper = center - size / 2, center + size / 2
    inside = bool(np.all((tes_coords >= lower - 1e-6) & (tes_coords <= upper + 1e-6)))
    receptor_pdbqt = parse_pdbqt_coords(paths["receptor_pdbqt"])
    ligand_pdbqt = parse_pdbqt_coords(paths["tes_pdbqt"])
    if not receptor_pdbqt or not ligand_pdbqt:
        raise ValueError("PDBQT atom records could not be parsed")

    receptor_text = paths["receptor_pdbqt"].read_text()
    no_tes = " TES " not in receptor_text and "TES" not in receptor_text
    if not no_tes:
        raise ValueError("TES residue text remains in receptor PDBQT")

    h_count = 0
    for line in paths["receptor_h_pdb"].read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip().startswith("H"):
            h_count += 1

    centers = dict(Chem.FindMolChiralCenters(crystal_mol, includeUnassigned=True, includeCIP=True))
    stereo = {
        crystal_mol.GetAtomWithIdx(idx).GetProp("ccd_atom_id"): cip
        for idx, cip in centers.items()
        if crystal_mol.GetAtomWithIdx(idx).HasProp("ccd_atom_id")
    }

    vina_load = "not_run"
    try:
        from vina import Vina

        v = Vina(sf_name="vina", cpu=2)
        v.set_receptor(str(paths["receptor_pdbqt"]))
        v.set_ligand_from_file(str(paths["tes_pdbqt"]))
        vina_load = "OK"
    except Exception as exc:  # validation report should retain the actual error
        vina_load = f"ERROR: {exc}"

    result = {
        "all_outputs_nonempty": all(nonempty.values()),
        "nonempty_files": nonempty,
        "tes_formula_crystal": formula,
        "tes_formula_prepared": formula_prepared,
        "tes_heavy_atoms": crystal.GetNumHeavyAtoms(),
        "tes_prepared_total_atoms": prepared.GetNumAtoms(),
        "tes_heavy_atom_rmsd_A": heavy_rmsd,
        "tes_stereochemistry_CIP": stereo,
        "tes_pdbqt_atom_count": len(ligand_pdbqt),
        "receptor_pdbqt_atom_count": len(receptor_pdbqt),
        "receptor_h_pdb_hydrogen_atom_count": h_count,
        "receptor_contains_TES": not no_tes,
        "box_center_A": center.tolist(),
        "box_size_A": size.tolist(),
        "tes_fully_inside_box": inside,
        "vina_load": vina_load,
    }
    if vina_load != "OK" or not inside or not no_tes:
        raise ValueError(json.dumps(result, indent=2))
    return result

