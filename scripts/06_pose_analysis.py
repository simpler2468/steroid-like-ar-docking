"""Step 4.2 pose geometry and protein-ligand interaction candidate analysis.

The script reads the existing 24 top-ranked SDF poses and the repaired 2AM9
receptor.  It does not run Vina, score poses, alter the receptor, or alter the
docking box.  Interaction labels are geometry-based candidates, not
experimental mechanistic claims.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS, rdMolDescriptors
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
TOP = ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv"
READY = ROOT / "results" / "ligand_preparation" / "docking_ready.csv"
QC = ROOT / "results" / "docking_2AM9" / "docking_qc.csv"
TOP_SDF = ROOT / "results" / "docking_2AM9" / "top_sdf"
OUT = ROOT / "results" / "pose_analysis"
STRUCTURES = OUT / "structures"
FIG = ROOT / "figures" / "pose_analysis"
RECEPTOR_PDB = PREP / "receptor_repaired.pdb"
RECEPTOR_H_PDB = PREP / "receptor_repaired_H.pdb"
TES_SDF = PREP / "TES_crystal.sdf"
BOX = PREP / "vina_box_repaired.txt"

KEY_RESIDUES = {"A:ASN705", "A:GLN711", "A:ARG752", "A:THR877"}
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98}
INTERACTION_TYPES = [
    "hydrogen_bond_candidate",
    "polar_contact",
    "hydrophobic_contact",
    "close_vdw_contact",
    "pi_contact_candidate",
    "halogen_contact_candidate",
    "steric_clash",
]


def load_sdf(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    mol = supplier[0] if len(supplier) else None
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Cannot load 3D SDF: {path}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.asarray(conf.GetPositions(), dtype=float)


def element_from_pdb(line: str) -> str:
    element = line[76:78].strip().upper()
    if element:
        return element
    name = line[12:16].strip().upper()
    if name.startswith("CL"):
        return "CL"
    if name.startswith("BR"):
        return "BR"
    return name[:1]


def parse_pdb(path: Path, include_h: bool) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = element_from_pdb(line)
        if not include_h and element == "H":
            continue
        try:
            residue_number = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        except ValueError:
            continue
        chain = line[21].strip() or "_"
        resname = line[17:20].strip().upper()
        icode = line[26].strip()
        residue_key = f"{chain}:{resname}{residue_number}{icode}"
        atoms.append({
            "atom_name": line[12:16].strip(),
            "element": element,
            "xyz": xyz,
            "chain": chain,
            "resname": resname,
            "residue_number": residue_number,
            "icode": icode,
            "residue_key": residue_key,
        })
    return atoms


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return float("nan")
    value = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def residue_is_acceptor(atom: dict[str, Any]) -> bool:
    e, name, res = atom["element"], atom["atom_name"].upper(), atom["resname"]
    if e not in {"O", "N", "S"}:
        return False
    if e == "O":
        return not name.startswith("H")
    if e == "S":
        return True
    # Conservative nitrogen acceptor rules for standard amino acids.
    if name == "N" or res in {"LYS", "ARG"}:
        return False
    if (res, name) in {("ASN", "ND2"), ("GLN", "NE2"), ("PRO", "N")}:
        return False
    return True


def attach_receptor_hydrogens(heavy: list[dict[str, Any]], all_atoms: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    hydrogens = [a for a in all_atoms if a["element"] == "H"]
    donors: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, atom in enumerate(heavy):
        if atom["element"] not in {"N", "O", "S"}:
            continue
        for h in hydrogens:
            if h["residue_key"] != atom["residue_key"]:
                continue
            if np.linalg.norm(atom["xyz"] - h["xyz"]) <= 1.35:
                donors[i].append(h)
    return donors


def ligand_features(mol: Chem.Mol) -> dict[str, Any]:
    xyz = coords(mol)
    heavy = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() > 1]
    donors: list[tuple[int, int]] = []
    acceptors: list[int] = []
    polar: list[int] = []
    halogens: list[tuple[int, int]] = []
    carbonyl_oxygen: list[int] = []
    hydroxyl_oxygen: list[int] = []
    aromatic_atoms: list[int] = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        symbol = atom.GetSymbol().upper()
        if symbol in {"O", "N", "S"}:
            polar.append(i)
        if symbol in {"O", "N", "S"} and atom.GetTotalNumHs() > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    donors.append((i, nbr.GetIdx()))
            if symbol == "O":
                hydroxyl_oxygen.append(i)
        if symbol == "O" and atom.GetFormalCharge() <= 0 and any(bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetAtomicNum() == 6 for bond in atom.GetBonds()):
            carbonyl_oxygen.append(i)
            acceptors.append(i)
        elif symbol == "O" and atom.GetFormalCharge() <= 0 and atom.GetTotalNumHs() == 0:
            acceptors.append(i)
        elif symbol in {"N", "S"} and atom.GetFormalCharge() <= 0 and atom.GetTotalNumHs() == 0:
            acceptors.append(i)
        if atom.GetIsAromatic():
            aromatic_atoms.append(i)
        if symbol in {"F", "CL", "BR", "I"}:
            carbon_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
            for c in carbon_nbrs:
                halogens.append((i, c))
    ring_centroids: list[np.ndarray] = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            ring_centroids.append(xyz[list(ring)].mean(axis=0))
    return {
        "xyz": xyz,
        "heavy_indices": heavy,
        "donors": donors,
        "acceptors": sorted(set(acceptors)),
        "polar": sorted(set(polar)),
        "halogens": halogens,
        "carbonyl_oxygen": sorted(set(carbonyl_oxygen)),
        "hydroxyl_oxygen": sorted(set(hydroxyl_oxygen)),
        "aromatic_atoms": aromatic_atoms,
        "aromatic_ring_centroids": ring_centroids,
    }


def mcs_query(mols: list[Chem.Mol]) -> Chem.Mol | None:
    result = rdFMCS.FindMCS(
        mols,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchChiralTag=False,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        timeout=45,
    )
    if result.canceled or not result.smartsString:
        return None
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None or query.GetNumAtoms() < 6:
        return None
    return query


def principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    if len(points) < 2:
        return np.eye(3)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.T
    if axes.shape[1] < 3:
        padded = np.eye(3)
        padded[:, : axes.shape[1]] = axes
        axes = padded
    return axes[:, :3]


def core_metrics(mol: Chem.Mol, native: Chem.Mol, query: Chem.Mol | None) -> dict[str, Any]:
    xyz = coords(mol)
    native_xyz = coords(native)
    if query is None:
        return {"core_atom_count": 0, "core_centroid_x": np.nan, "core_centroid_y": np.nan, "core_centroid_z": np.nan, "core_centroid_to_TES_A": np.nan, "core_raw_rmsd_to_TES_A": np.nan, "core_main_axis_angle_deg": np.nan, "core_match_status": "not_available"}
    match = mol.GetSubstructMatch(query)
    native_match = native.GetSubstructMatch(query)
    if not match or not native_match or len(match) != len(native_match):
        return {"core_atom_count": 0, "core_centroid_x": np.nan, "core_centroid_y": np.nan, "core_centroid_z": np.nan, "core_centroid_to_TES_A": np.nan, "core_raw_rmsd_to_TES_A": np.nan, "core_main_axis_angle_deg": np.nan, "core_match_status": "no_match"}
    core = xyz[list(match)]
    native_core = native_xyz[list(native_match)]
    centroid = core.mean(axis=0)
    native_centroid = native_core.mean(axis=0)
    axes = principal_axes(core)
    native_axes = principal_axes(native_core)
    angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(axes[:, 0], native_axes[:, 0]))), -1.0, 1.0))))
    return {"core_atom_count": len(match), "core_centroid_x": float(centroid[0]), "core_centroid_y": float(centroid[1]), "core_centroid_z": float(centroid[2]), "core_centroid_to_TES_A": float(np.linalg.norm(centroid - native_centroid)), "core_raw_rmsd_to_TES_A": float(np.sqrt(np.mean(np.sum((core - native_core) ** 2, axis=1)))), "core_main_axis_angle_deg": angle, "core_match_status": "matched"}


def detect_functional_group_centroids(mol: Chem.Mol, features: dict[str, Any]) -> dict[str, np.ndarray | None]:
    xyz = features["xyz"]
    groups: dict[str, np.ndarray | None] = {}
    for name, idxs in [("carbonyl_oxygen", features["carbonyl_oxygen"]), ("hydroxyl_oxygen", features["hydroxyl_oxygen"]), ("aromatic_atoms", features["aromatic_atoms"])]:
        groups[name] = xyz[idxs].mean(axis=0) if idxs else None
    return groups


def parse_box(path: Path) -> np.ndarray:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            try:
                values[key.strip()] = float(value.strip())
            except ValueError:
                pass
    return np.array([values["center_x"], values["center_y"], values["center_z"]], dtype=float)


def min_distance_to_residue(ligand_heavy_xyz: np.ndarray, receptor_atoms: list[dict[str, Any]], residue_key: str) -> float:
    residue_xyz = np.array([a["xyz"] for a in receptor_atoms if a["residue_key"] == residue_key], dtype=float)
    if residue_xyz.size == 0:
        return float("nan")
    return float(cdist(ligand_heavy_xyz, residue_xyz).min())


def residue_interactions(ligand: Chem.Mol, ligand_feature: dict[str, Any], receptor: list[dict[str, Any]], receptor_donors: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    lig_xyz = ligand_feature["xyz"]
    heavy_receptor = receptor
    by_res: dict[str, list[int]] = defaultdict(list)
    for j, atom in enumerate(heavy_receptor):
        by_res[atom["residue_key"]].append(j)
    rows = []
    for residue_key, indices in by_res.items():
        residue_atoms = [heavy_receptor[j] for j in indices]
        dmat = cdist(lig_xyz[ligand_feature["heavy_indices"]], np.array([a["xyz"] for a in residue_atoms]))
        min_distance = float(dmat.min()) if dmat.size else float("inf")
        types: set[str] = set()
        counts: dict[str, int] = defaultdict(int)
        resname = residue_atoms[0]["resname"]
        # Heavy atom proximity, steric clashes and close vdW contacts.
        for li in ligand_feature["heavy_indices"]:
            le = ligand.GetAtomWithIdx(li).GetSymbol().upper()
            for rj in indices:
                ratom = heavy_receptor[rj]
                distance = float(np.linalg.norm(lig_xyz[li] - ratom["xyz"]))
                if distance < 2.0:
                    types.add("steric_clash")
                    counts["steric_clash"] += 1
                rsum = VDW_RADII.get(le, 1.7) + VDW_RADII.get(ratom["element"], 1.7)
                if distance <= rsum + 0.35:
                    types.add("close_vdw_contact")
                    counts["close_vdw_contact"] += 1
        # Hydrogen-bond candidates with explicit donor geometry.
        lig_donors = ligand_feature["donors"]
        lig_acceptors = ligand_feature["acceptors"]
        for ld, lh in lig_donors:
            for rj in indices:
                ratom = heavy_receptor[rj]
                if not residue_is_acceptor(ratom):
                    continue
                da = float(np.linalg.norm(lig_xyz[ld] - ratom["xyz"]))
                ha = float(np.linalg.norm(lig_xyz[lh] - ratom["xyz"]))
                if da <= 3.6 and ha <= 2.7 and angle_deg(lig_xyz[ld], lig_xyz[lh], ratom["xyz"]) >= 110:
                    types.add("hydrogen_bond_candidate")
                    counts["hydrogen_bond_candidate"] += 1
        for rj, hydrogens in receptor_donors.items():
            if rj not in indices:
                continue
            donor_atom = heavy_receptor[rj]
            for h in hydrogens:
                for la in lig_acceptors:
                    da = float(np.linalg.norm(donor_atom["xyz"] - lig_xyz[la]))
                    ha = float(np.linalg.norm(h["xyz"] - lig_xyz[la]))
                    if da <= 3.6 and ha <= 2.7 and angle_deg(donor_atom["xyz"], h["xyz"], lig_xyz[la]) >= 110:
                        types.add("hydrogen_bond_candidate")
                        counts["hydrogen_bond_candidate"] += 1
        # Polar contacts are intentionally less specific than H-bond candidates.
        receptor_polar = [j for j in indices if heavy_receptor[j]["element"] in {"O", "N", "S"}]
        for li in ligand_feature["polar"]:
            for rj in receptor_polar:
                if np.linalg.norm(lig_xyz[li] - heavy_receptor[rj]["xyz"]) <= 4.0:
                    types.add("polar_contact")
                    counts["polar_contact"] += 1
        # Hydrophobic carbon contacts.
        if resname in HYDROPHOBIC_RESIDUES:
            for li in ligand_feature["heavy_indices"]:
                if ligand.GetAtomWithIdx(li).GetSymbol().upper() not in {"C", "S"}:
                    continue
                for rj in indices:
                    if heavy_receptor[rj]["element"] == "C" and np.linalg.norm(lig_xyz[li] - heavy_receptor[rj]["xyz"]) <= 4.2:
                        types.add("hydrophobic_contact")
                        counts["hydrophobic_contact"] += 1
        # Aromatic ring centroid proximity; reported as candidate only.
        if resname in AROMATIC_RESIDUES and ligand_feature["aromatic_ring_centroids"]:
            ring_atoms = [a for a in residue_atoms if a["atom_name"].upper() in {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "CE3", "CZ2", "CZ3", "CH2", "ND1", "CG", "NE2"}]
            if len(ring_atoms) >= 3:
                centroid = np.array([a["xyz"] for a in ring_atoms]).mean(axis=0)
                if min(np.linalg.norm(c - centroid) for c in ligand_feature["aromatic_ring_centroids"]) <= 5.5:
                    types.add("pi_contact_candidate")
                    counts["pi_contact_candidate"] += 1
        # Halogen contact with angle C-X...A.
        for x_idx, c_idx in ligand_feature["halogens"]:
            for rj in indices:
                ratom = heavy_receptor[rj]
                if ratom["element"] not in {"O", "N", "S"}:
                    continue
                xa = float(np.linalg.norm(lig_xyz[x_idx] - ratom["xyz"]))
                if xa <= 3.8 and angle_deg(lig_xyz[c_idx], lig_xyz[x_idx], ratom["xyz"]) >= 135:
                    types.add("halogen_contact_candidate")
                    counts["halogen_contact_candidate"] += 1
        if min_distance <= 4.5 or types:
            row = {"residue_key": residue_key, "resname": resname, "min_heavy_distance_A": min_distance, "interaction_types": ";".join(sorted(types)), "is_key_residue": residue_key in KEY_RESIDUES}
            for interaction_type in INTERACTION_TYPES:
                row[f"{interaction_type}_count"] = int(counts.get(interaction_type, 0))
            rows.append(row)
    return rows


def make_interaction_table(compound_id: str, base: dict[str, Any], interactions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    global_counts = {t: 0 for t in INTERACTION_TYPES}
    for interaction in interactions:
        row = {"compound_id": compound_id, **base, **interaction}
        rows.append(row)
        for interaction_type in INTERACTION_TYPES:
            global_counts[interaction_type] += int(interaction[f"{interaction_type}_count"])
    return rows, global_counts


def add_key_fingerprint_columns(fp: pd.DataFrame) -> pd.DataFrame:
    key_rows = []
    for key in sorted(KEY_RESIDUES):
        for interaction_type in INTERACTION_TYPES:
            key_rows.append(f"{key}__{interaction_type}")
    for column in key_rows:
        if column not in fp.columns:
            fp[column] = 0
    return fp


def family_assignments(pose: pd.DataFrame, fp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    descriptor_cols = ["ligand_centroid_to_TES_centroid_A", "core_centroid_to_TES_A", "core_raw_rmsd_to_TES_A", "core_main_axis_angle_deg", "ligand_centroid_to_box_center_A", "min_receptor_heavy_distance_A"] + [c for c in fp.columns if "__" in c]
    feature = pose.set_index("compound_id")[descriptor_cols].copy()
    feature = feature.replace([np.inf, -np.inf], np.nan).fillna(feature.median(numeric_only=True)).fillna(0.0)
    scaled = StandardScaler().fit_transform(feature.to_numpy(dtype=float))
    best = None
    for k in range(2, min(6, len(feature) - 1)):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled, labels))
        if best is None or score > best[0]:
            best = (score, k, labels)
    if best is None:
        best = (float("nan"), 1, np.zeros(len(feature), dtype=int))
    silhouette, k, labels = best
    assignments = pd.DataFrame({"compound_id": feature.index, "pose_family": labels + 1, "family_selection_silhouette": silhouette})
    return assignments, {"method": "AgglomerativeClustering on standardized pose descriptors and key-residue interaction bits", "selected_family_count": int(k), "selected_silhouette": silhouette, "candidate_k_range": [2, min(5, len(feature) - 1)], "interpretation": "Exploratory organization, not supervised classification."}


def write_pdb(mol: Chem.Mol, path: Path, name: str) -> None:
    copy = Chem.Mol(mol)
    copy.SetProp("_Name", name)
    Chem.MolToPDBFile(copy, str(path), flavor=0)


def main() -> None:
    for path in [RECEPTOR_PDB, RECEPTOR_H_PDB, TES_SDF, TOP, READY, QC, BOX]:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    heavy_receptor = parse_pdb(RECEPTOR_PDB, include_h=False)
    all_receptor = parse_pdb(RECEPTOR_H_PDB, include_h=True)
    receptor_donors = attach_receptor_hydrogens(heavy_receptor, all_receptor)
    receptor_xyz = np.array([a["xyz"] for a in heavy_receptor])
    box_center = parse_box(BOX)
    native = load_sdf(TES_SDF)
    native_feature = ligand_features(native)
    ready = pd.read_csv(READY)
    top = pd.read_csv(TOP)
    qc = pd.read_csv(QC)
    top = top[(top.pose_rank == 1) & (top.docking_status == "success")][["compound_id", "vina_score_kcal_mol", "pose_pdbqt", "top_pose_sdf"]].copy()
    ready = ready[ready.docking_status == "docking_ready"].copy()
    if len(top) != 24 or len(ready) != 24 or set(top.compound_id) != set(ready.compound_id):
        raise ValueError("Step 4.2 requires exactly the 24 docking-ready top poses")
    if not bool(qc.set_index("compound_id").loc[sorted(top.compound_id)].input_ligand_vina_load_ok.all()):
        raise ValueError("At least one top pose lacks the recorded Step 3.2 input ligand QC")
    data = top.merge(ready[["compound_id", "canonical_isomeric_smiles", "original_smiles", "cluster_id", "row_index", "split", "resolution_status"]], on="compound_id", how="left", validate="one_to_one")
    data = data.sort_values(["cluster_id", "row_index"]).reset_index(drop=True)
    mols = []
    for _, row in data.iterrows():
        path = TOP_SDF / f"{row.compound_id}.sdf"
        mol = load_sdf(path)
        if Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=True) != Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(row.canonical_isomeric_smiles)), isomericSmiles=True):
            raise ValueError(f"Top SDF structure mismatch: {row.compound_id}")
        mols.append(mol)
        write_pdb(mol, STRUCTURES / f"{row.compound_id}.pdb", row.compound_id)
    write_pdb(native, STRUCTURES / "TES_crystal.pdb", "TES_native")

    common_query = mcs_query(mols + [native])
    mcs_smarts = Chem.MolToSmarts(common_query) if common_query is not None else None
    native_groups = detect_functional_group_centroids(native, native_feature)
    interaction_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for row, mol in zip(data.itertuples(index=False), mols):
        feature = ligand_features(mol)
        xyz = feature["xyz"]
        heavy_xyz = xyz[feature["heavy_indices"]]
        centroid = heavy_xyz.mean(axis=0)
        bbox_min, bbox_max = heavy_xyz.min(axis=0), heavy_xyz.max(axis=0)
        native_heavy_xyz = native_feature["xyz"][native_feature["heavy_indices"]]
        min_to_receptor = float(cdist(heavy_xyz, receptor_xyz).min())
        contacts = residue_interactions(mol, feature, heavy_receptor, receptor_donors)
        base = {"cluster_id": int(row.cluster_id), "row_index": int(row.row_index), "vina_score_kcal_mol": float(row.vina_score_kcal_mol)}
        contact_rows, counts = make_interaction_table(row.compound_id, base, contacts)
        interaction_rows.extend(contact_rows)
        group_centroids = detect_functional_group_centroids(mol, feature)
        pose = {
            "compound_id": row.compound_id,
            "cluster_id": int(row.cluster_id),
            "row_index": int(row.row_index),
            "split": row.split,
            "vina_score_kcal_mol": float(row.vina_score_kcal_mol),
            "pose_sdf": str((TOP_SDF / f"{row.compound_id}.sdf").relative_to(ROOT)),
            "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "centroid_z": float(centroid[2]),
            "ligand_centroid_to_TES_centroid_A": float(np.linalg.norm(centroid - native_heavy_xyz.mean(axis=0))),
            "ligand_centroid_to_box_center_A": float(np.linalg.norm(centroid - box_center)),
            "bbox_x_A": float(bbox_max[0] - bbox_min[0]), "bbox_y_A": float(bbox_max[1] - bbox_min[1]), "bbox_z_A": float(bbox_max[2] - bbox_min[2]),
            "bbox_volume_A3": float(np.prod(bbox_max - bbox_min)),
            "heavy_atom_count": int(len(feature["heavy_indices"])),
            "min_receptor_heavy_distance_A": min_to_receptor,
            "key_residue_min_distances": ";".join(f"{key}={min_distance_to_residue(heavy_xyz, heavy_receptor, key):.3f}" for key in sorted(KEY_RESIDUES)),
            "hbond_candidate_count": int(counts["hydrogen_bond_candidate"]),
            "polar_contact_count": int(counts["polar_contact"]),
            "hydrophobic_contact_count": int(counts["hydrophobic_contact"]),
            "close_vdw_contact_count": int(counts["close_vdw_contact"]),
            "pi_contact_candidate_count": int(counts["pi_contact_candidate"]),
            "halogen_contact_candidate_count": int(counts["halogen_contact_candidate"]),
            "steric_clash_count": int(counts["steric_clash"]),
            "carbonyl_oxygen_count": len(feature["carbonyl_oxygen"]),
            "hydroxyl_oxygen_count": len(feature["hydroxyl_oxygen"]),
            "aromatic_atom_count": len(feature["aromatic_atoms"]),
        }
        pose.update(core_metrics(mol, native, common_query))
        for group_name, group_centroid in group_centroids.items():
            native_centroid = native_groups.get(group_name)
            pose[f"{group_name}_to_TES_same_group_A"] = float(np.linalg.norm(group_centroid - native_centroid)) if group_centroid is not None and native_centroid is not None else np.nan
        global_rows.append({"compound_id": row.compound_id, "cluster_id": int(row.cluster_id), "vina_score_kcal_mol": float(row.vina_score_kcal_mol), **counts})
        pose_rows.append(pose)

    pose_df = pd.DataFrame(pose_rows)
    contacts_df = pd.DataFrame(interaction_rows)
    global_df = pd.DataFrame(global_rows)
    fp_records = []
    for compound_id in pose_df.compound_id:
        row = {"compound_id": compound_id}
        sub = contacts_df[contacts_df.compound_id == compound_id]
        for _, contact in sub.iterrows():
            for interaction_type in INTERACTION_TYPES:
                if int(contact[f"{interaction_type}_count"]) > 0:
                    row[f"{contact.residue_key}__{interaction_type}"] = 1
        fp_records.append(row)
    fp_df = pd.DataFrame(fp_records).fillna(0)
    fp_df = add_key_fingerprint_columns(fp_df)
    for col in fp_df.columns:
        if col != "compound_id":
            fp_df[col] = fp_df[col].astype(int).clip(0, 1)
    pose_df = pose_df.merge(fp_df, on="compound_id", how="left", validate="one_to_one")
    assignments, family_meta = family_assignments(pose_df, fp_df)
    pose_df = pose_df.merge(assignments, on="compound_id", how="left", validate="one_to_one")

    # Interaction fingerprint Jaccard similarity.
    bit_cols = [c for c in fp_df.columns if c != "compound_id"]
    fp_index = fp_df.set_index("compound_id")
    fp_sim = np.eye(len(fp_index), dtype=float)
    ids = list(fp_index.index)
    for i, j in combinations(range(len(ids)), 2):
        a, b = fp_index.loc[ids[i], bit_cols].to_numpy(dtype=bool), fp_index.loc[ids[j], bit_cols].to_numpy(dtype=bool)
        union = np.logical_or(a, b).sum()
        sim = float(np.logical_and(a, b).sum() / union) if union else 1.0
        fp_sim[i, j] = fp_sim[j, i] = sim
    pd.DataFrame(fp_sim, index=ids, columns=ids).to_csv(OUT / "interaction_fingerprint_similarity.csv", encoding="utf-8-sig")

    pose_out_cols = [c for c in pose_df.columns if c not in {"mol"}]
    pose_df[pose_out_cols].to_csv(OUT / "pose_descriptors.csv", index=False, encoding="utf-8-sig")
    contacts_df.to_csv(OUT / "ligand_residue_contacts.csv", index=False, encoding="utf-8-sig")
    global_df.to_csv(OUT / "interaction_summary.csv", index=False, encoding="utf-8-sig")
    fp_df.to_csv(OUT / "interaction_fingerprints.csv", index=False, encoding="utf-8-sig")

    # Representative cases requested in the protocol.
    family_map = assignments.set_index("compound_id").pose_family.to_dict()
    score_map = data.set_index("compound_id").vina_score_kcal_mol.to_dict()
    cases = [
        ("stereo_pair_5378_5380", "cluster_176_row_5378", "stereo_A", "same-connectivity stereo comparison"),
        ("stereo_pair_5378_5380", "cluster_176_row_5380", "stereo_B", "same-connectivity stereo comparison"),
        ("stereo_pair_5356_5359", "cluster_246_row_5356", "stereo_A", "same-connectivity stereo comparison"),
        ("stereo_pair_5356_5359", "cluster_246_row_5359", "stereo_B", "same-connectivity stereo comparison"),
        ("cluster3_5354_vs_2412", "cluster_3_row_5354", "more_favorable_score", "within-cluster score contrast"),
        ("cluster3_5354_vs_2412", "cluster_3_row_2412", "less_favorable_score", "within-cluster score contrast"),
        ("cluster246_representatives", "cluster_246_row_5372", "more_favorable_score", "Cluster 246 score extreme"),
        ("cluster246_representatives", "cluster_246_row_704", "less_favorable_score", "Cluster 246 score extreme"),
        ("androgens_reference_cluster3", "cluster_3_row_5354", "androgens_annotation", "original Androgens annotation"),
        ("androgens_reference_cluster176", "cluster_176_row_5378", "androgens_annotation", "original Androgens annotation"),
        ("androgens_reference_cluster246", "cluster_246_row_5330", "androgens_annotation", "original Androgens annotation"),
        ("androgens_reference_cluster416", "cluster_416_row_5373", "androgens_annotation", "original Androgens annotation"),
    ]
    representative_rows = []
    for case_name, compound_id, role, reason in cases:
        row = pose_df[pose_df.compound_id == compound_id].iloc[0]
        representative_rows.append({"case_name": case_name, "compound_id": compound_id, "cluster_id": int(row.cluster_id), "row_index": int(row.row_index), "role": role, "reason": reason, "vina_score_kcal_mol": float(row.vina_score_kcal_mol), "pose_family": int(family_map[compound_id]), "hbond_candidate_count": int(row.hbond_candidate_count), "polar_contact_count": int(row.polar_contact_count), "hydrophobic_contact_count": int(row.hydrophobic_contact_count), "steric_clash_count": int(row.steric_clash_count)})
    pd.DataFrame(representative_rows).to_csv(OUT / "representative_cases.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "scope": {"compound_count": 24, "top_pose_only": True, "receptor": str(RECEPTOR_PDB.relative_to(ROOT)), "receptor_hydrogen_source": str(RECEPTOR_H_PDB.relative_to(ROOT)), "native_reference": str(TES_SDF.relative_to(ROOT)), "docking_rerun": False},
        "methods": {"interaction_method": "custom geometry rules using explicit 3D coordinates", "hydrogen_bond_candidate": "donor-acceptor distance <=3.6 A, H-acceptor <=2.7 A, donor-H-acceptor angle >=110 degrees", "polar_contact": "polar heavy-atom distance <=4.0 A; not called an H-bond", "hydrophobic_contact": "C/S ligand to carbon of hydrophobic residue <=4.2 A", "close_vdw_contact": "distance <= sum of element vdW radii +0.35 A", "pi_contact_candidate": "aromatic ring centroid distance <=5.5 A; orientation not asserted", "halogen_contact_candidate": "X-acceptor <=3.8 A and C-X-acceptor angle >=135 degrees", "steric_clash": "heavy-atom distance <2.0 A", "common_core": "MCS across 24 top poses plus TES, complete rings and element/bond matching, no forced superposition", "pose_family": family_meta},
        "mcs": {"smarts": mcs_smarts, "query_atom_count": int(common_query.GetNumAtoms()) if common_query is not None else 0},
        "receptor_atom_count_heavy": len(heavy_receptor),
        "receptor_residue_count": len({a["residue_key"] for a in heavy_receptor}),
        "interaction_type_counts": {t: int(global_df[t].sum()) for t in INTERACTION_TYPES},
        "key_residues": sorted(KEY_RESIDUES),
        "family_assignments": family_meta,
        "software": {"rdkit": rdBase.rdkitVersion, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (OUT / "pose_analysis_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    print(json.dumps({"status": "success", "compounds": 24, "contacts": len(contacts_df), "mcs_atoms": metadata["mcs"]["query_atom_count"], "families": family_meta["selected_family_count"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
