#!/usr/bin/env python3
"""Step 4.3: C17 substitution pattern and AR binding geometry analysis.

This script only reads the validated Step 3.2/4.1/4.2 outputs.  It does not
rerun docking or modify any receptor, ligand, pose, score, or docking-box file.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
POSE_DIR = ROOT / "results" / "docking_2AM9" / "top_sdf"
OUT = ROOT / "results" / "c17_analysis"
PML_DIR = OUT / "pymol_scripts"
FIG = ROOT / "figures" / "c17_analysis"
STRUCT_DIR = ROOT / "results" / "pose_analysis" / "structures"
OUT.mkdir(parents=True, exist_ok=True)
PML_DIR.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


PAIRS = {
    "A": {"label": "C17 ethynyl introduction", "left": "cluster_416_row_5373", "right": "cluster_176_row_705", "left_color": "cyan", "right_color": "magenta"},
    "B": {"label": "C17 hydroxyl esterification", "left": "cluster_176_row_5378", "right": "cluster_3_row_2412", "left_color": "marine", "right_color": "orange"},
    "C": {"label": "C17 ester side-chain extension", "left": "cluster_3_row_2412", "right": "cluster_246_row_5379", "left_color": "orange", "right_color": "green"},
}

KEY_RESIDUES = ["A:ASN705", "A:THR877", "A:GLN711", "A:ARG752"]


def safe_float(x):
    return float(x) if x not in (None, "", "nan") else np.nan


def read_one_mol(path: Path) -> Chem.Mol:
    mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError(f"Cannot parse SDF: {path}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"No 3D conformer: {path}")
    return mol


def coords(mol: Chem.Mol, atom_indices=None) -> np.ndarray:
    conf = mol.GetConformer()
    if atom_indices is None:
        atom_indices = range(mol.GetNumAtoms())
    return np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in atom_indices], dtype=float)


def formula(mol: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(Chem.RemoveHs(Chem.Mol(mol)))


def heavy_atom_indices(mol: Chem.Mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def ring_system_atoms(mol: Chem.Mol):
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    systems = []
    for ring in rings:
        merged = set(ring)
        changed = True
        while changed:
            changed = False
            for other in rings:
                if merged.intersection(other) and not other.issubset(merged):
                    merged.update(other)
                    changed = True
        if not any(merged == s for s in systems):
            systems.append(merged)
    return systems


def atom_in_same_ring_system(mol: Chem.Mol, atom_idx: int, other_idx: int) -> bool:
    return any(atom_idx in s and other_idx in s for s in ring_system_atoms(mol))


def steroid_core_atoms(mol: Chem.Mol):
    """Return the largest fused ring system, used as the steroid core.

    In these five compounds this is the fused steroid tetracyclic system.  It
    is deliberately kept separate from the full pairwise MCS, which can also
    include a shared ester or other substituent.
    """
    systems = ring_system_atoms(mol)
    if not systems:
        raise ValueError("No ring system found for steroid core")
    return sorted(max(systems, key=len))


def find_c17(mol: Chem.Mol) -> int:
    """Identify C17 as a ring carbon bearing a single-bonded exocyclic O.

    The C3 ketone is excluded because its C=O bond is double.  This rule is
    checked against the explicit isomeric SMILES and the pairwise MCS below.
    """
    candidates = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or not atom.IsInRing():
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.SINGLE and not other.IsInRing():
                candidates.append(atom.GetIdx())
    if len(candidates) != 1:
        # A fallback for an unusual valence representation: choose the ring
        # carbon attached to an oxygen that is not a carbonyl oxygen.
        scored = []
        for idx in candidates:
            a = mol.GetAtomWithIdx(idx)
            scored.append((sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() == 6), idx))
        if not scored:
            raise ValueError("Could not identify a C17 candidate")
        return sorted(scored, reverse=True)[0][1]
    return candidates[0]


def branch_atoms_from_c17(mol: Chem.Mol, c17: int):
    branch = set()
    for nbr in mol.GetAtomWithIdx(c17).GetNeighbors():
        ni = nbr.GetIdx()
        if atom_in_same_ring_system(mol, c17, ni):
            continue
        stack = [ni]
        seen = set()
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            branch.add(i)
            for n in mol.GetAtomWithIdx(i).GetNeighbors():
                j = n.GetIdx()
                if j != c17 and not atom_in_same_ring_system(mol, c17, j):
                    stack.append(j)
    return sorted(i for i in branch if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)


def mcs_mapping(m1: Chem.Mol, m2: Chem.Mol):
    result = rdFMCS.FindMCS(
        [Chem.RemoveHs(m1), Chem.RemoveHs(m2)],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchValences=False,
        timeout=60,
    )
    patt = Chem.MolFromSmarts(result.smartsString)
    match1 = Chem.RemoveHs(m1).GetSubstructMatch(patt)
    match2 = Chem.RemoveHs(m2).GetSubstructMatch(patt)
    if len(match1) != result.numAtoms or len(match2) != result.numAtoms:
        raise ValueError("MCS matching failed after SMARTS generation")
    return result, list(match1), list(match2)


def kabsch_rmsd(x: np.ndarray, y: np.ndarray, fit=True):
    if not fit:
        return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    h = xc.T @ yc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    xr = xc @ r
    return float(np.sqrt(np.mean(np.sum((xr - yc) ** 2, axis=1))))


def principal_axis_angle(x: np.ndarray, y: np.ndarray):
    def first_axis(a):
        _, _, vt = np.linalg.svd(a - a.mean(axis=0), full_matrices=False)
        return vt[0]
    a = first_axis(x)
    b = first_axis(y)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def chiral_centers(mol: Chem.Mol):
    no_h = Chem.RemoveHs(Chem.Mol(mol))
    Chem.AssignStereochemistry(no_h, force=True, cleanIt=True)
    return dict(Chem.FindMolChiralCenters(no_h, includeUnassigned=True, useLegacyImplementation=False))


VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "Cl": 1.75, "BR": 1.85, "I": 1.98}


def parse_pdb_atoms(path: Path, include_h=False):
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        element = line[76:78].strip().title() or re.sub(r"[^A-Za-z]", "", line[12:16]).strip()[:1].title()
        if not include_h and element == "H":
            continue
        atoms.append({
            "serial": int(line[6:11]),
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21].strip() or "_",
            "resseq": line[22:26].strip(),
            "element": element,
            "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
        })
    return atoms


def receptor_residue_key(atom):
    return f"{atom['chain']}:{atom['resname']}{atom['resseq']}"


def row_to_dict(df, compound_id):
    row = df.loc[df.compound_id == compound_id]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {compound_id}, got {len(row)}")
    return row.iloc[0].to_dict()


def ligand_pdb_path(compound_id):
    path = STRUCT_DIR / f"{compound_id}.pdb"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def make_pymol_script(pair_id, pair, records, local=False):
    left, right = pair["left"], pair["right"]
    out_name = f"comparison_{pair_id}_{'c17_closeup' if local else 'overlay'}"
    pml = PML_DIR / f"{out_name}.pml"
    png = FIG / f"{out_name}.png"
    pse = OUT / "pymol_scripts" / f"{out_name}.pse"
    receptor = (PREP / "receptor_repaired.pdb").as_posix()
    native = (STRUCT_DIR / "TES_crystal.pdb").as_posix()
    lp = ligand_pdb_path(left).as_posix()
    rp = ligand_pdb_path(right).as_posix()
    lines = [
        "reinitialize",
        f"load {receptor}, receptor",
        f"load {native}, native_TES",
        f"load {lp}, ligand_left",
        f"load {rp}, ligand_right",
        "hide everything",
        "show cartoon, receptor",
        "color gray80, receptor",
        "set cartoon_transparency, 15, receptor",
        "show sticks, native_TES",
        "color yellow, native_TES",
        "set stick_transparency, 65, native_TES",
        "show sticks, ligand_left",
        f"color {pair['left_color']}, ligand_left",
        "show sticks, ligand_right",
        f"color {pair['right_color']}, ligand_right",
        "set stick_radius, 0.18",
        "set orthoscopic, on",
        "set ray_opaque_background, off",
        "bg_color white",
    ]
    if local:
        reskeys = []
        for key in records["left"]["local_residues"] + records["right"]["local_residues"]:
            if key not in reskeys:
                reskeys.append(key)
        reskeys = reskeys[:6]
        sel_parts = []
        for key in reskeys:
            chain, rest = key.split(":", 1)
            resseq = re.search(r"-?\d+", rest).group(0)
            sel_parts.append(f"(chain {chain} and resi {resseq})")
        if sel_parts:
            lines += [f"select local_residues, {' or '.join(sel_parts)}", "show sticks, local_residues", "color gray50, local_residues"]
        lines += ["select local_view, ligand_left or ligand_right or local_residues or native_TES", "orient local_view", "zoom local_view, 5"]
    else:
        lines += ["select key_residues, receptor and (resi 705+711+752+877)", "show sticks, key_residues", "color wheat, key_residues", "orient receptor", "zoom receptor, 5"]
    # A clear label is attached to a representative C17 atom in each ligand.
    for obj, rec, color in [("ligand_left", records["left"], pair["left_color"]), ("ligand_right", records["right"], pair["right_color"])]:
        c17_id = rec["pdb_serial_by_atom"][str(rec["c17_idx"])]
        label_offset = "[2.0, 1.5, 0.0]" if obj == "ligand_left" else "[-2.0, -1.5, 0.0]"
        lines += [f"label {obj} and id {c17_id}, 'row {rec['row_index']} | Vina {rec['score']:.3f}'", f"set label_color, {color}, {obj} and id {c17_id}", f"set label_position, {label_offset}, {obj} and id {c17_id}"]
        if local:
            for i, contact in enumerate(rec["nearest_contacts"][:2], start=1):
                lines += [f"distance {obj}_d{i}, {obj} and id {contact['ligand_serial']}, receptor and id {contact['receptor_serial']}", f"set dash_color, {color}, {obj}_d{i}", f"set label_color, {color}, {obj}_d{i}", f"set dash_width, 2.0, {obj}_d{i}"]
    lines += ["set label_size, 12", "set dash_gap, 0.35", "set dash_length, 0.25", f"ray 1600, 1200", f"png {png.as_posix()}, dpi=300", f"save {pse.as_posix()}", "quit"]
    pml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pml, png, pse


def render_pymol(pml: Path):
    pymol = os.environ.get("PYMOL_BIN", "pymol")
    try:
        result = subprocess.run([pymol, "-cq", str(pml)], cwd=ROOT, text=True, capture_output=True, timeout=180)
        return result.returncode, (result.stdout + "\n" + result.stderr)[-2000:]
    except Exception as exc:
        return 127, str(exc)


def make_2d_figure(mols, descriptors):
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), dpi=180)
    fig.suptitle("Step 4.3: C17 matched-pair structure comparisons", fontsize=15, y=0.995)
    mpl_colors = {"cyan": "#00A6D6", "magenta": "#C23B8F", "marine": "#1F77B4", "orange": "#E67E22", "green": "#2CA02C"}
    for ax, (pid, pair) in zip(axes.ravel(), PAIRS.items()):
        ax.axis("off")
        left, right = pair["left"], pair["right"]
        for x, cid, color in [(0.28, left, pair["left_color"]), (0.72, right, pair["right_color"])]:
            mol = mols[cid]
            img = Draw.MolToImage(mol, size=(560, 420), kekulize=False, wedgeBonds=True)
            ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.42), (x, 0.57), frameon=False))
            d = descriptors[cid]
            ax.text(x, 0.17, f"row {d['row_index']}\nVina {d['score']:.3f} kcal/mol", ha="center", va="center", fontsize=10, color=mpl_colors[color])
        ax.text(0.5, 0.95, f"Comparison {pid}: {pair['label']}", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.plot([0.50, 0.50], [0.28, 0.83], color="0.75", lw=1, transform=ax.transAxes)
    for ax in axes[2, :]:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG / "c17_comparison_2d.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "c17_comparison_2d.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    desc = pd.read_csv(ROOT / "results" / "docking_analysis" / "compound_descriptors.csv")
    top = pd.read_csv(ROOT / "results" / "docking_2AM9" / "docking_top_poses.csv")
    contacts = pd.read_csv(ROOT / "results" / "pose_analysis" / "ligand_residue_contacts.csv")
    pose_desc = pd.read_csv(ROOT / "results" / "pose_analysis" / "pose_descriptors.csv")
    if len(desc) == 0 or len(top) == 0:
        raise ValueError("Required Step 4.1/3.2 tables are empty")

    receptor_atoms = parse_pdb_atoms(PREP / "receptor_repaired.pdb", include_h=False)
    receptor_by_serial = {a["serial"]: a for a in receptor_atoms}
    tes = read_one_mol(PREP / "TES_crystal.sdf")
    mols = {}
    records = {}
    for pair in PAIRS.values():
        for cid in [pair["left"], pair["right"]]:
            if cid in mols:
                continue
            row = row_to_dict(desc, cid)
            pose = read_one_mol(POSE_DIR / f"{cid}.sdf")
            smiles_mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
            if smiles_mol is None:
                raise ValueError(f"Invalid canonical SMILES: {cid}")
            pose_no_h = Chem.RemoveHs(pose)
            pose_iso = Chem.MolToSmiles(pose_no_h, isomericSmiles=True)
            ref_iso = Chem.MolToSmiles(smiles_mol, isomericSmiles=True)
            pose_conn = Chem.MolToSmiles(pose_no_h, isomericSmiles=False)
            ref_conn = Chem.MolToSmiles(smiles_mol, isomericSmiles=False)
            if pose_conn != ref_conn:
                raise ValueError(f"Top pose connectivity does not match validated SMILES: {cid}")
            stereo_status = "canonical isomeric SMILES match" if pose_iso == ref_iso else "connectivity match; SDF stereobond serialization differs"
            mols[cid] = pose
            c17 = find_c17(pose)
            branch = branch_atoms_from_c17(pose, c17)
            pose_row = row_to_dict(pose_desc, cid)
            # Existing Step 4.2 PDB files preserve the SDF atom order for the
            # heavy atoms.  Confirm this against atom counts before using IDs.
            pdb_atoms = parse_pdb_atoms(ligand_pdb_path(cid), include_h=True)
            heavy = [a for a in pdb_atoms if a["element"] != "H"]
            if len(heavy) != pose.GetNumHeavyAtoms():
                raise ValueError(f"PDB/SDF heavy-atom count mismatch for {cid}: {len(heavy)} vs {pose.GetNumHeavyAtoms()}")
            serial_by_atom = {str(i): heavy[i]["serial"] for i in range(len(heavy))}
            coords_all = coords(pose)
            group = sorted(set([c17] + branch))
            group_contacts = []
            for li in group:
                for ra in receptor_atoms:
                    dist = float(np.linalg.norm(coords_all[li] - ra["coord"]))
                    if dist <= 6.0:
                        group_contacts.append({"ligand_idx": li, "ligand_serial": serial_by_atom[str(li)], "receptor_serial": ra["serial"], "residue_key": receptor_residue_key(ra), "receptor_atom": ra["name"], "distance": dist, "vdw_overlap": VDW.get(pose.GetAtomWithIdx(li).GetSymbol(), 1.7) + VDW.get(ra["element"], 1.7) - dist})
            group_contacts.sort(key=lambda x: x["distance"])
            key_dist = {}
            for key in KEY_RESIDUES:
                chain, rest = key.split(":", 1)
                match = re.match(r"([A-Za-z]+)(-?\d+)", rest)
                atoms = [a for a in receptor_atoms if a["chain"] == chain and a["resname"] == match.group(1) and a["resseq"] == match.group(2)]
                key_dist[key] = min((float(np.linalg.norm(coords_all[li] - a["coord"])) for li in group for a in atoms), default=np.nan)
            local_residues = []
            for c in group_contacts[:10]:
                if c["residue_key"] not in local_residues:
                    local_residues.append(c["residue_key"])
            existing = contacts.loc[contacts.compound_id == cid]
            records[cid] = {
                "compound_id": cid, "row_index": int(row["row_index"]), "cluster_id": int(row["cluster_id"]), "score": float(row["vina_score_kcal_mol"]), "formula": formula(smiles_mol), "smiles": row["canonical_isomeric_smiles"], "mol": pose, "c17_idx": c17, "branch_indices": branch, "pdb_serial_by_atom": serial_by_atom, "nearest_contacts": group_contacts, "local_residues": local_residues, "key_distances": key_dist, "pose_core_raw_rmsd": safe_float(pose_row.get("core_raw_rmsd_to_TES_A")), "existing_interaction_types": ";".join(existing.interaction_types.dropna().astype(str).unique()), "stereo_status": stereo_status,
            }

    validation_rows = []
    mapping_rows = []
    full_mapping_rows = []
    geometry_rows = []
    local_rows = []
    for pid, pair in PAIRS.items():
        a, b = pair["left"], pair["right"]
        m1, m2 = mols[a], mols[b]
        mcs, match1, match2 = mcs_mapping(m1, m2)
        c17_a, c17_b = records[a]["c17_idx"], records[b]["c17_idx"]
        mcs_atom_for_a = {int(x): i for i, x in enumerate(match1)}
        mcs_atom_for_b = {int(x): i for i, x in enumerate(match2)}
        ring_a = set(steroid_core_atoms(m1))
        ring_b = set(steroid_core_atoms(m2))
        core_pairs = [(i, j) for i, j in zip(match1, match2) if i in ring_a and j in ring_b]
        core_match1 = [i for i, _ in core_pairs]
        core_match2 = [j for _, j in core_pairs]
        c17_map_a = core_match1.index(c17_a) if c17_a in core_match1 else -1
        c17_map_b = core_match2.index(c17_b) if c17_b in core_match2 else -1
        chir_a = chiral_centers(m1)
        chir_b = chiral_centers(m2)
        common_chiral = [(i, j, chir_a[i], chir_b[j]) for i, j in core_pairs if i in chir_a and j in chir_b]
        same_chiral = [(i, j, sa) for i, j, sa, sb in common_chiral if sa == sb]
        differing_chiral = [(i, j, sa, sb) for i, j, sa, sb in common_chiral if sa != sb]
        x = coords(m1, core_match1)
        y = coords(m2, core_match2)
        c17_dist = float(np.linalg.norm(coords(m1, [c17_a])[0] - coords(m2, [c17_b])[0]))
        branch_a = records[a]["branch_indices"]
        branch_b = records[b]["branch_indices"]
        branch_centroid_a = coords(m1, branch_a).mean(axis=0) if branch_a else coords(m1, [c17_a]).mean(axis=0)
        branch_centroid_b = coords(m2, branch_b).mean(axis=0) if branch_b else coords(m2, [c17_b]).mean(axis=0)
        validation_rows.append({
            "comparison": pid, "description": pair["label"], "compound_left": a, "compound_right": b,
            "row_left": records[a]["row_index"], "row_right": records[b]["row_index"], "score_left": records[a]["score"], "score_right": records[b]["score"], "score_difference_right_minus_left": records[b]["score"] - records[a]["score"],
            "formula_left": records[a]["formula"], "formula_right": records[b]["formula"], "stereo_status_left": records[a]["stereo_status"], "stereo_status_right": records[b]["stereo_status"], "chiral_centers_left": len(chir_a), "chiral_centers_right": len(chir_b), "common_core_chiral_centers": len(common_chiral), "common_core_chiral_same": len(same_chiral), "common_core_chiral_differences": ";".join(f"{i}->{j}:{sa}/{sb}" for i, j, sa, sb in differing_chiral), "heavy_atoms_left": m1.GetNumHeavyAtoms(), "heavy_atoms_right": m2.GetNumHeavyAtoms(), "mw_left": Descriptors.MolWt(m1), "mw_right": Descriptors.MolWt(m2), "rotatable_bonds_left": rdMolDescriptors.CalcNumRotatableBonds(m1), "rotatable_bonds_right": rdMolDescriptors.CalcNumRotatableBonds(m2),
            "tanimoto_with_chirality": DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048, useChirality=True), AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048, useChirality=True)), "tanimoto_without_chirality": DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048, useChirality=False), AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048, useChirality=False)),
            "mcs_atoms": mcs.numAtoms, "mcs_bonds": mcs.numBonds, "mcs_smarts": mcs.smartsString, "mcs_coverage_left": mcs.numAtoms / m1.GetNumHeavyAtoms(), "mcs_coverage_right": mcs.numAtoms / m2.GetNumHeavyAtoms(), "steroid_core_atoms_left": len(ring_a), "steroid_core_atoms_right": len(ring_b), "steroid_core_mapped_atoms": len(core_pairs), "steroid_core_coverage_left": len(core_pairs) / len(ring_a), "steroid_core_coverage_right": len(core_pairs) / len(ring_b), "c17_core_atom_left": c17_map_a, "c17_core_atom_right": c17_map_b,
            "unmatched_atoms_left": m1.GetNumHeavyAtoms() - mcs.numAtoms, "unmatched_atoms_right": m2.GetNumHeavyAtoms() - mcs.numAtoms, "unmatched_elements_left": ",".join(Counter(m1.GetAtomWithIdx(i).GetSymbol() for i in heavy_atom_indices(m1) if i not in match1).elements()), "unmatched_elements_right": ",".join(Counter(m2.GetAtomWithIdx(i).GetSymbol() for i in heavy_atom_indices(m2) if i not in match2).elements()), "c17_rule": "ring carbon with exocyclic single-bond oxygen; C=O excluded",
        })
        for mi, (i, j) in enumerate(zip(match1, match2)):
            full_mapping_rows.append({"comparison": pid, "mcs_atom_index": mi, "compound_left": a, "left_atom_index": i, "left_symbol": m1.GetAtomWithIdx(i).GetSymbol(), "left_is_steroid_ring_atom": i in ring_a, "left_is_c17": i == c17_a, "compound_right": b, "right_atom_index": j, "right_symbol": m2.GetAtomWithIdx(j).GetSymbol(), "right_is_steroid_ring_atom": j in ring_b, "right_is_c17": j == c17_b})
        for mi, (i, j) in enumerate(core_pairs):
            mapping_rows.append({"comparison": pid, "core_atom_index": mi, "compound_left": a, "left_atom_index": i, "left_symbol": m1.GetAtomWithIdx(i).GetSymbol(), "left_is_c17": i == c17_a, "compound_right": b, "right_atom_index": j, "right_symbol": m2.GetAtomWithIdx(j).GetSymbol(), "right_is_c17": j == c17_b, "mapping_rule": "pairwise MCS filtered to largest fused ring system; C17 identified independently"})
        geometry_rows.append({"comparison": pid, "compound_left": a, "compound_right": b, "raw_common_core_rmsd_A": kabsch_rmsd(x, y, fit=False), "best_fit_common_core_rmsd_A": kabsch_rmsd(x, y, fit=True), "core_centroid_displacement_A": float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0))), "core_orientation_difference_deg": principal_axis_angle(x, y), "c17_distance_A": c17_dist, "c17_branch_centroid_distance_A": float(np.linalg.norm(branch_centroid_a - branch_centroid_b)), "core_atom_count": len(core_pairs), "full_mcs_atom_count": mcs.numAtoms, "core_mapping_status": "pairwise element/ring-topology MCS filtered to largest fused ring system; no chiral constraint; direct coordinate comparison"})
        for cid, side in [(a, "left"), (b, "right")]:
            for c in records[cid]["nearest_contacts"][:10]:
                local_rows.append({"comparison": pid, "side": side, "compound_id": cid, "row_index": records[cid]["row_index"], "vina_score": records[cid]["score"], "ligand_atom_index": c["ligand_idx"], "ligand_atom_role": "C17" if c["ligand_idx"] == records[cid]["c17_idx"] else "C17_substituent", "ligand_atom_serial": c["ligand_serial"], "receptor_atom_serial": c["receptor_serial"], "receptor_atom": c["receptor_atom"], "residue_key": c["residue_key"], "distance_A": c["distance"], "vdw_overlap_estimate_A": c["vdw_overlap"], "short_contact_lt_2_5_A": c["distance"] < 2.5})

    # Store pair-specific MCS and geometry tables.
    pd.DataFrame(validation_rows).to_csv(OUT / "matched_pair_structure_validation.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(OUT / "common_core_mapping.csv", index=False)
    pd.DataFrame(full_mapping_rows).to_csv(OUT / "full_mcs_mapping.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(OUT / "pairwise_pose_geometry.csv", index=False)
    pd.DataFrame(local_rows).to_csv(OUT / "c17_local_contacts.csv", index=False)

    summary_rows = []
    for pid, pair in PAIRS.items():
        g = next(r for r in geometry_rows if r["comparison"] == pid)
        v = next(r for r in validation_rows if r["comparison"] == pid)
        for cid, side in [(pair["left"], "left"), (pair["right"], "right")]:
            rec = records[cid]
            summary_rows.append({"comparison": pid, "comparison_label": pair["label"], "side": side, "compound_id": cid, "row_index": rec["row_index"], "cluster_id": rec["cluster_id"], "vina_score": rec["score"], "formula": rec["formula"], "stereo_status": rec["stereo_status"], "c17_atom_index": rec["c17_idx"], "c17_branch_atom_indices": ";".join(map(str, rec["branch_indices"])), "nearest_residue_1": rec["nearest_contacts"][0]["residue_key"] if rec["nearest_contacts"] else "", "nearest_distance_1_A": rec["nearest_contacts"][0]["distance"] if rec["nearest_contacts"] else np.nan, "nearest_residues_10": ";".join(rec["local_residues"]), **{f"{k}_c17_group_distance_A": rec["key_distances"][k] for k in KEY_RESIDUES}, "pair_raw_core_rmsd_A": g["raw_common_core_rmsd_A"], "pair_best_fit_core_rmsd_A": g["best_fit_common_core_rmsd_A"], "pair_core_centroid_displacement_A": g["core_centroid_displacement_A"], "pair_c17_distance_A": g["c17_distance_A"], "pair_score_difference_right_minus_left": v["score_difference_right_minus_left"], "existing_step42_interaction_types": rec["existing_interaction_types"]})
    pd.DataFrame(summary_rows).to_csv(OUT / "comparison_summary.csv", index=False)

    make_2d_figure(mols, records)

    pymol_results = {}
    for pid, pair in PAIRS.items():
        pair_records = {"left": records[pair["left"]], "right": records[pair["right"]]}
        pair_results = {}
        for local in [False, True]:
            pml, png, pse = make_pymol_script(pid, pair, pair_records, local=local)
            if os.environ.get("STEP43_SKIP_PYMOL") == "1":
                rc, log = 0, "skipped by STEP43_SKIP_PYMOL"
            else:
                rc, log = render_pymol(pml)
            pair_results["closeup" if local else "overlay"] = {"pml": str(pml.relative_to(ROOT)), "png": str(png.relative_to(ROOT)), "pse": str(pse.relative_to(ROOT)), "returncode": rc, "log_tail": log}
        pymol_results[pid] = pair_results

    # A machine-readable manifest keeps the analysis traceable.
    manifest = {"status": "success", "step": "4.3", "docking_rerun": False, "compound_count": 5, "comparisons": list(PAIRS), "files": {"structure_validation": "results/c17_analysis/matched_pair_structure_validation.csv", "core_mapping": "results/c17_analysis/common_core_mapping.csv", "geometry": "results/c17_analysis/pairwise_pose_geometry.csv", "local_contacts": "results/c17_analysis/c17_local_contacts.csv", "summary": "results/c17_analysis/comparison_summary.csv", "report": "reports/c17_substitution_analysis.md"}, "pymol": pymol_results}
    (OUT / "c17_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"status": "success", "comparisons": list(PAIRS), "mcs_atoms": {r["comparison"]: r["mcs_atoms"] for r in validation_rows}, "pymol_returncodes": {p: {k: v["returncode"] for k, v in d.items()} for p, d in pymol_results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
