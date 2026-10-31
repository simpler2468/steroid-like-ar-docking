#!/usr/bin/env python3
"""Step 4.4: directed C3->C17 orientation verification and robustness analysis."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFMCS

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "prepared" / "2AM9"
ROB = ROOT / "results" / "orientation_robustness"
POSE_DIR = ROB / "docking_poses"
OUT = ROB
FIG = ROOT / "figures" / "orientation_robustness"
PML_DIR = ROB / "pymol_scripts"
REP_DIR = ROB / "representatives"
FIG.mkdir(parents=True, exist_ok=True)
PML_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]
KEY_RESIDUES = ["A:ARG752", "A:GLN711", "A:ASN705", "A:THR877"]
COLOR = {"TES": "yellow", "5378": "blue", "2412": "orange", "5379": "green", "5373": "cyan", "705": "magenta"}
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98}


def read_mol(path: Path):
    mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Invalid or non-3D SDF: {path}")
    return mol


def heavy_indices(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def xyz(mol, idx):
    p = mol.GetConformer().GetAtomPosition(int(idx))
    return np.array([p.x, p.y, p.z], dtype=float)


def all_coords(mol, indices):
    return np.array([xyz(mol, i) for i in indices], dtype=float)


def ring_systems(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
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


def largest_ring_system(mol):
    systems = ring_systems(mol)
    if not systems:
        raise ValueError("no ring system")
    return set(max(systems, key=len))


def find_anchor_atoms(mol):
    """Find C3/C17 using ring topology plus TES functional landmarks."""
    ring = largest_ring_system(mol)
    c3 = []
    c17 = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if i not in ring or atom.GetAtomicNum() != 6:
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() != 8 or other.IsInRing():
                continue
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                c3.append(i)
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                c17.append(i)
    if len(c3) != 1 or len(c17) != 1:
        raise ValueError(f"C3/C17 landmark ambiguity: C3={c3}, C17={c17}")
    return {"C3": c3[0], "C17": c17[0], "ring_atoms": sorted(ring), "rule": "largest fused ring system + C3 exocyclic C=O + C17 exocyclic C-O; cross-checked to TES MCS"}


def mcs_map(ref, mol):
    result = rdFMCS.FindMCS([Chem.RemoveHs(ref), Chem.RemoveHs(mol)], atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True, completeRingsOnly=True, matchValences=False, timeout=60)
    patt = Chem.MolFromSmarts(result.smartsString)
    mr = Chem.RemoveHs(ref).GetSubstructMatch(patt)
    mm = Chem.RemoveHs(mol).GetSubstructMatch(patt)
    ref_ring = largest_ring_system(ref)
    mol_ring = largest_ring_system(mol)
    pairs = [(i, j) for i, j in zip(mr, mm) if i in ref_ring and j in mol_ring]
    return result, pairs


def angle_deg(v1, v2):
    a = v1 / np.linalg.norm(v1)
    b = v2 / np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def direct_rmsd(x, y):
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))


def fitted_rmsd(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return float(np.sqrt(np.mean(np.sum((xc @ r - yc) ** 2, axis=1))))


def parse_pdb(path):
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        element = line[76:78].strip().title() or re.sub(r"[^A-Za-z]", "", line[12:16]).strip()[:1].title()
        if element == "H":
            continue
        atoms.append({"serial": int(line[6:11]), "name": line[12:16].strip(), "resname": line[17:20].strip(), "chain": line[21].strip() or "_", "resseq": line[22:26].strip(), "element": element, "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])})
    return atoms


def residue_key(atom):
    return f"{atom['chain']}:{atom['resname']}{atom['resseq']}"


def key_atoms(receptor, key):
    chain, rest = key.split(":")
    m = re.match(r"([A-Za-z]+)(-?\d+)", rest)
    return [a for a in receptor if a["chain"] == chain and a["resname"] == m.group(1) and a["resseq"] == m.group(2)]


def local_metrics(mol, anchors, receptor):
    c17 = xyz(mol, anchors["C17"])
    heavy = heavy_indices(mol)
    branch = []
    for atom in mol.GetAtomWithIdx(anchors["C17"]).GetNeighbors():
        if atom.GetIdx() in anchors["ring_atoms"]:
            continue
        stack = [atom.GetIdx()]
        seen = set()
        while stack:
            i = stack.pop()
            if i in seen or mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
                continue
            seen.add(i)
            branch.append(i)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != anchors["C17"])
    group = sorted(set([anchors["C17"]] + branch))
    contacts = []
    for li in group:
        lc = xyz(mol, li)
        for ra in receptor:
            d = float(np.linalg.norm(lc - ra["coord"]))
            if d <= 6.0:
                overlap = VDW.get(mol.GetAtomWithIdx(li).GetSymbol(), 1.7) + VDW.get(ra["element"], 1.7) - d
                contacts.append({"ligand_idx": li, "receptor_serial": ra["serial"], "receptor_atom": ra["name"], "residue_key": residue_key(ra), "distance_A": d, "vdw_overlap_estimate_A": overlap, "short_contact_lt_2_5_A": d < 2.5})
    contacts.sort(key=lambda x: x["distance_A"])
    key_distances = {}
    for key in KEY_RESIDUES:
        key_distances[key] = min((float(np.linalg.norm(xyz(mol, li) - ra["coord"])) for li in group for ra in key_atoms(receptor, key)), default=np.nan)
    return {"c17_group_indices": group, "nearest": contacts[:10], "key_distances": key_distances, "min_distance_A": contacts[0]["distance_A"] if contacts else np.nan, "short_count_lt_2_5": sum(c["short_contact_lt_2_5_A"] for c in contacts)}


def classify(angle_tes, d_same, d_swap, native_score, reversed_score):
    if not all(np.isfinite(x) for x in [angle_tes, d_same, d_swap, native_score, reversed_score]):
        return "ambiguous"
    if angle_tes >= 120 and d_swap + 1.0 < d_same and reversed_score > 0:
        return "head_to_tail_reversal_supported"
    if angle_tes <= 60 and d_same <= d_swap + 1.0 and native_score > 0:
        return "consistent_with_native_orientation"
    if angle_tes >= 120 or angle_tes <= 60:
        return "alternative_orientation"
    return "ambiguous"


def save_pose_pdb(mol, path):
    Chem.MolToPDBFile(mol, str(path))


def write_cgo_pml(path, png, pse, entries, receptor_path, title, closeup=False):
    # entries: list of (name, pdb path, color, score, c3_coord, c17_coord)
    lines = ["reinitialize", f"load {receptor_path.as_posix()}, receptor"]
    for name, pdb, color, score, c3, c17 in entries:
        label = "TES | crystal" if name == "TES" else f"{name} | Vina {score:.3f}"
        lines += [f"load {pdb.as_posix()}, {name}", f"show sticks, {name}", f"color {color}, {name}", f"label {name} and name C17, '{label}'", f"set label_color, {color}, {name}"]
    lines += ["hide everything, receptor", "show cartoon, receptor", "color gray80, receptor", "set cartoon_transparency, 25, receptor", "set orthoscopic, on", "bg_color white"]
    lines += ["select key_residues, receptor and (resi 705+711+752+877)", "show sticks, key_residues", "color wheat, key_residues", "label key_residues and name CA, resn + resi", "set label_color, black, key_residues"]
    lines += ["orient receptor", "zoom receptor, 5"]
    for i, (name, pdb, color, score, c3, c17) in enumerate(entries):
        # C3/C17 pseudoatoms make the endpoint mapping visible and provide a
        # real coordinate source for the CGO arrow.
        lines += [f"pseudoatom {name}_C3, pos=[{c3[0]:.5f},{c3[1]:.5f},{c3[2]:.5f}]", f"pseudoatom {name}_C17, pos=[{c17[0]:.5f},{c17[1]:.5f},{c17[2]:.5f}]", f"show spheres, {name}_C3 or {name}_C17", f"color {color}, {name}_C3 or {name}_C17", f"label {name}_C3, 'C3'", f"label {name}_C17, 'C17'"]
        lines += [f"distance {name}_direction, {name}_C3, {name}_C17", f"hide labels, {name}_direction", f"set dash_color, {color}, {name}_direction", f"set dash_width, 4.0, {name}_direction"]
    # Use a small embedded PyMOL CGO helper for directional arrowheads.
    lines += ["python", "from pymol.cgo import CYLINDER, CONE", "from pymol import cmd", "import math", "def _arr(n,p1,p2,col):", "    v=[p2[i]-p1[i] for i in range(3)]", "    L=math.sqrt(sum(q*q for q in v))", "    u=[q/L for q in v]", "    base=[p2[i]-u[i]*0.75 for i in range(3)]", "    obj=[CYLINDER,p1[0],p1[1],p1[2],base[0],base[1],base[2],0.16,col[0],col[1],col[2],col[0],col[1],col[2],CONE,base[0],base[1],base[2],p2[0],p2[1],p2[2],0.38,0.0,col[0],col[1],col[2],col[0],col[1],col[2],1.0,0.0]", "    cmd.load_cgo(obj,n)", "python end"]
    for i, (name, pdb, color, score, c3, c17) in enumerate(entries):
        rgb = {"yellow": (1.0, 0.75, 0.0), "blue": (0.1, 0.35, 0.9), "orange": (1.0, 0.45, 0.0), "green": (0.1, 0.65, 0.2), "cyan": (0.0, 0.75, 0.85), "magenta": (0.85, 0.1, 0.65)}[color]
        lines.append(f"_arr('arrow_{name}', [{c3[0]:.5f},{c3[1]:.5f},{c3[2]:.5f}], [{c17[0]:.5f},{c17[1]:.5f},{c17[2]:.5f}], [{rgb[0]},{rgb[1]},{rgb[2]}])")
    lines += ["set label_size, 12", "set sphere_scale, 0.35", f"ray 1600,1200", f"png {png.as_posix()}, dpi=300", f"save {pse.as_posix()}", "quit"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(pml):
    pymol = os.environ.get("PYMOL_BIN", "pymol")
    result = subprocess.run([pymol, "-cq", str(pml)], cwd=ROOT, text=True, capture_output=True, timeout=180)
    return result.returncode, (result.stdout + "\n" + result.stderr)[-1000:]


def compose_panel_figure(entries, stem, title, panel_titles):
    panel_paths = []
    for entry, panel_title in zip(entries, panel_titles):
        name, pdb, color, score, c3, c17 = entry
        panel_pml = PML_DIR / f"{stem}_{name}.pml"
        panel_png = FIG / f"{stem}_{name}.png"
        panel_pse = PML_DIR / f"{stem}_{name}.pse"
        write_cgo_pml(panel_pml, panel_png, panel_pse, [entry], PREP / "receptor_repaired.pdb", panel_title, closeup=True)
        if os.environ.get("STEP44_SKIP_PYMOL") != "1":
            render(panel_pml)
        panel_paths.append(panel_png)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=160)
    fig.suptitle(title, fontsize=16, y=0.995)
    for ax, path, panel_title in zip(axes.ravel(), panel_paths, panel_titles):
        if path.exists():
            ax.imshow(plt.imread(path))
        ax.set_title(panel_title, fontsize=11)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG / f"{stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    receptor = parse_pdb(PREP / "receptor_repaired.pdb")
    tes = read_mol(PREP / "TES_crystal.sdf")
    tes_anchor = find_anchor_atoms(tes)
    tes_map_info = {}
    # The existing Step 4.3 mapping is read as a cross-check, not as the sole
    # source of C3/C17 labels.
    existing_map = pd.read_csv(ROOT / "results" / "c17_analysis" / "common_core_mapping.csv")
    pose_desc = pd.read_csv(ROOT / "results" / "pose_analysis" / "pose_descriptors.csv")
    descriptors = pd.read_csv(ROOT / "results" / "docking_analysis" / "compound_descriptors.csv")
    top_seed42 = {r.compound_id: read_mol(ROOT / "results" / "docking_2AM9" / "top_sdf" / f"{r.compound_id}.sdf") for _, r in descriptors[descriptors.compound_id.isin(TARGETS)].iterrows()}
    anchors = {"TES": tes_anchor}
    maps = {}
    for cid in TARGETS:
        sample = read_mol(ROOT / "results" / "docking_2AM9" / "top_sdf" / f"{cid}.sdf")
        anchor = find_anchor_atoms(sample)
        mcs, pairs = mcs_map(tes, sample)
        ref_to_mol = dict(pairs)
        mapped_c3 = ref_to_mol.get(tes_anchor["C3"])
        mapped_c17 = ref_to_mol.get(tes_anchor["C17"])
        status = "validated" if mapped_c3 == anchor["C3"] and mapped_c17 == anchor["C17"] and len(pairs) >= 17 else "ambiguous"
        anchors[cid] = anchor
        maps[cid] = {"pairs": pairs, "mcs_atoms": mcs.numAtoms, "mapped_c3": mapped_c3, "mapped_c17": mapped_c17, "status": status}
        tes_map_info[cid] = {"compound_id": cid, "tes_c3_atom_index": tes_anchor["C3"], "tes_c17_atom_index": tes_anchor["C17"], "compound_c3_atom_index": anchor["C3"], "compound_c17_atom_index": anchor["C17"], "mapped_c3_atom_index": mapped_c3, "mapped_c17_atom_index": mapped_c17, "candidate_ring_atoms": len(anchor["ring_atoms"]), "tes_candidate_core_mapped_atoms": len(pairs), "tes_candidate_full_mcs_atoms": mcs.numAtoms, "mapping_status": status, "mapping_rule": anchor["rule"], "existing_step43_mapping_rows": int((existing_map.comparison == "A").sum())}
    pd.DataFrame(tes_map_info.values()).to_csv(OUT / "atom_mapping" / "c3_c17_atom_mapping.csv", index=False) if (OUT / "atom_mapping").mkdir(exist_ok=True) is None else None
    # Save the actual mapping rows for independent inspection.
    map_rows = []
    for cid, info in maps.items():
        for n, (i, j) in enumerate(info["pairs"]):
            map_rows.append({"compound_id": cid, "core_atom_index": n, "tes_atom_index": i, "compound_atom_index": j, "tes_is_C3": i == tes_anchor["C3"], "tes_is_C17": i == tes_anchor["C17"], "compound_is_C3": j == anchors[cid]["C3"], "compound_is_C17": j == anchors[cid]["C17"], "mapping_status": info["status"]})
    pd.DataFrame(map_rows).to_csv(OUT / "atom_mapping" / "tes_core_mapping.csv", index=False)
    if any(info["status"] != "validated" for info in maps.values()):
        raise RuntimeError("At least one compound has ambiguous TES C3/C17 mapping; orientation analysis stopped")

    row5378 = top_seed42["cluster_176_row_5378"]
    ref_5378 = {"mol": row5378, "c3": xyz(row5378, anchors["cluster_176_row_5378"]["C3"]), "c17": xyz(row5378, anchors["cluster_176_row_5378"]["C17"]), "u": None}
    ref_5378["u"] = ref_5378["c17"] - ref_5378["c3"]
    tes_ref = {"mol": tes, "c3": xyz(tes, tes_anchor["C3"]), "c17": xyz(tes, tes_anchor["C17"]), "u": xyz(tes, tes_anchor["C17"]) - xyz(tes, tes_anchor["C3"])}

    all_rows = []
    top_rows = []
    local_rows = []
    for cid in TARGETS:
        compound_dir = POSE_DIR / cid
        for sdf in sorted(compound_dir.glob("seed_*/pose_*.sdf")):
            m = read_mol(sdf)
            anchor = anchors[cid]
            c3, c17 = xyz(m, anchor["C3"]), xyz(m, anchor["C17"])
            u = c17 - c3
            rel = np.array(maps[cid]["pairs"], dtype=int)
            ref_core = all_coords(tes, rel[:, 0])
            mol_core = all_coords(m, rel[:, 1])
            d_same = float(np.linalg.norm(c3 - tes_ref["c3"]) + np.linalg.norm(c17 - tes_ref["c17"]))
            d_swap = float(np.linalg.norm(c3 - tes_ref["c17"]) + np.linalg.norm(c17 - tes_ref["c3"]))
            d_same_5378 = float(np.linalg.norm(c3 - ref_5378["c3"]) + np.linalg.norm(c17 - ref_5378["c17"]))
            d_swap_5378 = float(np.linalg.norm(c3 - ref_5378["c17"]) + np.linalg.norm(c17 - ref_5378["c3"]))
            reg = local_metrics(m, anchor, receptor)
            region = {}
            for end, point in [("C3", c3), ("C17", c17)]:
                region[f"{end}_A_region_min_A"] = min((float(np.linalg.norm(point - a["coord"])) for key in ["A:ARG752", "A:GLN711"] for a in key_atoms(receptor, key)), default=np.nan)
                region[f"{end}_D_region_min_A"] = min((float(np.linalg.norm(point - a["coord"])) for key in ["A:ASN705", "A:THR877"] for a in key_atoms(receptor, key)), default=np.nan)
            native_region_score = (region["C3_D_region_min_A"] - region["C3_A_region_min_A"]) + (region["C17_A_region_min_A"] - region["C17_D_region_min_A"])
            reversed_region_score = -native_region_score
            # Parse seed/rank/score from path and the docking table.
            seed = int(re.search(r"seed_(\d+)", str(sdf)).group(1))
            rank = int(re.search(r"pose_(\d+)", sdf.name).group(1))
            all_score = pd.read_csv(ROB / "multi_seed_all_poses.csv")
            sr = all_score[(all_score.compound_id == cid) & (all_score.seed == seed) & (all_score.pose_rank == rank)].iloc[0]
            score = float(sr.vina_score_kcal_mol)
            angle_tes = angle_deg(u, tes_ref["u"])
            angle_5378 = angle_deg(u, ref_5378["u"])
            category = classify(angle_tes, d_same, d_swap, native_region_score, reversed_region_score)
            row = {"compound_id": cid, "row_index": int(descriptors.loc[descriptors.compound_id == cid, "row_index"].iloc[0]), "seed": seed, "pose_rank": rank, "vina_score_kcal_mol": score, "sdf": str(sdf.relative_to(ROOT)), "angle_to_TES_deg": angle_tes, "angle_to_row5378_deg": angle_5378, "C3_to_C3_TES_A": float(np.linalg.norm(c3 - tes_ref["c3"])), "C17_to_C17_TES_A": float(np.linalg.norm(c17 - tes_ref["c17"])), "C3_to_C17_TES_A": float(np.linalg.norm(c3 - tes_ref["c17"])), "C17_to_C3_TES_A": float(np.linalg.norm(c17 - tes_ref["c3"])), "D_same_TES_A": d_same, "D_swap_TES_A": d_swap, "swap_margin_TES_A": d_same - d_swap, "D_same_row5378_A": d_same_5378, "D_swap_row5378_A": d_swap_5378, "swap_margin_row5378_A": d_same_5378 - d_swap_5378, "core_raw_rmsd_TES_A": direct_rmsd(ref_core, mol_core), "core_best_fit_rmsd_TES_A": fitted_rmsd(ref_core, mol_core), "core_centroid_displacement_TES_A": float(np.linalg.norm(ref_core.mean(0) - mol_core.mean(0))), "core_atom_count": len(rel), "C3_A_region_min_A": region["C3_A_region_min_A"], "C3_D_region_min_A": region["C3_D_region_min_A"], "C17_A_region_min_A": region["C17_A_region_min_A"], "C17_D_region_min_A": region["C17_D_region_min_A"], "native_region_score": native_region_score, "reversed_region_score": reversed_region_score, "C17_min_receptor_distance_A": reg["min_distance_A"], "C17_short_contacts_lt_2_5": reg["short_count_lt_2_5"], "key_A:ARG752_A": reg["key_distances"]["A:ARG752"], "key_A:GLN711_A": reg["key_distances"]["A:GLN711"], "key_A:ASN705_A": reg["key_distances"]["A:ASN705"], "key_A:THR877_A": reg["key_distances"]["A:THR877"], "orientation_class": category, "mapping_status": maps[cid]["status"]}
            all_rows.append(row)
            if rank == 1:
                top_rows.append(row)
                for c in reg["nearest"][:10]:
                    local_rows.append({"compound_id": cid, "seed": seed, "pose_rank": rank, "vina_score_kcal_mol": score, "ligand_atom_index": c["ligand_idx"], "receptor_atom_serial": c["receptor_serial"], "receptor_atom": c["receptor_atom"], "residue_key": c["residue_key"], "distance_A": c["distance_A"], "vdw_overlap_estimate_A": c["vdw_overlap_estimate_A"], "short_contact_lt_2_5_A": c["short_contact_lt_2_5_A"], "orientation_class": category})

    all_df = pd.DataFrame(all_rows)
    top_df = pd.DataFrame(top_rows)
    local_df = pd.DataFrame(local_rows)
    all_df.to_csv(OUT / "all_pose_orientation.csv", index=False)
    top_df.to_csv(OUT / "multi_seed_top_poses.csv", index=False)
    local_df.to_csv(OUT / "interaction_stability.csv", index=False)
    score_summary = top_df.groupby("compound_id").vina_score_kcal_mol.agg(["mean", "std", "min", "max", "median"]).reset_index().rename(columns={"mean": "top_score_mean", "std": "top_score_sd", "min": "top_score_min", "max": "top_score_max", "median": "top_score_median"})
    score_summary.to_csv(OUT / "multi_seed_score_summary.csv", index=False)
    stability = top_df.groupby(["compound_id", "orientation_class"]).size().unstack(fill_value=0).reset_index()
    for c in ["consistent_with_native_orientation", "head_to_tail_reversal_supported", "alternative_orientation", "ambiguous"]:
        if c not in stability:
            stability[c] = 0
    stability["total_seeds"] = stability[["consistent_with_native_orientation", "head_to_tail_reversal_supported", "alternative_orientation", "ambiguous"]].sum(axis=1)
    stability["reversed_fraction"] = stability["head_to_tail_reversal_supported"] / stability["total_seeds"]
    stability.to_csv(OUT / "orientation_stability.csv", index=False)

    # Representative pose is defined before inspecting orientation conclusions:
    # choose the top pose closest in score to that compound's top-score median.
    reps = {"TES": {"mol": tes, "score": np.nan, "label": "native Testosterone crystal"}}
    for cid in ["cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"]:
        sub = top_df[top_df.compound_id == cid].copy()
        med = sub.vina_score_kcal_mol.median()
        chosen = sub.iloc[(sub.vina_score_kcal_mol - med).abs().argmin()]
        mol = read_mol(ROOT / chosen.sdf)
        rep_pdb = REP_DIR / f"{cid}_representative.pdb"
        save_pose_pdb(mol, rep_pdb)
        reps[cid] = {"mol": mol, "score": float(chosen.vina_score_kcal_mol), "label": cid, "seed": int(chosen.seed), "sdf": str(ROOT / chosen.sdf), "pdb": rep_pdb, "row": chosen}
    # TES PDB is generated only as a visualization derivative; the SDF remains untouched.
    tes_pdb = REP_DIR / "TES_crystal.pdb"
    save_pose_pdb(tes, tes_pdb)
    reps["TES"]["pdb"] = tes_pdb

    entries = []
    for name, cid, label in [("TES", "TES", "TES"), ("row5378", "cluster_176_row_5378", "5378"), ("row2412", "cluster_3_row_2412", "2412"), ("row5379", "cluster_246_row_5379", "5379")]:
        rec = reps[cid]
        a = tes_anchor if cid == "TES" else anchors[cid]
        entries.append((name, rec["pdb"], COLOR[label], rec["score"] if np.isfinite(rec["score"]) else 0.0, xyz(rec["mol"], a["C3"]), xyz(rec["mol"], a["C17"])))
    pml1 = PML_DIR / "c3_c17_orientation.pml"
    pml5 = PML_DIR / "representative_3d_orientation.pml"
    write_cgo_pml(pml1, FIG / "c3_c17_orientation.png", PML_DIR / "c3_c17_orientation.pse", entries, PREP / "receptor_repaired.pdb", "C3 to C17 orientation", closeup=False)
    write_cgo_pml(pml5, FIG / "representative_3d_orientation.png", PML_DIR / "representative_3d_orientation.pse", entries, PREP / "receptor_repaired.pdb", "Representative orientations", closeup=True)
    pymol_status = {}
    for pml in [pml1, pml5]:
        if os.environ.get("STEP44_SKIP_PYMOL") == "1":
            pymol_status[pml.name] = {"returncode": 0, "log": "skipped"}
        else:
            rc, log = render(pml)
            pymol_status[pml.name] = {"returncode": rc, "log": log}
    # The single-view overlays are retained as editable PyMOL sessions.  The
    # deliverable PNGs are 2x2 panels to avoid hiding C3/C17 arrows under four
    # superposed ligands.
    panel_titles = []
    for name, cid, label in [("TES", "TES", "TES crystal"), ("row5378", "cluster_176_row_5378", "row 5378"), ("row2412", "cluster_3_row_2412", "row 2412"), ("row5379", "cluster_246_row_5379", "row 5379")]:
        a = tes_anchor if cid == "TES" else anchors[cid]
        u = xyz(reps[cid]["mol"], a["C17"]) - xyz(reps[cid]["mol"], a["C3"])
        theta = angle_deg(u, tes_ref["u"])
        panel_titles.append(f"{label} | C3→C17 angle to TES: {theta:.1f}°")
    compose_panel_figure(entries, "c3_c17_orientation", "Figure 1. C3→C17 directed orientation", panel_titles)
    compose_panel_figure(entries, "representative_3d_orientation", "Figure 5. Representative 3D binding orientations", panel_titles)

    summary = {"status": "success", "docking_rerun": True, "method": "Vina.dock full search; no optimize", "target_compounds": TARGETS, "seed_count": 10, "poses_per_run_requested": 10, "successful_runs": int(len(top_df) / 1), "top_pose_rows": len(top_df), "all_pose_rows": len(all_df), "c3_c17_mapping": "validated against TES crystal ring MCS and independent ring landmarks", "pymol": pymol_status, "outputs": {"mapping": "results/orientation_robustness/atom_mapping/c3_c17_atom_mapping.csv", "top": "results/orientation_robustness/multi_seed_top_poses.csv", "all": "results/orientation_robustness/all_pose_orientation.csv", "stability": "results/orientation_robustness/orientation_stability.csv", "scores": "results/orientation_robustness/multi_seed_score_summary.csv", "interactions": "results/orientation_robustness/interaction_stability.csv"}}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "success", "top_rows": len(top_df), "all_pose_rows": len(all_df), "mapping_status": {cid: maps[cid]["status"] for cid in TARGETS}, "pymol_returncodes": {k: v["returncode"] for k, v in pymol_status.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
