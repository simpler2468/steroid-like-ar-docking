"""Step 6 Part A: C17 local packing validation from existing poses only.

This script never calls Vina.  It reads the existing 2AM9/2AMA top-ranked
pose SDF files for Row 5373 and Row 705, recomputes contact metrics using the
same receptor/PDB coordinate frames, and writes machine-readable summaries
and a publication-style diagnostic figure.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_analysis"
FIG = ROOT / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "cluster_416_row_5373": {"row_index": 5373, "label": "Row 5373"},
    "cluster_176_row_705": {"row_index": 705, "label": "Row 705"},
}
SEEDS = {42, 123, 2026, 7, 99, 11, 17, 73, 101, 314}
RECEPTORS = {
    "2AM9": ROOT / "data" / "prepared" / "2AM9" / "receptor_repaired.pdb",
    "2AMA": ROOT / "data" / "prepared" / "2AMA" / "receptor_aligned.pdb",
}
POSE_TABLES = {
    "2AM9": ROOT / "results" / "orientation_robustness" / "all_pose_orientation.csv",
    "2AMA": ROOT / "results" / "receptor_ensemble" / "multiseed_orientation" / "2AMA" / "orientation_all_poses.csv",
}
KEY_RESIDUES = {"ASN705", "THR877", "LEU701"}
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}


def load_step44():
    path = ROOT / "scripts" / "08_orientation_verification.py"
    spec = importlib.util.spec_from_file_location("step44_orientation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_receptor(path: Path) -> list[dict]:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        element = line[76:78].strip().upper() or line[12:16].strip()[0].upper()
        if element == "H":
            continue
        atoms.append({
            "serial": int(line[6:11]), "name": line[12:16].strip(),
            "resname": line[17:20].strip(), "chain": line[21].strip() or "_",
            "resseq": int(line[22:26]), "element": element,
            "coord": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
        })
    if not atoms:
        raise ValueError(f"No protein atoms found in {path}")
    return atoms


def residue_label(atom: dict) -> str:
    return f"{atom['resname']}{atom['resseq']}"


def xyz(mol: Chem.Mol, idx: int) -> np.ndarray:
    p = mol.GetConformer().GetAtomPosition(int(idx))
    return np.array([p.x, p.y, p.z], dtype=float)


def heavy_atoms(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def branch_atoms(mol: Chem.Mol, start: int, blocked: set[int]) -> set[int]:
    result: set[int] = set()
    stack = [start]
    while stack:
        idx = stack.pop()
        if idx in result or idx in blocked or mol.GetAtomWithIdx(idx).GetAtomicNum() == 1:
            continue
        result.add(idx)
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            if neighbor.GetIdx() not in blocked:
                stack.append(neighbor.GetIdx())
    return result


def c17_regions(mol: Chem.Mol, step44) -> dict[str, set[int]]:
    anchors = step44.find_anchor_atoms(mol)
    c17 = anchors["C17"]
    ring = set(anchors["ring_atoms"])
    direct_nonring = [n for n in mol.GetAtomWithIdx(c17).GetNeighbors() if n.GetIdx() not in ring and n.GetAtomicNum() > 1]
    oxygen = {n.GetIdx() for n in direct_nonring if n.GetAtomicNum() == 8}
    substituent: set[int] = set()
    ethynyl: set[int] = set()
    for neighbor in direct_nonring:
        branch = branch_atoms(mol, neighbor.GetIdx(), {c17} | ring)
        if neighbor.GetAtomicNum() != 8:
            substituent.update(branch)
            if any(b.GetBondType() == Chem.BondType.TRIPLE for idx in branch for b in mol.GetAtomWithIdx(idx).GetBonds()):
                ethynyl.update(branch)
    return {
        "c17_core": {c17},
        "c17_oh": {c17} | oxygen,
        "c17_nonring_substituent": substituent,
        "c17_local": {c17} | oxygen | substituent,
        "c17_ethynyl": ethynyl,
    }


def region_metrics(mol: Chem.Mol, indices: set[int], receptor: list[dict]) -> dict:
    if not indices:
        return {
            "region_atom_count": 0, "min_distance_A": np.nan,
            "nearest_ligand_atom": "", "nearest_ligand_element": "",
            "nearest_receptor_atom": "", "nearest_residue": "",
            "nearest_receptor_element": "", "vdw_overlap_estimate_A": np.nan,
            "short_contact_lt_2_5_A": False, "polar_pair_candidate_lt_3_5_A": False,
            "nearby_residues_5A": "",
        }
    pairs = []
    for li in sorted(indices):
        atom = mol.GetAtomWithIdx(li)
        if atom.GetAtomicNum() == 1:
            continue
        p = xyz(mol, li)
        for ra in receptor:
            d = float(np.linalg.norm(p - ra["coord"]))
            ligand_element = atom.GetSymbol().upper()
            overlap = VDW.get(ligand_element, 1.70) + VDW.get(ra["element"], 1.70) - d
            polar = ligand_element in {"N", "O", "S"} and ra["element"] in {"N", "O", "S"} and d <= 3.5
            pairs.append({"li": li, "ligand_element": ligand_element, "ra": ra, "distance": d, "overlap": overlap, "polar": polar})
    pairs.sort(key=lambda x: x["distance"])
    nearest = pairs[0]
    residue_min = {}
    for pair in pairs:
        label = residue_label(pair["ra"])
        residue_min[label] = min(residue_min.get(label, np.inf), pair["distance"])
    nearby = sorted((label for label, d in residue_min.items() if d <= 5.0), key=lambda label: residue_min[label])
    return {
        "region_atom_count": len([i for i in indices if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]),
        "min_distance_A": nearest["distance"],
        "nearest_ligand_atom": str(nearest["li"]),
        "nearest_ligand_element": nearest["ligand_element"],
        "nearest_receptor_atom": f"{nearest['ra']['chain']}:{nearest['ra']['name']}",
        "nearest_residue": residue_label(nearest["ra"]),
        "nearest_receptor_element": nearest["ra"]["element"],
        "vdw_overlap_estimate_A": nearest["overlap"],
        "short_contact_lt_2_5_A": bool(nearest["distance"] < 2.5),
        "polar_pair_candidate_lt_3_5_A": any(p["polar"] for p in pairs),
        "nearby_residues_5A": ";".join(nearby),
    }


def residue_metrics(mol: Chem.Mol, indices: set[int], receptor: list[dict], residue_names: set[str]) -> list[dict]:
    result = []
    for residue in sorted(residue_names):
        target = [a for a in receptor if residue_label(a) == residue]
        d = min((float(np.linalg.norm(xyz(mol, li) - ra["coord"])) for li in indices if mol.GetAtomWithIdx(li).GetAtomicNum() > 1 for ra in target), default=np.nan)
        result.append({"residue": residue, "min_distance_A": d, "within_5A": bool(np.isfinite(d) and d <= 5.0), "within_2_5A": bool(np.isfinite(d) and d < 2.5), "is_key_residue": residue in KEY_RESIDUES})
    return result


def read_top_rows(receptor: str) -> pd.DataFrame:
    table = pd.read_csv(POSE_TABLES[receptor])
    table["seed"] = table["seed"].astype(int)
    table["pose_rank"] = table["pose_rank"].astype(int)
    top = table[(table["compound_id"].isin(TARGETS)) & (table["pose_rank"] == 1) & (table["seed"].isin(SEEDS))].copy()
    if len(top) != 20:
        raise ValueError(f"{receptor}: expected 20 target top poses, found {len(top)}")
    return top


def make_pymol_script(rows: pd.DataFrame) -> None:
    path = FIG / "figure_H3_c17_environment.pml"
    lines = ["reinitialize", "bg_color white", "# Coordinates are loaded from existing project PDB/SDF outputs; no docking is run here."]
    for receptor in ("2AM9", "2AMA"):
        rec = RECEPTORS[receptor].relative_to(ROOT).as_posix()
        lines += [f"load {rec}, {receptor}_receptor", f"hide everything, {receptor}_receptor", f"show cartoon, {receptor}_receptor", f"color gray80, {receptor}_receptor"]
        for compound in TARGETS:
            row = rows[(rows.receptor == receptor) & (rows.compound_id == compound) & (rows.seed == 42)].iloc[0]
            sdf = Path(row.pose_sdf).as_posix()
            obj = f"{receptor}_{compound.split('_')[-1]}_seed42"
            lines += [f"load {sdf}, {obj}", f"show sticks, {obj}", f"color {'cyan' if row.row_index == 5373 else 'magenta'}, {obj}", f"label {obj} and name C17, 'Row {int(row.row_index)} C17'"]
        lines += [f"select {receptor}_key, {receptor}_receptor and resi 701+705+877", f"show sticks, {receptor}_key", f"color wheat, {receptor}_key", f"zoom {receptor}_receptor and resi 701+705+877, 7"]
    lines += ["set label_size, 12", "# Run with PyMOL from the project root for interactive inspection."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = [("2AM9", 5373), ("2AMA", 5373), ("2AM9", 705), ("2AMA", 705)]
    labels = [f"{r}\nRow {c}" for r, c in order]
    colors = {5373: "#1f77b4", 705: "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), dpi=180)
    ax = axes[0, 0]
    for x, (rec, row) in enumerate(order):
        vals = df[(df.receptor == rec) & (df.row_index == row) & (df.region == "c17_local")]["min_distance_A"].dropna().to_numpy()
        jitter = np.linspace(-0.13, 0.13, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=30, color=colors[row], alpha=0.78, edgecolor="white", linewidth=0.4, label=f"Row {row}" if x in (0, 2) else None)
        ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color="black", capsize=4, markersize=4, linewidth=1.2)
    ax.axhline(2.5, color="#777777", linestyle="--", linewidth=1, label="2.5 Å screen")
    ax.set_xticks(range(4), labels); ax.set_ylabel("C17 local minimum distance (Å)"); ax.set_title("C17 local packing: all 40 top poses")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    region_order = ["c17_oh", "c17_nonring_substituent"]
    offsets = {"c17_oh": -0.17, "c17_nonring_substituent": 0.17}
    for x, (rec, row) in enumerate(order):
        for region in region_order:
            vals = df[(df.receptor == rec) & (df.row_index == row) & (df.region == region)]["min_distance_A"].dropna().to_numpy()
            if not len(vals):
                continue
            xx = x + offsets[region]
            ax.scatter(np.full(len(vals), xx) + np.linspace(-0.06, 0.06, len(vals)), vals, s=22, color=colors[row], alpha=0.7, marker="o" if region == "c17_oh" else "^")
            ax.errorbar(xx, vals.mean(), yerr=vals.std(ddof=1), fmt="_", color="black", capsize=3, linewidth=1.4)
    ax.axhline(2.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(range(4), labels); ax.set_ylabel("Region minimum distance (Å)"); ax.set_title("Shared C17–OH vs non-ring substituent")
    ax.text(0.01, 0.02, "● C17–OH   ▲ non-ring substituent", transform=ax.transAxes, fontsize=8)

    ax = axes[1, 0]
    plot_regions = ["c17_local", "c17_oh", "c17_nonring_substituent"]
    plot_labels = ["overall\nC17 local", "shared\nC17–OH", "extra\nsubstituent"]
    bar_order = [("2AM9", 5373), ("2AMA", 5373), ("2AM9", 705), ("2AMA", 705)]
    width = 0.24
    for j, region in enumerate(plot_regions):
        vals = []
        for rec, row in bar_order:
            sub = df[(df.receptor == rec) & (df.row_index == row) & (df.region == region)]
            vals.append(float(sub["short_contact_lt_2_5_A"].mean()) if len(sub) else np.nan)
        ax.bar(np.arange(4) + (j - 1) * width, vals, width, label=plot_labels[j], color=["#4c78a8", "#72b7b2", "#f58518"][j], alpha=0.85)
    ax.set_xticks(range(4), labels); ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction of runs < 2.5 Å"); ax.set_title("Short-contact frequency")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    key = summary[(summary.region == "c17_local") & (summary.residue.isin(sorted(KEY_RESIDUES)))].copy()
    key["x"] = [f"{r}\n{c}" for r, c in zip(key.receptor, key.row_index)]
    pivot = key.pivot(index="residue", columns="x", values="mean_distance_A").reindex(sorted(KEY_RESIDUES))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), list(pivot.columns)); ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_title("C17 local distance to selected residues"); ax.set_xlabel("Receptor / compound"); ax.set_ylabel("Residue")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value): ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value < np.nanmedian(pivot.to_numpy()) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean minimum distance (Å)")
    fig.suptitle("H3 local packing robustness from existing multi-seed top poses", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "Distances are geometric proximity measures, not standalone hydrogen-bond or steric-clash proofs; all points are existing poses.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"figure_H3_local_packing_robustness.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    step44 = load_step44()
    rows = []
    residue_rows = []
    mapping_info = {}
    for receptor_name, receptor_path in RECEPTORS.items():
        receptor = parse_receptor(receptor_path)
        table = read_top_rows(receptor_name)
        for _, item in table.iterrows():
            sdf_path = ROOT / str(item["sdf"])
            mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False, sanitize=True)
            if mol is None or mol.GetNumConformers() == 0:
                raise ValueError(f"Invalid pose SDF: {sdf_path}")
            regions = c17_regions(mol, step44)
            cid = str(item["compound_id"])
            mapping_info[cid] = {k: sorted(v) for k, v in regions.items()}
            for region, indices in regions.items():
                metrics = region_metrics(mol, indices, receptor)
                row = {
                    "receptor": receptor_name, "compound_id": cid, "row_index": int(item["row_index"]),
                    "seed": int(item["seed"]), "pose_rank": int(item["pose_rank"]),
                    "vina_score_kcal_mol": float(item["vina_score_kcal_mol"]), "region": region,
                    "pose_sdf": str(sdf_path.relative_to(ROOT)), **metrics,
                }
                rows.append(row)
                for rm in residue_metrics(mol, indices, receptor, KEY_RESIDUES):
                    residue_rows.append({"receptor": receptor_name, "compound_id": cid, "row_index": int(item["row_index"]), "seed": int(item["seed"]), "region": region, **rm})
    contacts = pd.DataFrame(rows)
    residue_df = pd.DataFrame(residue_rows)
    if len(contacts[contacts.region == "c17_local"]) != 40:
        raise ValueError("Expected exactly 40 C17-local top-pose rows")
    summary_rows = []
    for (receptor, row_index, compound_id, region), group in contacts.groupby(["receptor", "row_index", "compound_id", "region"], dropna=False):
        distances = group["min_distance_A"].dropna()
        nearest = group.loc[group["nearest_residue"].astype(str).str.len() > 0, "nearest_residue"]
        summary_rows.append({
            "receptor": receptor, "compound_id": compound_id, "row_index": row_index, "region": region,
            "n_runs": len(group), "mean_distance_A": distances.mean() if len(distances) else np.nan, "sd_distance_A": distances.std(ddof=1) if len(distances) > 1 else np.nan,
            "median_distance_A": distances.median() if len(distances) else np.nan, "min_distance_A": distances.min() if len(distances) else np.nan, "max_distance_A": distances.max() if len(distances) else np.nan,
            "runs_lt_2_5_A": int(group["short_contact_lt_2_5_A"].sum()), "fraction_lt_2_5_A": group["short_contact_lt_2_5_A"].mean(),
            "polar_pair_candidate_runs": int(group["polar_pair_candidate_lt_3_5_A"].sum()),
            "most_frequent_nearest_residue": nearest.mode().iloc[0] if len(nearest) and len(nearest.mode()) else "",
        })
    summary = pd.DataFrame(summary_rows)
    contacts.to_csv(OUT / "h3_local_contacts_all_runs.csv", index=False)
    summary.to_csv(OUT / "h3_contact_summary.csv", index=False)
    residue_df.to_csv(OUT / "h3_residue_contact_comparison.csv", index=False)
    make_pymol_script(contacts[(contacts.region == "c17_local")])
    make_figure(contacts, residue_df.groupby(["receptor", "row_index", "compound_id", "region", "residue"], as_index=False).agg(mean_distance_A=("min_distance_A", "mean")))
    statuses = {r: "available" for r in RECEPTORS}
    validation = {
        "status": "pass", "step": "6A", "docking_rerun": False, "target_compounds": [5373, 705],
        "receptors": statuses, "expected_top_pose_runs": 40, "actual_top_pose_runs": int(len(contacts[contacts.region == "c17_local"])),
        "all_pose_rank_one": bool(((contacts[contacts.region == "c17_local"])["pose_rank"] == 1).all()),
        "seeds": sorted(SEEDS), "regions": sorted(contacts.region.unique()), "atom_mapping": mapping_info,
        "distance_screen_note": "<2.5 A is a geometric screen only; no contact is labeled a hydrogen bond or steric clash from distance alone.",
        "source_tables": {k: str(v.relative_to(ROOT)) for k, v in POSE_TABLES.items()},
    }
    (OUT / "h3_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "pass", "runs": len(contacts[contacts.region == "c17_local"]), "summary_rows": len(summary), "files": [str((OUT / f).relative_to(ROOT)) for f in ["h3_local_contacts_all_runs.csv", "h3_contact_summary.csv", "h3_residue_contact_comparison.csv", "h3_validation.json"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
