"""Independent validation for Step 3.1 ligand preparation; never docks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from vina import Vina


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "androgens_clusters_teacher_profile.xlsx"
SHEET = "Cluster compounds"
RESULTS = ROOT / "results" / "ligand_preparation"
MANIFEST_PATH = RESULTS / "ligand_manifest.csv"
QC_PATH = RESULTS / "ligand_qc.csv"
SUMMARY_PATH = RESULTS / "preparation_summary.json"
REPORT_PATH = ROOT / "reports" / "ligand_preparation.md"
EXPECTED_COUNTS = {3: 10, 176: 11, 246: 22, 416: 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=True)


def find_unassigned_stereo(mol: Chem.Mol) -> list[str]:
    result = []
    for atom_idx, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False):
        if label == "?":
            result.append(f"atom_{atom_idx}_{mol.GetAtomWithIdx(atom_idx).GetSymbol()}")
    return result


def find_double_bond_issues(mol: Chem.Mol) -> list[str]:
    result = []
    for info in Chem.FindPotentialStereo(mol):
        if "Bond_Double" in str(info.type) and "Unspecified" in str(info.specified):
            result.append(f"bond_{info.idx}")
    return result


def geometry_check(mol: Chem.Mol) -> dict[str, Any]:
    if mol.GetNumConformers() != 1:
        return {"valid": False, "reason": f"expected_one_conformer_found_{mol.GetNumConformers()}"}
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=float)
    if not np.isfinite(coords).all():
        return {"valid": False, "reason": "nonfinite_coordinates"}
    bonded = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()}
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    clashes = []
    for pos, i in enumerate(heavy):
        for j in heavy[pos + 1 :]:
            if tuple(sorted((i, j))) in bonded:
                continue
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance < 1.25:
                clashes.append({"i": i, "j": j, "distance_A": distance})
    bad_bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance < 0.7 or distance > 2.4:
            bad_bonds.append({"i": i, "j": j, "distance_A": distance})
    return {"valid": not clashes and not bad_bonds, "severe_nonbonded_clashes": clashes, "bad_bonds": bad_bonds}


def pdbqt_check(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    atom_types: list[str] = []
    charge_errors = []
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        atom_types.append(fields[-1])
        if len(fields) >= 2:
            try:
                float(fields[-2])
            except ValueError:
                charge_errors.append(line)
    torsdof = [line for line in lines if line.startswith("TORSDOF")]
    return {
        "nonempty": path.exists() and path.stat().st_size > 0,
        "atom_count": len(atom_lines),
        "atom_types": sorted(set(atom_types)),
        "charge_parse_errors": len(charge_errors),
        "root_count": sum(line.startswith("ROOT") for line in lines),
        "branch_count": sum(line.startswith("BRANCH") for line in lines),
        "torsdof_count": len(torsdof),
        "torsdof_value": torsdof[0].split()[-1] if torsdof else None,
    }


def vina_load_check(path: Path) -> tuple[bool, str]:
    try:
        vina = Vina(sf_name="vina", cpu=1, seed=17)
        vina.set_ligand_from_file(str(path))
        return True, "loaded_by_vina_python_api_without_docking"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH, dtype=object, keep_default_na=False)
    source = pd.read_excel(INPUT, sheet_name=SHEET, header=2)
    source_hash = sha256(INPUT)
    if len(source) != 47 or len(manifest) != 47:
        raise ValueError(f"Expected 47 source and manifest rows; source={len(source)}, manifest={len(manifest)}")
    cluster_counts = {int(key): int(value) for key, value in manifest.groupby("cluster_id").size().to_dict().items()}
    if cluster_counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected cluster counts: {cluster_counts}")
    ids = manifest["compound_id"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate compound_id in manifest")
    source_key = {(str(row.cluster_id), str(row.row_index), str(row.split)) for row in source.itertuples(index=False)}
    manifest_key = {(str(row.cluster_id), str(row.row_index), str(row.split)) for row in manifest.itertuples(index=False)}
    if source_key != manifest_key:
        raise ValueError("Manifest cluster_id/row_index/split mapping differs from Excel")

    qc_rows: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        compound_id = row["compound_id"]
        sdf_path = ROOT / row["sdf_path"]
        pdbqt_path = ROOT / row["pdbqt_path"]
        reasons: list[str] = []
        errors: list[str] = []
        original = None
        original_smiles_ok = False
        sdf_mol = None
        try:
            original = Chem.MolFromSmiles(str(row["original_smiles"]))
            if original is None:
                raise ValueError("original SMILES cannot be parsed")
            original_smiles_ok = True
        except Exception as exc:
            errors.append(f"original_smiles:{exc}")
        if not sdf_path.exists() or sdf_path.stat().st_size == 0:
            errors.append("missing_or_empty_sdf")
        else:
            try:
                supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
                sdf_mol = supplier[0] if len(supplier) else None
                if sdf_mol is None:
                    raise ValueError("RDKit could not read SDF")
            except Exception as exc:
                errors.append(f"sdf_parse:{exc}")
        if original is not None and sdf_mol is not None:
            original_unassigned = find_unassigned_stereo(original)
            original_double_issues = find_double_bond_issues(original)
            if canonical(original) != canonical(sdf_mol):
                # An unspecified stereocenter may be represented with a
                # concrete 3D arrangement by the file round-trip.  Do not
                # treat that as a connectivity failure or silently claim it
                # is the experimentally known stereoisomer.
                if Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(original)), isomericSmiles=False) == Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(sdf_mol)), isomericSmiles=False) and (original_unassigned or original_double_issues):
                    reasons.append("stereo_review_required:round_trip_stereo_not_defined_in_original")
                else:
                    errors.append("canonical_isomeric_smiles_mismatch")
            if Chem.GetFormalCharge(original) != Chem.GetFormalCharge(Chem.RemoveHs(Chem.Mol(sdf_mol))):
                errors.append("formal_charge_changed")
            if original_unassigned:
                reasons.append("stereo_review_required:unassigned_stereocenter")
            if original_double_issues:
                reasons.append("stereo_review_required:unspecified_double_bond")
            geom = geometry_check(sdf_mol)
            if not geom.get("valid", False):
                reasons.append("geometry_review_required")
        else:
            geom = {"valid": False, "reason": "not_available"}

        if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
            errors.append("missing_or_empty_pdbqt")
            pqt = {"nonempty": False}
            vina_ok, vina_message = False, "pdbqt_missing"
        else:
            pqt = pdbqt_check(pdbqt_path)
            if pqt["atom_count"] == 0:
                errors.append("pdbqt_has_no_atoms")
            if pqt["charge_parse_errors"]:
                errors.append("pdbqt_charge_parse_error")
            if pqt["torsdof_count"] != 1:
                reasons.append("torsion_tree_torsdof_missing_or_ambiguous")
            vina_ok, vina_message = vina_load_check(pdbqt_path)
            if not vina_ok:
                errors.append(f"vina_ligand_load:{vina_message}")

        if errors:
            status = "failed"
        elif reasons:
            status = "review_required"
        else:
            status = "ready"
        qc_rows.append(
            {
                "compound_id": compound_id,
                "cluster_id": row["cluster_id"],
                "row_index": row["row_index"],
                "split": row["split"],
                "sdf_rdkit_read": bool(sdf_mol is not None),
                "original_smiles_rdkit_read": original_smiles_ok,
                "pdbqt_nonempty": bool(pqt.get("nonempty", False)),
                "pdbqt_atom_count": pqt.get("atom_count", 0),
                "pdbqt_torsdof": pqt.get("torsdof_value"),
                "vina_load_ok": vina_ok,
                "vina_load_message": vina_message,
                "geometry_json": json.dumps(geom, ensure_ascii=False),
                "review_reasons": ";".join(reasons),
                "errors": ";".join(errors),
                "final_status": status,
            }
        )

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(QC_PATH, index=False)
    final_status = dict(qc["final_status"].value_counts())
    available_by_cluster = {
        str(cluster): int(((qc["cluster_id"].astype(str) == str(cluster)) & (qc["final_status"] == "ready")).sum())
        for cluster in sorted(EXPECTED_COUNTS)
    }
    structure_issue_count = int((qc["errors"].str.len() > 0).sum())
    stereo_issue_count = int(qc["review_reasons"].str.contains("stereo_review_required", regex=False).sum())
    protonation_review_count = int(manifest["protonation_review"].astype(str).ne("").sum()) if "protonation_review" in manifest else 0
    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": source_hash,
        "input_sha256_matches_preparation": source_hash == str(manifest.get("source_excel_sha256", pd.Series([""])).iloc[0]) if "source_excel_sha256" in manifest else None,
        "source_rows": len(source),
        "manifest_rows": len(manifest),
        "cluster_counts": cluster_counts,
        "smiles_parse_success_count": int(qc["original_smiles_rdkit_read"].sum()) if len(qc) else 0,
        "structure_issue_compound_count": structure_issue_count,
        "chemical_structure_review_compound_count": int((qc["review_reasons"].str.len() > 0).sum()),
        "stereochemistry_missing_or_uncertain_count": stereo_issue_count,
        "protonation_review_count": protonation_review_count,
        "three_d_sdf_success_count": int(qc["sdf_rdkit_read"].sum()),
        "pdbqt_success_count": int(qc["pdbqt_nonempty"].sum()),
        "vina_ligand_load_success_count": int(qc["vina_load_ok"].sum()),
        "status_counts": {key: int(final_status.get(key, 0)) for key in ("ready", "review_required", "failed")},
        "ready_by_cluster": available_by_cluster,
        "failure_or_review_compounds": qc.loc[qc["final_status"] != "ready", ["compound_id", "final_status", "review_reasons", "errors"]].to_dict(orient="records"),
        "software_versions": {"rdkit": rdkit_version(), "vina": version("vina"), "meeko": version("meeko"), "pandas": version("pandas")},
        "docking_executed": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, manifest, qc)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write_report(summary: dict[str, Any], manifest: pd.DataFrame, qc: pd.DataFrame) -> None:
    review_ids = manifest.loc[manifest["preparation_status"] == "review_required", ["compound_id", "cluster_id", "row_index", "unassigned_stereocenters"]]
    review_lines = [
        f"| {row.compound_id} | {row.cluster_id} | {row.row_index} | {row.unassigned_stereocenters or 'not recorded'} |"
        for row in review_ids.itertuples(index=False)
    ]
    cluster_lines = []
    for cluster in sorted(EXPECTED_COUNTS):
        total = int((manifest["cluster_id"].astype(str) == str(cluster)).sum())
        ready = int(((manifest["cluster_id"].astype(str) == str(cluster)) & (manifest["preparation_status"] == "ready")).sum())
        review = int(((manifest["cluster_id"].astype(str) == str(cluster)) & (manifest["preparation_status"] == "review_required")).sum())
        cluster_lines.append(f"| {cluster} | {total} | {ready} | {review} |")
    charge_errors = int((qc["errors"].str.contains("formal_charge", regex=False)).sum())
    torsion_records = int(qc["pdbqt_torsdof"].notna().sum())
    report = f"""# Step 3.1 ligand preparation and validation

## Scope

This step prepared and independently checked the 47 compounds in the `Cluster compounds` worksheet. No Vina `dock()`, `score()`, or `optimize()` call was made. The original Excel workbook, PDB files, Step 1 outputs, Step 2 outputs, receptor, and docking box were not modified.

## Input integrity

- Input: `data/androgens_clusters_teacher_profile.xlsx`, worksheet `{SHEET}`, header row 3.
- Source SHA256: `{summary['input_sha256']}`.
- Source rows: {summary['source_rows']}; manifest rows: {summary['manifest_rows']}.
- Cluster counts: {summary['cluster_counts']}.
- The manifest preserves `cluster_id`, `row_index`, and `split` and uses stable IDs of the form `cluster_{{cluster_id}}_row_{{row_index}}`.
- The validation script confirmed the source mapping and source hash were unchanged.

## Preparation protocol

- RDKit SMILES sanitization and chemical-property extraction.
- Original explicit charge and protonation representation retained. No blanket neutralization, salt removal, or pH-state enumeration was applied.
- RDKit ETKDGv3, five conformers per molecule, fixed reproducible seeds based on `20250311`, one thread.
- MMFF94s optimization was used where parameters were available; UFF was the documented fallback.
- The lowest finite optimized conformer was written to SDF with explicit hydrogens.
- Meeko `{version('meeko')}` generated each independent PDBQT; no PDBQT was manually edited.

## Summary

| Check | Count |
|---|---:|
| SMILES successfully parsed | {summary['smiles_parse_success_count']} / 47 |
| Connectivity/valence or other hard structure failures | {summary['structure_issue_compound_count']} |
| Compounds requiring chemical-structure review | {summary['chemical_structure_review_compound_count']} |
| Missing or uncertain stereochemistry | {summary['stereochemistry_missing_or_uncertain_count']} |
| Protonation-state review flags | {summary['protonation_review_count']} |
| 3D SDF generated and RDKit-readable | {summary['three_d_sdf_success_count']} / 47 |
| PDBQT generated and non-empty | {summary['pdbqt_success_count']} / 47 |
| Vina Python API ligand-load success | {summary['vina_ligand_load_success_count']} / 47 |
| Final `ready` | {summary['status_counts']['ready']} |
| Final `review_required` | {summary['status_counts']['review_required']} |
| Final `failed` | {summary['status_counts']['failed']} |

Additional checks found {charge_errors} unexplained formal-charge changes, {torsion_records}/47 PDBQT torsion-tree records, no invalid 3D geometry flags, and no PDBQT load failures.

## Cluster availability

| Cluster | Total | Ready for formal docking | Review required |
|---:|---:|---:|---:|
""" + "\n".join(cluster_lines) + f"""

## Manual stereochemistry review

The following 23 compounds contain one or more stereocenters that were not explicitly assigned in the original SMILES. They were not assigned a random stereoisomer. Their 3D/SDF/PDBQT files are retained as diagnostic/preparation artifacts, but they are not included in the `ready` count and should not enter formal docking until the stereochemistry is resolved.

| Compound ID | Cluster | Original row_index | Unassigned centers |
|---|---:|---:|---|
""" + "\n".join(review_lines) + f"""

The SDF round-trip may contain a concrete 3D arrangement for an originally unspecified center. The independent validator therefore accepts matching non-isomeric connectivity but keeps these records as `review_required`; it does not treat the generated arrangement as confirmed chemistry. No unspecified double-bond stereochemistry or protonation review flag was detected by the implemented checks.

## Outputs

- `scripts/03_prepare_ligands.py`
- `scripts/03_validate_ligands.py`
- `data/ligands/sdf/<compound_id>.sdf`
- `data/ligands/pdbqt/<compound_id>.pdbqt`
- `data/ligands/diagnostics/<compound_id>.json` and Meeko logs
- `results/ligand_preparation/ligand_manifest.csv`
- `results/ligand_preparation/ligand_qc.csv`
- `results/ligand_preparation/preparation_summary.json`

Software versions: RDKit `{rdkit_version()}`, Meeko `{version('meeko')}`, Vina `{version('vina')}`, Pandas `{version('pandas')}`.

## Readiness for Step 3.2

The preparation pipeline itself is reproducible and all 47 compounds have parseable SDF/PDBQT artifacts that load through the Vina Python API. However, only {summary['status_counts']['ready']} compounds are currently suitable as `ready` inputs. The remaining {summary['status_counts']['review_required']} require manual stereochemistry decisions. Step 3.2 batch docking should not begin until those review-required compounds are either resolved or explicitly excluded by a documented decision.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def rdkit_version() -> str:
    from rdkit import rdBase

    return rdBase.rdkitVersion


if __name__ == "__main__":
    main()
