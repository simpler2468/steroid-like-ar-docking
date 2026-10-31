"""Offline connectivity, stereochemistry-candidate, and teacher-profile analysis."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "androgens_clusters_teacher_profile.xlsx"
SHEET = "Cluster compounds"
RESULTS = ROOT / "results" / "ligand_preparation"
MANIFEST = RESULTS / "ligand_manifest.csv"
QC = RESULTS / "ligand_qc.csv"
GROUPS_OUT = RESULTS / "connectivity_groups.csv"
CANDIDATES_OUT = RESULTS / "stereochemistry_candidates.csv"
TEACHER_OUT = RESULTS / "teacher_profile_duplicates.csv"
SUMMARY_OUT = RESULTS / "offline_stereochemistry_summary.json"


def read_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    return mol


def stereo_details(mol: Chem.Mol) -> dict[str, Any]:
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    assigned = [f"atom_{idx}_{label}" for idx, label in centers if label != "?"]
    unassigned = [f"atom_{idx}_{mol.GetAtomWithIdx(idx).GetSymbol()}" for idx, label in centers if label == "?"]
    potential_double = []
    explicit_double = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = str(bond.GetStereo())
        item = {"bond": bond.GetIdx(), "begin": bond.GetBeginAtomIdx(), "end": bond.GetEndAtomIdx(), "stereo": stereo}
        if stereo in {"STEREOE", "STEREOZ"}:
            explicit_double.append(item)
        elif stereo == "STEREONONE":
            potential_double.append(item)
    for info in Chem.FindPotentialStereo(mol):
        if "Bond_Double" in str(info.type) and "Unspecified" in str(info.specified):
            if not any(item["bond"] == info.idx for item in potential_double):
                potential_double.append({"bond": info.idx, "potential_stereo": "unspecified"})
    return {
        "assigned_centers": assigned,
        "unassigned_centers": unassigned,
        "assigned_center_count": len(assigned),
        "unassigned_center_count": len(unassigned),
        "explicit_double_bonds": explicit_double,
        "unspecified_double_bonds": potential_double,
    }


def inchi_key(mol: Chem.Mol) -> str:
    try:
        value = Chem.MolToInchiKey(mol)
        return value or ""
    except Exception:
        return ""


def stereo_compatibility(target: Chem.Mol, candidate: Chem.Mol) -> tuple[bool, str]:
    target_no_stereo = Chem.MolFromSmiles(Chem.MolToSmiles(target, isomericSmiles=False))
    matches = candidate.GetSubstructMatches(target_no_stereo, useChirality=False, uniquify=False)
    if not matches:
        return False, "no_connectivity_atom_mapping"
    Chem.AssignStereochemistry(target, force=True, cleanIt=True)
    Chem.AssignStereochemistry(candidate, force=True, cleanIt=True)
    target_centers = Chem.FindMolChiralCenters(target, includeUnassigned=True, useLegacyImplementation=False)
    for mapping in matches:
        compatible = True
        compared = 0
        for target_idx, label in target_centers:
            if label == "?":
                continue
            compared += 1
            candidate_idx = mapping[target_idx]
            candidate_atom = candidate.GetAtomWithIdx(candidate_idx)
            if not candidate_atom.HasProp("_CIPCode") or candidate_atom.GetProp("_CIPCode") != label:
                compatible = False
                break
        if compatible:
            return True, "assigned_stereocenters_compatible" if compared else "target_has_no_assigned_stereocenters"
    return False, "assigned_stereocenters_conflict"


def main() -> None:
    manifest = pd.read_csv(MANIFEST, dtype=object, keep_default_na=False)
    qc = pd.read_csv(QC, dtype=object, keep_default_na=False)
    source = pd.read_excel(INPUT, sheet_name=SHEET, header=2)
    if len(source) != 47 or len(manifest) != 47 or len(qc) != 47:
        raise ValueError(f"Expected 47 rows in source/manifest/qc: {len(source)}/{len(manifest)}/{len(qc)}")
    merged = manifest.merge(qc[["compound_id", "final_status"]], on="compound_id", how="left", validate="one_to_one")
    records: list[dict[str, Any]] = []
    mols: dict[str, Chem.Mol] = {}
    for row in merged.to_dict(orient="records"):
        mol = read_mol(row["original_smiles"])
        mols[row["compound_id"]] = mol
        details = stereo_details(mol)
        conn = Chem.MolToSmiles(mol, isomericSmiles=False)
        iso = Chem.MolToSmiles(mol, isomericSmiles=True)
        record = {
            "compound_id": row["compound_id"],
            "cluster_id": row["cluster_id"],
            "row_index": row["row_index"],
            "split": row["split"],
            "original_smiles": row["original_smiles"],
            "canonical_isomeric_smiles": iso,
            "canonical_nonisomeric_smiles": conn,
            "inchi_key": inchi_key(mol),
            "connectivity_id": "",
            "final_step3_1_status": row["final_status"],
            "formal_charge": int(Chem.GetFormalCharge(mol)),
            "component_count": len(Chem.GetMolFrags(mol, asMols=False)),
            "assigned_chiral_centers": ";".join(details["assigned_centers"]),
            "unassigned_chiral_centers": ";".join(details["unassigned_centers"]),
            "assigned_chiral_center_count": details["assigned_center_count"],
            "unassigned_chiral_center_count": details["unassigned_center_count"],
            "explicit_double_bond_stereo": json.dumps(details["explicit_double_bonds"], ensure_ascii=False),
            "unspecified_double_bond_stereo": json.dumps(details["unspecified_double_bonds"], ensure_ascii=False),
        }
        records.append(record)
    connectivity_keys = sorted({record["canonical_nonisomeric_smiles"] for record in records})
    conn_ids = {key: f"conn_{idx:03d}" for idx, key in enumerate(connectivity_keys, 1)}
    for record in records:
        record["connectivity_id"] = conn_ids[record["canonical_nonisomeric_smiles"]]
    record_by_id = {record["compound_id"]: record for record in records}
    groups = []
    for conn, group in pd.DataFrame(records).groupby("canonical_nonisomeric_smiles", sort=True):
        group_records = group.to_dict(orient="records")
        groups.append({
            "connectivity_id": conn_ids[conn],
            "canonical_nonisomeric_smiles": conn,
            "inchi_key_nonstereo": inchi_key(Chem.MolFromSmiles(conn)),
            "record_count": len(group_records),
            "compound_ids": ";".join(item["compound_id"] for item in group_records),
            "clusters": ";".join(sorted({str(item["cluster_id"]) for item in group_records})),
            "cluster_count": len({str(item["cluster_id"]) for item in group_records}),
            "ready_count": sum(item["final_step3_1_status"] == "ready" for item in group_records),
            "review_count": sum(item["final_step3_1_status"] == "review_required" for item in group_records),
            "distinct_isomeric_smiles": len({item["canonical_isomeric_smiles"] for item in group_records}),
            "distinct_inchi_keys": len({item["inchi_key"] for item in group_records}),
        })
    group_df = pd.DataFrame(groups).sort_values("connectivity_id")
    group_df.to_csv(GROUPS_OUT, index=False)

    candidate_rows = []
    for target in records:
        if target["final_step3_1_status"] != "review_required":
            continue
        same_conn = [candidate for candidate in records if candidate["canonical_nonisomeric_smiles"] == target["canonical_nonisomeric_smiles"] and candidate["compound_id"] != target["compound_id"]]
        for candidate in sorted(same_conn, key=lambda item: (str(item["cluster_id"]) != str(target["cluster_id"]), item["compound_id"])):
            compatible, relation = stereo_compatibility(mols[target["compound_id"]], mols[candidate["compound_id"]])
            candidate_rows.append({
                "target_compound_id": target["compound_id"],
                "target_cluster_id": target["cluster_id"],
                "target_row_index": target["row_index"],
                "candidate_compound_id": candidate["compound_id"],
                "candidate_cluster_id": candidate["cluster_id"],
                "candidate_row_index": candidate["row_index"],
                "same_cluster": str(target["cluster_id"]) == str(candidate["cluster_id"]),
                "candidate_step3_1_status": candidate["final_step3_1_status"],
                "target_isomeric_smiles": target["canonical_isomeric_smiles"],
                "candidate_isomeric_smiles": candidate["canonical_isomeric_smiles"],
                "same_isomeric_smiles": target["canonical_isomeric_smiles"] == candidate["canonical_isomeric_smiles"],
                "assigned_stereo_compatible": compatible,
                "stereo_relation": relation,
            })
    pd.DataFrame(candidate_rows).to_csv(CANDIDATES_OUT, index=False)

    source_teacher_cols = [column for column in source.columns if str(column).startswith("teacher_probability__")]
    teacher_rows = []
    for conn, group in pd.DataFrame(records).groupby("canonical_nonisomeric_smiles", sort=True):
        ids = group["compound_id"].tolist()
        for left_id, right_id in itertools.combinations(ids, 2):
            left = source.loc[source["row_index"].astype(str) == str(record_by_id[left_id]["row_index"]), source_teacher_cols]
            right = source.loc[source["row_index"].astype(str) == str(record_by_id[right_id]["row_index"]), source_teacher_cols]
            left_values = pd.to_numeric(left.iloc[0], errors="coerce").to_numpy(dtype=float)
            right_values = pd.to_numeric(right.iloc[0], errors="coerce").to_numpy(dtype=float)
            equal = bool(np.all((np.isnan(left_values) & np.isnan(right_values)) | np.isclose(left_values, right_values, equal_nan=True, atol=1e-12, rtol=0)))
            finite_diff = np.abs(left_values - right_values)
            max_diff = float(np.nanmax(finite_diff)) if np.isfinite(finite_diff).any() else 0.0
            teacher_rows.append({
                "connectivity_id": conn_ids[conn],
                "compound_id_left": left_id,
                "compound_id_right": right_id,
                "cluster_left": record_by_id[left_id]["cluster_id"],
                "cluster_right": record_by_id[right_id]["cluster_id"],
                "same_cluster": record_by_id[left_id]["cluster_id"] == record_by_id[right_id]["cluster_id"],
                "profile_exactly_equal": equal,
                "max_absolute_probability_difference": max_diff,
                "left_status": record_by_id[left_id]["final_step3_1_status"],
                "right_status": record_by_id[right_id]["final_step3_1_status"],
            })
    pd.DataFrame(teacher_rows).to_csv(TEACHER_OUT, index=False)

    review_ids = {record["compound_id"] for record in records if record["final_step3_1_status"] == "review_required"}
    candidate_df = pd.DataFrame(candidate_rows)
    review_with_ready = set(candidate_df.loc[candidate_df["candidate_step3_1_status"] == "ready", "target_compound_id"]) if not candidate_df.empty else set()
    summary = {
        "record_count": len(records),
        "distinct_connectivity_count": len(connectivity_keys),
        "distinct_isomeric_smiles_count": len({record["canonical_isomeric_smiles"] for record in records}),
        "distinct_inchikey_count": len({record["inchi_key"] for record in records if record["inchi_key"]}),
        "connectivity_groups_with_multiple_records": int((group_df["record_count"] > 1).sum()),
        "connectivity_groups_present_in_multiple_clusters": int((group_df["cluster_count"] > 1).sum()),
        "review_record_count": len(review_ids),
        "review_records_with_any_ready_connectivity_match": len(review_with_ready),
        "review_records_with_same_cluster_ready_match": int(sum(target in set(candidate_df.loc[(candidate_df["candidate_step3_1_status"] == "ready") & candidate_df["same_cluster"], "target_compound_id"]) for target in review_ids)) if not candidate_df.empty else 0,
        "teacher_pair_count": len(teacher_rows),
        "teacher_profile_exact_pair_count": int(sum(row["profile_exactly_equal"] for row in teacher_rows)),
        "teacher_profile_different_pair_count": int(sum(not row["profile_exactly_equal"] for row in teacher_rows)),
        "special_records": {
            "cluster_176_row_8743": [row for row in candidate_rows if row["target_compound_id"] == "cluster_176_row_8743"],
            "cluster_246_row_8772": [row for row in candidate_rows if row["target_compound_id"] == "cluster_246_row_8772"],
            "cluster_246_row_8768": [row for row in candidate_rows if row["target_compound_id"] == "cluster_246_row_8768"],
        },
        "no_external_identity_claims": True,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "special_records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
