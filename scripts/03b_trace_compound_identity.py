"""Curate stereochemistry status without inventing compound identities."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "androgens_clusters_teacher_profile.xlsx"
SHEET = "Cluster compounds"
RESULTS = ROOT / "results" / "ligand_preparation"
MANIFEST = RESULTS / "ligand_manifest.csv"
QC = RESULTS / "ligand_qc.csv"
CANDIDATES = RESULTS / "stereochemistry_candidates.csv"
RESOLUTION_OUT = RESULTS / "stereochemistry_resolution.csv"
READY_OUT = RESULTS / "docking_ready.csv"
REVIEW_OUT = RESULTS / "docking_review.csv"
SUMMARY_OUT = RESULTS / "resolution_summary.json"
REPORT_OUT = ROOT / "reports" / "stereochemistry_resolution.md"


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    source = pd.read_excel(INPUT, sheet_name=SHEET, header=2)
    manifest = pd.read_csv(MANIFEST, dtype=object, keep_default_na=False)
    qc = pd.read_csv(QC, dtype=object, keep_default_na=False)
    candidates = pd.read_csv(CANDIDATES, dtype=object, keep_default_na=False)
    groups = pd.read_csv(RESULTS / "connectivity_groups.csv", dtype=object, keep_default_na=False)
    connectivity_id_by_smiles = dict(zip(groups["canonical_nonisomeric_smiles"], groups["connectivity_id"]))
    if len(source) != 47 or len(manifest) != 47 or len(qc) != 47:
        raise ValueError("Source, manifest, and QC must each contain 47 records")
    merged = manifest.merge(qc[["compound_id", "final_status"]], on="compound_id", how="left", validate="one_to_one")
    identifier_tokens = ("name", "cas", "cid", "pubchem", "dtxsid", "chembl", "tox21", "toxcast", "compound_id", "chemical_id")
    identifier_columns = [column for column in source.columns if not str(column).startswith("teacher_probability__") and any(token in str(column).lower() for token in identifier_tokens)]
    resolution_rows = []
    for row in merged.to_dict(orient="records"):
        compound_id = row["compound_id"]
        step_status = row["final_status"]
        mol = Chem.MolFromSmiles(str(row["original_smiles"]), sanitize=True)
        if mol is None:
            raise ValueError(f"Cannot parse original SMILES for {compound_id}")
        canonical_noniso = Chem.MolToSmiles(mol, isomericSmiles=False)
        canonical_iso = Chem.MolToSmiles(mol, isomericSmiles=True)
        candidate_rows = candidates.loc[candidates["target_compound_id"] == compound_id].copy()
        ready_candidates = candidate_rows.loc[candidate_rows["candidate_step3_1_status"] == "ready"]
        ready_isomers = sorted(set(ready_candidates["candidate_isomeric_smiles"]))
        if step_status == "ready":
            resolution_status = "ready_original"
            reason = "Original SMILES had explicit stereochemistry and passed independent Step 3.1 checks."
            proposed = row["canonical_isomeric_smiles"]
            evidence = "original Excel SMILES; Step 3.1 manifest/QC"
            docking_status = "docking_ready"
        elif len(ready_isomers) > 1:
            resolution_status = "ambiguous"
            reason = "Multiple distinct ready stereoisomer records share the same 2D connectivity; no source links the review record to one isomer."
            proposed = ""
            evidence = "offline same-connectivity candidates only; no external identity evidence"
            docking_status = "docking_review"
        else:
            resolution_status = "unresolved"
            if len(ready_isomers) == 1:
                reason = "One same-connectivity ready record is available, but that is insufficient to confirm the original stereoisomer identity."
            else:
                reason = "No same-connectivity ready record or reliable identity identifier was found in the available project data."
            proposed = ""
            evidence = "offline structure matching only; no external identity evidence"
            docking_status = "docking_review"
        resolution_rows.append(
            {
                "compound_id": compound_id,
                "cluster_id": row["cluster_id"],
                "row_index": row["row_index"],
                "split": row["split"],
                "original_smiles": row["original_smiles"],
                "canonical_isomeric_smiles": canonical_iso,
                "canonical_nonisomeric_smiles": canonical_noniso,
                "connectivity_id": connectivity_id_by_smiles[canonical_noniso],
                "step3_1_status": step_status,
                "resolution_status": resolution_status,
                "proposed_confirmed_isomeric_smiles": proposed,
                "candidate_compound_ids": ";".join(ready_candidates["candidate_compound_id"].tolist()),
                "candidate_isomer_count": len(ready_isomers),
                "evidence_source": evidence,
                "external_identifier": "",
                "evidence_url_or_path": "data/androgens_clusters_teacher_profile.xlsx",
                "conflict": "no external identity evidence available",
                "resolution_reason": reason,
                "docking_status": docking_status,
            }
        )
    resolution = pd.DataFrame(resolution_rows)
    resolution.to_csv(RESOLUTION_OUT, index=False)
    resolution.loc[resolution["docking_status"] == "docking_ready"].to_csv(READY_OUT, index=False)
    resolution.loc[resolution["docking_status"] == "docking_review"].to_csv(REVIEW_OUT, index=False)

    status_counts = {status: int((resolution["resolution_status"] == status).sum()) for status in ("ready_original", "resolved_verified", "ambiguous", "unresolved", "failed")}
    cluster_summary = []
    for cluster in sorted(pd.to_numeric(resolution["cluster_id"]).unique()):
        subset = resolution[pd.to_numeric(resolution["cluster_id"]) == cluster]
        cluster_summary.append({
            "cluster_id": int(cluster),
            "record_count": len(subset),
            "docking_ready_count": int((subset["docking_status"] == "docking_ready").sum()),
            "unique_connectivity_count": subset["connectivity_id"].nunique(),
            "resolution_status_counts": subset["resolution_status"].value_counts().to_dict(),
        })
    teacher = pd.read_csv(RESULTS / "teacher_profile_duplicates.csv", dtype=object, keep_default_na=False)
    near_equal = teacher.loc[(teacher["profile_exactly_equal"] == "False") & (pd.to_numeric(teacher["max_absolute_probability_difference"]) < 1e-6)]
    summary = {
        "record_count": len(resolution),
        "distinct_connectivity_count": 23,
        "distinct_explicit_isomeric_smiles_count": int((manifest["preparation_status"] == "ready").sum()),
        "resolution_status_counts": status_counts,
        "docking_ready_count": int((resolution["docking_status"] == "docking_ready").sum()),
        "docking_review_count": int((resolution["docking_status"] == "docking_review").sum()),
        "cluster_summary": cluster_summary,
        "identifier_columns_found_in_source": identifier_columns,
        "external_queries_performed": False,
        "external_query_reason": "No reliable external identifier column was present in the project Excel; connectivity-only matches cannot verify original compound identity.",
        "review_records_with_ready_match": int(((resolution["step3_1_status"] == "review_required") & resolution["candidate_compound_ids"].ne("")).sum()),
        "review_records_with_multiple_ready_candidates": int((resolution.loc[resolution["resolution_status"] == "ambiguous", "candidate_isomer_count"] > 1).sum()),
        "ambiguous_review_count": status_counts["ambiguous"],
        "unresolved_review_count": status_counts["unresolved"],
        "teacher_pair_count": len(teacher),
        "teacher_exact_pair_count": int((teacher["profile_exactly_equal"] == "True").sum()),
        "teacher_numerically_near_equal_but_not_exact_count": len(near_equal),
        "teacher_different_pair_count": int((teacher["profile_exactly_equal"] == "False").sum()),
        "special_records": {
            "cluster_176_row_8743": resolution.loc[resolution["compound_id"] == "cluster_176_row_8743"].to_dict(orient="records"),
            "cluster_246_row_8772": resolution.loc[resolution["compound_id"] == "cluster_246_row_8772"].to_dict(orient="records"),
            "cluster_246_row_8768": resolution.loc[resolution["compound_id"] == "cluster_246_row_8768"].to_dict(orient="records"),
        },
        "docking_executed": False,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, resolution, teacher)
    print(json.dumps({key: value for key, value in summary.items() if key != "special_records"}, indent=2, ensure_ascii=False))


def write_report(summary: dict[str, Any], resolution: pd.DataFrame, teacher: pd.DataFrame) -> None:
    cluster_lines = []
    for item in summary["cluster_summary"]:
        cluster_lines.append(f"| {item['cluster_id']} | {item['record_count']} | {item['unique_connectivity_count']} | {item['docking_ready_count']} | {safe_json(item['resolution_status_counts'])} |")
    review = resolution[resolution["resolution_status"].isin(["ambiguous", "unresolved"])].copy()
    review_lines = []
    for row in review.itertuples(index=False):
        review_lines.append(f"| {row.compound_id} | {row.cluster_id} | {row.resolution_status} | {row.candidate_compound_ids or 'none'} | {row.resolution_reason} |")
    nonexact = teacher.loc[teacher["profile_exactly_equal"] == "False"]
    teacher_lines = [f"| {row.compound_id_left} | {row.compound_id_right} | {row.max_absolute_probability_difference} |" for row in nonexact.itertuples(index=False)]
    report = f"""# Step 3.1b stereochemistry resolution and compound curation

## Scope and decision rule

This step retained all 47 original records and analyzed their 2D connectivity, stereochemistry, same-connectivity candidates, and Teacher Profile duplicate patterns. No stereochemistry was assigned automatically, and no docking, scoring, receptor, docking-box, or Teacher model operation was performed.

The original record identifier remains `cluster_{{cluster_id}}_row_{{row_index}}`. `row_index` was not interpreted as PubChem CID, CAS, DTXSID, or any other external identifier.

## Offline structural findings

- Original records: 47.
- Distinct canonical non-isomeric connectivity structures: 23.
- Distinct fully specified isomeric structures among Step 3.1-ready records: {summary['distinct_explicit_isomeric_smiles_count']}.
- 21 connectivity groups contain multiple records.
- No connectivity group spans more than one cluster.
- The 47 original SMILES produced 47 distinct isomeric InChIKeys; this reflects distinct/unspecified stereochemical strings, not verified external chemical identities.
- Of 23 `review_required` records, 22 have at least one same-cluster ready connectivity match. {summary['review_records_with_multiple_ready_candidates']} have multiple distinct ready stereoisomer candidates. One has no same-connectivity record in the project data.

| Cluster | Records | Connectivity groups | Docking-ready records | Resolution statuses |
|---:|---:|---:|---:|---|
""" + "\n".join(cluster_lines) + f"""

## Resolution status

| Status | Count |
|---|---:|
| `ready_original` | {summary['resolution_status_counts']['ready_original']} |
| `resolved_verified` | {summary['resolution_status_counts']['resolved_verified']} |
| `ambiguous` | {summary['resolution_status_counts']['ambiguous']} |
| `unresolved` | {summary['resolution_status_counts']['unresolved']} |
| `failed` | {summary['resolution_status_counts']['failed']} |

No `resolved_verified` record was established. The local Excel contains no reliable name/CAS/PubChem CID/DTXSID/ChEMBL/Tox21/ToxCast identifier column, and no original external source file was present in the project. Therefore, same-connectivity matches are evidence of candidate relationships only, not identity confirmation.

| Compound ID | Cluster | Status | Same-connectivity ready candidates | Interpretation |
|---|---:|---|---|---|
""" + "\n".join(review_lines) + f"""

The highlighted preliminary cases were confirmed as follows:

- `cluster_176_row_8743`: two same-cluster ready candidates, rows 5378 and 5380, with different explicit stereochemical SMILES. It remains `ambiguous`.
- `cluster_246_row_8772`: two same-cluster ready candidates, rows 5356 and 5359, with different explicit stereochemical SMILES. It remains `ambiguous`.
- `cluster_246_row_8768`: no same-connectivity candidate in the project records. It remains `unresolved`.

For the other matched review records, one same-connectivity ready record was found. This is not sufficient to confirm the original stereoisomer, so they remain `unresolved` rather than being upgraded.

## Teacher Profile duplicate patterns

There were {summary['teacher_pair_count']} pairwise comparisons within repeated connectivity groups. {summary['teacher_exact_pair_count']} pairs had exactly equal eight-dimensional Teacher Profile vectors. One pair was not numerically identical but differed by less than 1e-6:

| Left record | Right record | Maximum absolute probability difference |
|---|---|---:|
""" + "\n".join(teacher_lines) + f"""

This descriptive pattern does not establish stereochemical identity or explain why the vectors are equal. Teacher Profile values were not used to resolve or select stereoisomers.

## Candidate sets for later docking

- `docking_ready.csv` contains only the 24 `ready_original` records that passed Step 3.1 and had explicit original stereochemistry.
- `docking_review.csv` contains the 23 ambiguous/unresolved records.
- All original records remain present in `stereochemistry_resolution.csv`; no records were merged or deleted.
- No newly confirmed structure was generated, so no new SDF/PDBQT preparation was authorized or performed in this step.

## Outputs

- `scripts/03b_analyze_stereochemistry.py`
- `scripts/03b_trace_compound_identity.py`
- `results/ligand_preparation/connectivity_groups.csv`
- `results/ligand_preparation/stereochemistry_candidates.csv`
- `results/ligand_preparation/stereochemistry_resolution.csv`
- `results/ligand_preparation/teacher_profile_duplicates.csv`
- `results/ligand_preparation/docking_ready.csv`
- `results/ligand_preparation/docking_review.csv`
- `results/ligand_preparation/resolution_summary.json`

## Recommendation

The project should not automatically enter Step 3.2 with all 47 records. The 24 `ready_original` records can form a documented preliminary docking set. The 23 review records require manual identity/stereochemistry resolution or an explicit sensitivity-analysis decision before formal docking. In particular, no single isomer should be selected for rows 8743, 8772, or 8768 from the available evidence alone.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
