"""Independent Gate 3 completeness checks for ensemble batch docking."""

from __future__ import annotations

import json
from collections import Counter

from ensemble_utils import ROOT, read_csv_rows, write_csv


def main() -> None:
    root = ROOT / "results" / "receptor_ensemble" / "batch_docking"
    gate = {}
    rows_out = []
    for pdb_id in ("2AMA", "2AMB"):
        top_path = root / pdb_id / "docking_top_poses.csv"
        all_path = root / pdb_id / "docking_all_poses.csv"
        if not top_path.exists():
            gate[pdb_id] = {"status": "not_run", "reason": "Gate 2 did not pass"}
            rows_out.append({"receptor": pdb_id, "status": "not_run"})
            continue
        top, all_rows = read_csv_rows(top_path), read_csv_rows(all_path)
        cluster_counts = Counter(int(x["cluster_id"]) for x in top)
        pose_counts = Counter(x["compound_id"] for x in all_rows)
        failures = json.loads((root / pdb_id / "docking_failures.json").read_text())
        expected_clusters = {3: 5, 176: 6, 246: 11, 416: 2}
        count_ok = len(top) == 24 and cluster_counts == expected_clusters
        parse_ok = all(float(x["vina_score_kcal_mol"]) == float(x["vina_score_kcal_mol"]) for x in all_rows)
        result = {"receptor": pdb_id, "status": "gate3_pass" if count_ok and not failures and parse_ok else "gate3_failed", "top_pose_count": len(top), "all_pose_count": len(all_rows), "cluster_counts": dict(cluster_counts), "expected_cluster_counts": expected_clusters, "pose_count_distribution": dict(Counter(pose_counts.values())), "failure_count": len(failures), "failures": failures, "all_scores_finite": parse_ok, "note": "Actual pose count is retained; Vina may return fewer than 10 modes when modes fall outside the requested energy range."}
        (root / pdb_id / "gate3_validation.json").write_text(json.dumps(result, indent=2))
        gate[pdb_id] = result
        rows_out.append({"receptor": pdb_id, "status": result["status"], "top_pose_count": len(top), "all_pose_count": len(all_rows), "failure_count": len(failures), "pose_count_distribution": json.dumps(result["pose_count_distribution"])})
    write_csv(root / "gate3_summary.csv", rows_out)
    (ROOT / "results" / "receptor_ensemble" / "gate3_summary.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()

