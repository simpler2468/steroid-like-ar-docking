"""Independent Gate 2 validator for Step 5 native redocking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ensemble_utils import ROOT, load_single_sdf, read_csv_rows, write_csv
from redocking_utils import top_pose_interactions


SEEDS = [42, 123, 2026, 7, 99]


def main() -> None:
    root = ROOT / "results" / "receptor_ensemble" / "native_redocking"
    gate_rows = []
    targets = {}
    for pdb_id in ("2AMA", "2AMB"):
        prep = ROOT / "data" / "prepared" / pdb_id
        summary_path = root / pdb_id / "native_redocking_summary.json"
        if not summary_path.exists():
            targets[pdb_id] = {"status": "not_run", "reason": "Gate 1 did not produce an eligible receptor"}
            gate_rows.append({"receptor": pdb_id, "status": "not_run", "top_rmsd_success_count": "", "top_rmsd_mean_A": "", "top_rmsd_sd_A": "", "top_score_mean": "", "min_pose_rmsd_min_A": ""})
            continue
        summary = json.loads(summary_path.read_text())
        rows = read_csv_rows(root / pdb_id / "native_redocking_all_poses.csv")
        top = [r for r in rows if int(r["pose"]) == 1]
        top.sort(key=lambda r: SEEDS.index(int(r["seed"])))
        good = [float(r["crystal_rmsd_A"]) <= 2.0 for r in top]
        min_all = min(float(r["crystal_rmsd_A"]) for r in rows)
        top_scores = np.array([float(r["vina_score_kcal_mol"]) for r in top])
        top_rmsd = np.array([float(r["crystal_rmsd_A"]) for r in top])
        interactions = []
        for row in top:
            pred = load_single_sdf(ROOT / row["sdf"])
            crystal = load_single_sdf(prep / "native_crystal_aligned.sdf")
            # Reuse the graph mapping recorded by the independent RMSD pass.
            mapping = tuple(int(x) for x in json.loads(row["mapping"].replace("'", '"')) ) if row.get("mapping") else None
            # The CSV intentionally stores only scalar validation fields; for
            # interaction inspection use the unique graph mapping directly.
            from ensemble_utils import crystal_rmsd
            mapping = tuple(crystal_rmsd(crystal, pred)["mapping"])
            interactions.append({"seed": int(row["seed"]), **top_pose_interactions(prep / "receptor_aligned.pdb", crystal, pred, mapping)})
        passed = len(top) == 5 and sum(good) >= 4
        result = {
            "receptor": pdb_id,
            "status": "gate2_pass" if passed else "gate2_failed",
            "criterion": "top-scoring crystal RMSD <= 2 A in at least 4/5 seeds",
            "top_runs": [{"seed": int(r["seed"]), "score_kcal_mol": float(r["vina_score_kcal_mol"]), "top_crystal_rmsd_A": float(r["crystal_rmsd_A"]), "minimum_all_pose_crystal_rmsd_A": min(float(x["crystal_rmsd_A"]) for x in rows if int(x["seed"]) == int(r["seed"]))} for r in top],
            "top_score_mean": float(np.mean(top_scores)),
            "top_score_sd": float(np.std(top_scores, ddof=1)),
            "top_score_min": float(np.min(top_scores)),
            "top_score_max": float(np.max(top_scores)),
            "top_rmsd_mean_A": float(np.mean(top_rmsd)),
            "top_rmsd_sd_A": float(np.std(top_rmsd, ddof=1)),
            "top_rmsd_min_A": float(np.min(top_rmsd)),
            "top_rmsd_max_A": float(np.max(top_rmsd)),
            "top_rmsd_le_2A_count": int(sum(good)),
            "all_pose_minimum_crystal_rmsd_A": min_all,
            "interactions": interactions,
            "protocol": summary["protocol"],
        }
        (root / pdb_id / "gate2_validation.json").write_text(json.dumps(result, indent=2))
        targets[pdb_id] = result
        gate_rows.append({"receptor": pdb_id, "status": result["status"], "top_rmsd_success_count": result["top_rmsd_le_2A_count"], "top_rmsd_mean_A": result["top_rmsd_mean_A"], "top_rmsd_sd_A": result["top_rmsd_sd_A"], "top_score_mean": result["top_score_mean"], "min_pose_rmsd_min_A": result["all_pose_minimum_crystal_rmsd_A"]})
    write_csv(root / "gate2_summary.csv", gate_rows)
    overall = {"targets": targets, "passed_targets": [x for x, value in targets.items() if value.get("status") == "gate2_pass"], "failed_or_skipped_targets": [x for x, value in targets.items() if value.get("status") != "gate2_pass"]}
    (root / "gate2_summary.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps({"passed": overall["passed_targets"], "failed_or_skipped": overall["failed_or_skipped_targets"]}, indent=2))


if __name__ == "__main__":
    main()

