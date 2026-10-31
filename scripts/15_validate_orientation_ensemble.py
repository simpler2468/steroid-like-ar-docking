"""Independent validation for Step 5.4 multi-seed orientation outputs."""

from __future__ import annotations

import json
from collections import Counter

from ensemble_utils import ROOT, read_csv_rows, write_csv


SEEDS = {42, 123, 2026, 7, 99, 11, 17, 73, 101, 314}
TARGETS = {"cluster_416_row_5373", "cluster_176_row_705", "cluster_176_row_5378", "cluster_3_row_2412", "cluster_246_row_5379"}


def main() -> None:
    root = ROOT / "results" / "receptor_ensemble" / "multiseed_orientation"
    out = {}
    for pdb_id in ("2AMA", "2AMB"):
        p = root / pdb_id
        if not (p / "orientation_top_poses.csv").exists():
            out[pdb_id] = {"status": "not_run", "reason": "Gate 2 did not pass"}
            continue
        top = read_csv_rows(p / "orientation_top_poses.csv")
        all_rows = read_csv_rows(p / "orientation_all_poses.csv")
        meta = json.loads((p / "run_metadata.json").read_text())
        seed_pairs = {(r["compound_id"], int(r["seed"])) for r in top}
        mapping_ok = all(value.get("status") == "validated" for value in meta.get("mapping", {}).values())
        result = {"receptor": pdb_id, "status": "pass" if len(top) == 50 and len(seed_pairs) == 50 and TARGETS == {r["compound_id"] for r in top} and SEEDS == {int(r["seed"]) for r in top} and mapping_ok else "failed", "top_pose_count": len(top), "all_pose_count": len(all_rows), "top_compound_count": len({r["compound_id"] for r in top}), "top_seed_count": len({int(r["seed"]) for r in top}), "pose_count_distribution": dict(Counter(sum(1 for x in all_rows if x["compound_id"] == cid and int(x["seed"]) == seed) for cid, seed in seed_pairs)), "mapping_status": {k: v.get("status") for k, v in meta.get("mapping", {}).items()}, "seed42_reuse_count": sum(r.get("seed_source") == "reused_gate3_seed42_exact_protocol" for r in top)}
        (p / "orientation_validation.json").write_text(json.dumps(result, indent=2))
        out[pdb_id] = result
    (root / "validation_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

