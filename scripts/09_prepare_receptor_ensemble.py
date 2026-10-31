"""Step 5.1: prepare and align the 2AMA/2AMB AR receptors.

This script starts from the raw PDB files, repairs only missing atoms in
observed residues, and writes independent derived files.  It does not touch
the 2AM9 or earlier-step outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ensemble_utils import (
    ROOT,
    add_hs_preserving_coords,
    align_receptor_to_reference,
    build_ccd_ligand,
    inspect_raw_pdb,
    ligand_box_check,
    package_versions,
    repair_with_pdbfixer,
    run_meeko_receptor,
    transform_mol,
    write_box,
    write_clean_chain_a,
    write_sdf,
)
from prepare_utils import write_box_pdb
from redocking_utils import read_box_config


TARGETS = {
    "2AMA": {"ligand": "DHT", "ligand_number": 1001, "ccd": "DHT"},
    "2AMB": {"ligand": "17H", "ligand_number": 1001, "ccd": "17H"},
}
KEY_RESIDUES = [685, 688, 705, 711, 752, 761, 764, 770, 781, 788, 789, 819, 820, 829, 836, 840, 841, 845, 846, 847, 848, 854, 855, 857, 862, 865, 877, 878, 880, 893, 894]


def sequence_by_number(path: Path) -> dict[int, str]:
    vals: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == "A" and line[12:16].strip() == "CA":
            vals[int(line[22:26])] = line[17:20].strip()
    return vals


def main() -> None:
    box_ref = ROOT / "data" / "prepared" / "2AM9" / "vina_box_repaired.txt"
    ref_pdb = ROOT / "data" / "prepared" / "2AM9" / "receptor_repaired.pdb"
    center, size = read_box_config(box_ref)["center"], read_box_config(box_ref)["size"]
    ref_seq = sequence_by_number(ref_pdb)
    summary = {"versions": package_versions(), "box_source": str(box_ref.relative_to(ROOT)), "targets": {}}

    for pdb_id, info in TARGETS.items():
        raw = ROOT / "data" / "PDB" / f"{pdb_id}.pdb"
        out = ROOT / "data" / "prepared" / pdb_id
        out.mkdir(parents=True, exist_ok=True)
        inspection = inspect_raw_pdb(raw, info["ligand"], info["ligand_number"])
        (out / "raw_inspection.json").write_text(json.dumps(inspection, indent=2))

        clean = out / "receptor_repair_input.pdb"
        clean_info = write_clean_chain_a(raw, clean)
        repaired = out / "receptor_repaired.pdb"
        repair_info = repair_with_pdbfixer(clean, repaired)
        (out / "repair_info.json").write_text(json.dumps(repair_info, indent=2))

        aligned = out / "receptor_aligned.pdb"
        alignment = align_receptor_to_reference(repaired, ref_pdb, aligned, KEY_RESIDUES)
        # Meeko adds hydrogens/charges to the already aligned, repaired protein.
        try:
            meeko_info = run_meeko_receptor(aligned, out / "receptor")
        except Exception as exc:
            failure = {
                "status": "gate1_failed",
                "reason": "Meeko receptor template/bond validation failed",
                "error": str(exc),
                "raw_inspection": inspection,
                "clean_info": clean_info,
                "repair_info": repair_info,
                "alignment": alignment,
                "note": "No allow_bad_res or coordinate modification was applied; the receptor is not eligible for docking under the fixed protocol.",
            }
            (out / "preparation_failure.json").write_text(json.dumps(failure, indent=2))
            summary["targets"][pdb_id] = failure
            print(json.dumps({"target": pdb_id, "status": "gate1_failed", "error": str(exc)}, indent=2))
            continue

        ccd = out / f"{info['ccd']}.cif"
        crystal, ligand_meta = build_ccd_ligand(raw, ccd, info["ligand"], info["ligand_number"], pdb_id)
        prepared = add_hs_preserving_coords(crystal, f"{info['ligand']}_{pdb_id}_prepared")
        crystal_aligned = transform_mol(crystal, np.asarray(alignment["rotation"]), np.asarray(alignment["translation"]), f"{info['ligand']}_{pdb_id}_crystal_aligned")
        prepared_aligned = transform_mol(prepared, np.asarray(alignment["rotation"]), np.asarray(alignment["translation"]), f"{info['ligand']}_{pdb_id}_prepared_aligned")
        write_sdf(crystal, out / "native_crystal.sdf")
        write_sdf(prepared, out / "native_prepared.sdf")
        write_sdf(crystal_aligned, out / "native_crystal_aligned.sdf")
        write_sdf(prepared_aligned, out / "native_prepared_aligned.sdf")

        # Meeko ligand preparation is intentionally run on the aligned SDF so
        # docked poses and the fixed 2AM9 box share one coordinate system.
        import shutil
        import subprocess
        import sys
        exe = Path(sys.executable).resolve().parent / "mk_prepare_ligand.py"
        if not exe.exists():
            exe = Path(shutil.which("mk_prepare_ligand.py") or "")
        native_pdbqt = out / "native_aligned.pdbqt"
        cmd = [str(exe), "-i", str(out / "native_prepared_aligned.sdf"), "-o", str(native_pdbqt), "--charge_model", "gasteiger", "--add_index_map"]
        result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
        (out / "meeko_ligand.log").write_text(result.stdout + "\n" + result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Meeko ligand preparation failed for {pdb_id}: {result.stdout}\n{result.stderr}")

        box = out / "vina_box_ensemble.txt"
        write_box(box, "receptor.pdbqt", center, size, {"coordinate_system": "2AM9 repaired receptor frame", "native_ligand": info["ligand"]})
        write_box_pdb(out / "vina_box_ensemble.pdb", center, size)

        seq = sequence_by_number(repaired)
        mismatches = [{"resseq": n, "reference": ref_seq[n], "target": seq[n]} for n in sorted(set(ref_seq) & set(seq)) if ref_seq[n] != seq[n]]
        ligand_box = ligand_box_check(crystal_aligned, center, size)
        target = {
            "raw_inspection": inspection,
            "clean_info": clean_info,
            "repair_info": repair_info,
            "meeko_receptor": meeko_info,
            "ligand": ligand_meta,
            "ligand_box": ligand_box,
            "alignment": alignment,
            "sequence_common_residue_mismatches": mismatches,
            "key_residues_present": {str(n): n in seq for n in KEY_RESIDUES},
            "files": {key: str((out / name).relative_to(ROOT)) for key, name in {
                "receptor_repaired": "receptor_repaired.pdb",
                "receptor_aligned": "receptor_aligned.pdb",
                "receptor_pdbqt": "receptor.pdbqt",
                "native_crystal": "native_crystal.sdf",
                "native_prepared": "native_prepared.sdf",
                "native_crystal_aligned": "native_crystal_aligned.sdf",
                "native_prepared_aligned": "native_prepared_aligned.sdf",
                "native_pdbqt": "native_aligned.pdbqt",
                "box": "vina_box_ensemble.txt",
            }.items()},
        }
        (out / "preparation_summary.json").write_text(json.dumps(target, indent=2))
        summary["targets"][pdb_id] = target
        print(json.dumps({"target": pdb_id, "formula": ligand_meta["formula"], "heavy_atoms": ligand_meta["heavy_atoms"], "alignment_rmsd_A": alignment["backbone_rmsd_A"], "native_inside_box": ligand_box["all_heavy_atoms_inside"], "sequence_mismatches": len(mismatches)}, indent=2))

    (ROOT / "results" / "receptor_ensemble").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "receptor_ensemble" / "preparation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

