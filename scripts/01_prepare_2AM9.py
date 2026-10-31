"""Prepare AR chain A and the experimental TES ligand from PDB 2AM9.

Run from the project root with the steroid_docking environment active.
This script never edits the original PDB or any input spreadsheet.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rdkit import Chem

from prepare_utils import (
    build_tes_mol,
    inspect_2am9,
    read_vina_box,
    read_tes_from_pdb,
    repair_receptor_with_pdbfixer,
    run_command,
    validate_repaired_outputs,
    write_box_pdb,
    write_clean_receptor,
)


ROOT = Path(__file__).resolve().parents[1]
PDB_PATH = ROOT / "data" / "PDB" / "2AM9.pdb"
OUT = ROOT / "data" / "prepared" / "2AM9"
REPORT = ROOT / "reports" / "2AM9_preparation.md"


def main() -> None:
    if not PDB_PATH.exists():
        raise FileNotFoundError(PDB_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    legacy_report = REPORT.with_name("2AM9_preparation_legacy.md")
    if REPORT.exists() and not legacy_report.exists():
        shutil.copy2(REPORT, legacy_report)

    tes_residue, tes_coords = read_tes_from_pdb(PDB_PATH)
    inspection = inspect_2am9(PDB_PATH, tes_coords)
    (OUT / "2AM9_inspection.json").write_text(json.dumps(inspection, indent=2))

    ccd_path = OUT / "TES.cif"
    if not ccd_path.exists():
        raise FileNotFoundError(f"Missing RCSB TES CCD definition: {ccd_path}")
    crystal_mol, tes_meta = build_tes_mol(PDB_PATH, ccd_path)
    if tes_meta["formula"] != "C19H28O2" or tes_meta["heavy_atoms"] != 21:
        raise ValueError(f"TES chemical validation failed: {tes_meta}")
    expected_stereo = tes_meta["expected_stereo"]
    observed_stereo = tes_meta["observed_stereo"]
    if expected_stereo != observed_stereo:
        raise ValueError(f"TES stereochemistry does not match CCD: expected={expected_stereo}, observed={observed_stereo}")
    tes_outputs = [OUT / "TES_crystal.sdf", OUT / "TES_prepared.sdf", OUT / "TES.pdbqt"]
    if not all(path.exists() and path.stat().st_size > 0 for path in tes_outputs):
        raise FileNotFoundError("Existing validated TES reference files are required; this repair does not regenerate them")

    receptor_meta = write_clean_receptor(
        PDB_PATH,
        OUT / "receptor_repair_input.pdb",
        drop_residue_numbers=set(),
        preserve_seqres=True,
    )
    repair_info = repair_receptor_with_pdbfixer(OUT / "receptor_repair_input.pdb", OUT / "receptor_repaired.pdb")
    (OUT / "repair_info.json").write_text(json.dumps(repair_info, indent=2))

    tes_xyz = np.array(list(tes_coords.values()), dtype=float)
    center = (tes_xyz.min(axis=0) + tes_xyz.max(axis=0)) / 2.0
    extent = tes_xyz.max(axis=0) - tes_xyz.min(axis=0)
    # Approximately 5.8, 5.8, and 6.4 A margin around TES. The slightly
    # expanded box is intended to accommodate steroid-like substituents.
    size = np.array([18.0, 21.0, 16.0], dtype=float)
    if (OUT / "vina_box.txt").exists():
        center, size = read_vina_box(OUT / "vina_box.txt")
    else:
        raise FileNotFoundError("Existing vina_box.txt is required; this repair must not create a new box")
    write_box_pdb(OUT / "vina_box_repaired.pdb", center, size)
    (OUT / "vina_box_repaired.txt").write_text(
        "receptor = receptor_repaired.pdbqt\n"
        f"center_x = {center[0]:.3f}\ncenter_y = {center[1]:.3f}\ncenter_z = {center[2]:.3f}\n"
        f"size_x = {size[0]:.3f}\nsize_y = {size[1]:.3f}\nsize_z = {size[2]:.3f}\n"
        "exhaustiveness = 8\n"
        "num_modes = 20\n"
    )

    mk_receptor = shutil.which("mk_prepare_receptor.py")
    if not mk_receptor:
        raise RuntimeError("mk_prepare_receptor.py is not on PATH")
    receptor_log = run_command(
        [
            mk_receptor,
            "--read_pdb",
            str(OUT / "receptor_repaired.pdb"),
            "--output_basename",
            str(OUT / "receptor_repaired"),
            "--write_pdbqt",
            str(OUT / "receptor_repaired.pdbqt"),
            "--write_json",
            str(OUT / "receptor_repaired.json"),
            "--write_pdb",
            str(OUT / "receptor_repaired_H.pdb"),
            "--compute_charges",
            "--charge_model",
            "gasteiger",
        ],
        ROOT,
    )
    (OUT / "meeko_repaired_receptor.log").write_text(receptor_log + "\n")

    validation = validate_repaired_outputs(
        OUT,
        OUT / "receptor_repair_input.pdb",
        OUT / "receptor_repaired.pdb",
        OUT / "receptor_repaired_H.pdb",
        OUT / "receptor_repaired.pdbqt",
        OUT / "TES.pdbqt",
        OUT / "vina_box.txt",
        tes_coords,
        repair_info,
    )
    (OUT / "validation_repaired.json").write_text(json.dumps(validation, indent=2))

    versions = {}
    for package in ["gemmi", "rdkit", "meeko", "vina", "numpy", "scipy", "pandas", "prody", "pdbfixer", "openmm"]:
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = "unavailable"

    missing_atom_text = ", ".join(
        f"{row['resname']} {row['chain']}:{row['resseq']} ({', '.join(row['atoms'])})"
        for row in inspection["missing_atoms_REMARK470"]
    )
    missing_res_text = ", ".join(
        f"{row['resname']} {row['chain']}:{row['resseq']}" for row in inspection["missing_residues_REMARK465"]
    )
    het_text = ", ".join(f"{name} ({count} atoms)" for name, count in inspection["het_residue_atom_counts"].items())
    pocket_text = ", ".join(
        f"{row['residue']} ({row['min_distance_A']:.2f} A)" for row in inspection["pocket_residues_within_5A"]
    )
    report = f"""# 2AM9 AR receptor and native ligand preparation

## Scope

This report covers Step 1 only. No redocking, batch docking, score analysis, or processing of 2AMA/2AMB was performed. The original input file `{PDB_PATH.relative_to(ROOT)}` was not modified.

## 1. Raw structure inspection

- PDB: 2AM9; resolution: {inspection['resolution_A']:.2f} A; models: {inspection['models']}.
- Chain A is the only chain. It contains {inspection['chain_A_observed_protein_residues']} observed protein residues, numbered {inspection['observed_protein_range'][0]}-{inspection['observed_protein_range'][1]}.
- SEQRES/REMARK 465 indicate {len(inspection['missing_residues_REMARK465'])} missing residues at the N-terminal observed-structure boundary: {missing_res_text}.
- REMARK 470 reports missing atoms in: {missing_atom_text}.
- ATOM/HETATM records: {inspection['atom_records_ATOM']} / {inspection['atom_records_HETATM']}.
- HETATM components: {het_text}. TES is chain A residue 1000; water is HOH residues 1-223.
- Protein AltLoc: 14 residues contain A/B alternatives; each listed conformer has 0.50 occupancy. The preparation retains blank plus conformer A and records the choice; conformer B is excluded from the rigid receptor.
- TES pocket residues within 5 A of a TES heavy atom: {pocket_text}.
- Missing-atom residues within 6 A of TES: {', '.join(inspection['missing_atom_residues_within_6A']) or 'none'}.
- Histidines within 6 A of TES: {', '.join(row['residue'] for row in inspection['histidines_within_6A']) or 'none'}. No manual histidine tautomer assignment was imposed; Meeko standard residue templates and Gasteiger charge calculation were used. This remains a review point for visual inspection.

## 2. Native Testosterone preparation

TES coordinates were extracted from chain A, residue 1000, using the 21 experimental heavy-atom coordinates in 2AM9. Bond orders and CCD stereo annotations came from the local RCSB CCD definition `TES.cif`; the experimental PDB coordinates were not replaced by ideal or newly embedded coordinates.

- RDKit formula: `{tes_meta['formula']}`; expected: `C19H28O2`.
- Heavy atom count: {tes_meta['heavy_atoms']}; expected: 21.
- CCD vs coordinate-derived CIP labels: `{observed_stereo}`.
- Crystal SDF: `data/prepared/2AM9/TES_crystal.sdf`.
- Prepared explicit-H SDF: `data/prepared/2AM9/TES_prepared.sdf`; generated by RDKit `AddHs(addCoords=True)`, retaining the heavy-atom conformer.
- Meeko ligand PDBQT: `data/prepared/2AM9/TES.pdbqt`.

## 3. Receptor preparation and local repair

The repair input was rebuilt directly from the original 2AM9 PDB: chain A ATOM records were retained, TES/HOH/GOL/SO4/DTT and other HETATM records were removed, blank plus AltLoc A were retained, and AltLoc B was excluded. The input contains {receptor_meta['kept_atom_records']} protein ATOM records; the previously validated legacy files `receptor_clean.pdb` and `receptor.pdbqt` were not overwritten.

PDBFixer {versions['pdbfixer']} was applied to `receptor_repair_input.pdb`. `findMissingResidues()` identified the 654-669 unresolved segment, then `missingResidues` was set to an empty mapping before `findMissingAtoms()` and `addMissingAtoms()`. Thus no complete missing residue was modeled. The four existing residues were retained and the following 15 missing heavy atoms were added:

```text
LYS836: CG CD CE NZ
LYS847: CG CD CE NZ
ASN848: CG OD1 ND2
GLU893: CG CD OE1 OE2
```

No existing protein atom coordinate was changed by PDBFixer. Meeko {versions['meeko']} then prepared `receptor_repaired.pdb` with `--compute_charges --charge_model gasteiger`, writing `receptor_repaired_H.pdb`, `receptor_repaired.pdbqt`, and `receptor_repaired.json`. No minimization or MD was run.

## 4. Docking box

TES heavy-atom coordinate extent was {extent.tolist()} A. The box center is derived from the TES bounding-box midpoint:

```text
center_x = {center[0]:.3f}
center_y = {center[1]:.3f}
center_z = {center[2]:.3f}
size_x   = {size[0]:.3f}
size_y   = {size[1]:.3f}
size_z   = {size[2]:.3f}
```

The existing `vina_box.txt` was not modified. The repaired-receptor copy `vina_box_repaired.txt` has the same center and dimensions but points to `receptor_repaired.pdbqt`; `vina_box_repaired.pdb` is a matching PyMOL visualization.

## 5. Quality checks

All repaired-receptor checks passed:

- required repaired outputs are non-empty: `{validation['all_required_outputs_nonempty']}`;
- repair input atom count: {validation['geometry']['original_atom_count']}; repaired atom count: {validation['geometry']['repaired_atom_count']}; increase: {validation['geometry']['repaired_atom_count'] - validation['geometry']['original_atom_count']};
- protein residue count: {validation['geometry']['original_residue_count']} -> {validation['geometry']['repaired_residue_count']}; residue keys lost: `{validation['geometry']['residue_keys_lost']}`;
- existing-atom mapping compared: {validation['geometry']['existing_atom_count_compared']}; RMSD: {validation['geometry']['existing_atom_coordinate_rmsd_A']:.6f} A; maximum shift: {validation['geometry']['existing_atom_max_shift_A']:.6f} A;
- TES 6 A pocket existing-atom count: {validation['geometry']['pocket_existing_atom_coordinate_check']['count']}; maximum shift: {validation['geometry']['pocket_existing_atom_coordinate_check']['max_shift_A']:.6f} A;
- added-atom mapping exactly matches PDBFixer missing-atom list: `{validation['geometry']['added_atom_count_by_mapping'] == repair_info['added_atom_count']}`;
- severe nonbonded heavy-atom collisions: {len(validation['severe_heavy_atom_collisions'])}; new-side-chain bond-length outliers: {len(validation['bond_length_outliers'])};
- repaired receptor contains non-protein components: `{validation['repaired_pdb_contains_nonprotein_components']}`;
- Vina Python API repaired receptor + existing TES load: `{validation['repaired_receptor_vina_load']}`; no docking search was executed.

## 6. Software and parameters

- Python: {sys.version.split()[0]}; platform: {platform.platform()}
- package versions: {json.dumps(versions, sort_keys=True)}
- receptor: chain A, blank/A AltLoc policy, HETATM removed, PDBFixer missing-atom repair, Gasteiger charges
- ligand: RCSB CCD TES bond graph + 2AM9 experimental heavy-atom coordinates
- box: center/size above; `exhaustiveness=8`, `num_modes=20` are recorded only for later Vina use and were not applied in this step

## 7. Remaining structural cautions

The structure still has an unresolved N-terminal segment (654-669) and 14 A/B alternate-location residues. The four repaired side chains are outside the TES 6 A pocket neighborhood, but their PDBFixer-generated conformations should still be visually reviewed. Histidine protonation was left to Meeko's standard templates and should be visually checked if a pocket histidine becomes mechanistically important.

## 8. Next-step inputs

- receptor (new default candidate): `data/prepared/2AM9/receptor_repaired.pdbqt`
- receptor H-PDB: `data/prepared/2AM9/receptor_repaired_H.pdb`
- native ligand (unchanged): `data/prepared/2AM9/TES.pdbqt`
- Vina box: `data/prepared/2AM9/vina_box.txt`
- repaired box copy: `data/prepared/2AM9/vina_box_repaired.txt`
- visual checks: `data/prepared/2AM9/receptor_repaired_H.pdb`, `TES_crystal.sdf`, and `vina_box_repaired.pdb`

Generated by `scripts/01_prepare_2AM9.py`; independent validation is available via `scripts/01_validate_preparation.py`.
"""
    REPORT.write_text(report)
    print(json.dumps({"outdir": str(OUT), "report": str(REPORT), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()

