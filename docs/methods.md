# Methods

This compact release documents the analysis layer of the steroid-like AR
docking project. The initial data consisted of 47 records in four
teacher-derived bioactivity clusters; 24 compounds with explicit,
independently checked stereochemistry were used for the docking comparison.

## Receptors and docking

The primary receptor was the human androgen-receptor ligand-binding domain
structure 2AM9. 2AMA was used as an auxiliary receptor-conformation check;
2AMB was excluded after the MET895 structural quality-control assessment.
Native-ligand redocking and batch docking were performed using the
corresponding prepared 2AM9 and 2AMA receptor structures, a common docking box
in the aligned coordinate frame, and consistent AutoDock Vina settings.
Multi-seed and receptor-ensemble analyses
were descriptive robustness checks, not experimental affinity measurements.

## Public analysis reproduction

The default entry point is:

```bash
conda env create -f environment.github.yml
conda activate steroid_docking
python run_analysis_only.py
```

This summary-only mode reads the compact public tables and writes regenerated
score, orientation, and H3 local-contact summaries to `results/summary_only/`.
It does not call AutoDock Vina and does not require PDB, SDF, or PDBQT files.
Use `python run_analysis_only.py --mode full` only with the complete local pose
and receptor inputs.

## Full workflow boundary

The preparation and docking scripts are retained for provenance and for users
who obtain the required external inputs independently. The compact release
does not redistribute the raw teacher-profile workbook, RCSB coordinate files,
prepared receptor/ligand structures, or full pose collections. Consequently,
the public release supports analysis reproduction from published summaries;
bit-for-bit reconstruction of the original docking outputs requires the
reviewed external inputs, software versions, and preparation decisions.

## Interpretation

Vina scores, pose orientations, contact distances, and multi-seed agreement are
computational observations. They do not establish experimental AR binding,
agonism, antagonism, affinity, or receptor selectivity. Distances below the
project's screening thresholds are reported as geometric descriptors and are
not automatically assigned as hydrogen bonds or steric clashes.
