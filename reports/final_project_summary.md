# Final Project Summary

## Project

**Steroid-like Molecular Docking and AR Binding Analysis**

This project explores how bioactivity-profile clusters and C17 substitution
patterns relate to predicted androgen-receptor docking scores, binding
orientation, and local geometric contacts.

## Dataset and workflow

- Initial teacher-derived dataset: 47 steroid-like records across four clusters.
- Docking-ready, stereochemistry-validated compounds: 24.
- Primary receptor: AR LBD structure 2AM9.
- Validated auxiliary receptor: 2AMA.
- Excluded receptor: 2AMB, because MET895 failed structural/Meeko quality
  control and could not be repaired without unacceptable coordinate changes.

Workflow: bioactivity clusters → compound selection → stereochemistry
curation → receptor preparation → native-ligand redocking → batch docking →
pose/interaction analysis → C17 orientation robustness → receptor-ensemble
comparison → H3 local-packing validation.

## Main verified findings

- Testosterone redocking on 2AM9 and DHT redocking on 2AMA reproduced the
  experimental poses under the project’s direct heavy-atom RMSD protocol.
- Across 24 compounds, 2AM9 and 2AMA top-pose scores were strongly rank
  correlated: Spearman rho = 0.937, n = 24.
- Row 5373 remained native-like across the 10-seed analyses on both receptors.
- Row 2412 changed orientation classification between 2AM9 and 2AMA, showing
  receptor-conformation sensitivity under the defined computational protocol.
- H3 local packing was `partially_supported`: Row 705’s ethynyl proximity was
  similar across receptors, but its C17–OH short-distance pattern was strongly
  receptor-dependent.

## Scientific boundary

Docking scores are not experimental binding free energies. Geometric proximity
does not by itself prove a hydrogen bond, steric clash, binding affinity,
agonism, antagonism, or AR selectivity. The bioactivity clusters are
teacher-derived annotations rather than direct experimental labels.

## Reproducibility

The compact public environment specification is `environment.github.yml`.
The installed environment snapshot `environment.yml` is retained separately;
it contains machine-specific package metadata and should be sanitized before
public release if necessary.

The analysis-only entry point is:

```bash
conda env create -f environment.github.yml
conda activate steroid_docking
python run_analysis_only.py
```

The entry point reads existing docking outputs and regenerates the Step 6A H3
tables and figures. It does not call AutoDock Vina docking.

