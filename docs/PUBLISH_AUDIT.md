# Publish audit

This directory is a cleaned publish copy of the development workspace. It keeps the code
needed for the paper-facing SPECTRA method and Table 1 baselines.

## Kept

- Core `spectra` package
- Dataset loaders for the published segmentation datasets
- Prithvi, ScaleMAE, and SatMAE backbone support
- Band selection baseline adapter
- BRE-PoG adapter
- Nested LoRA and fixed baseline schedules
- ST-LoRA / STPlanner transfer and repair planning
- Fine-tuning, checkpoint evaluation, preflight, split generation, and timing scripts
- Table-1 YAML configs and fixed splits

## Removed from public scripts

- Ad hoc launch grids and SLURM logs
- Result JSONs, checkpoints, generated figures, and local queue files
- Notebook checkpoints and debug notebooks
- User-specific hard-coded dataset roots
- Public exposure of older experimental methods in the CLI

## Naming

The development workspace used earlier planner names. In the publish version,
the paper-facing method names are:

- `ST-LoRA`: Stage-wise Transferability-aware LoRA
- `STPlanner`: planner used inside ST-LoRA

A small backward-compatible `spectra.planner.star_v2` alias remains so older
checkpoints/scripts can still import the old class names if needed.
