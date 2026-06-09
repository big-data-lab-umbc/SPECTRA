# Publish audit

This directory is a cleaned publish copy of the development workspace. It keeps
the code needed for the paper-facing SPECTRA method and Table 1 baselines.

## Kept

- Core `spectra` package
- Dataset loaders for the published segmentation datasets
- Prithvi, ScaleMAE, and SatMAE backbone support
- Band selection baseline adapter
- BRE adapter
- Nested LoRA and fixed baseline schedules
- ST-LoRA / STPlanner transfer and repair planning
- Fine-tuning, checkpoint evaluation, preflight, split generation, and timing scripts
- Table-1 YAML configs and fixed splits

## Removed

- Legacy planner code and public references
- Unsupported visual-only backbone paths
- Historical tokenizer suffixes
- Historical loss diagnostics
- Historical transport-adapter code and schedules
- Older embedding/router variants
- Ad hoc launch grids, notebooks, checkpoints, generated figures, logs, and local queue files
- User-specific hard-coded dataset roots

## Naming

The publish version uses the paper-facing method names:

- `BRE`: Band-Routed Embedding
- `ST-LoRA`: Stage-wise Transferability-aware LoRA
- `STPlanner`: planner used inside ST-LoRA
