# SPECTRA

This repository contains the publishable implementation of **SPECTRA** for
segmentation fine-tuning of geospatial foundation models.

SPECTRA combines:

1. **BRE-PoG**: Band-Routed Embedding with per-output gates. BRE keeps the
   pretrained patch embedding frozen, selects pretrained-compatible target
   bands, and learns a zero-initialized residual correction from all available
   source bands.
2. **ST-LoRA**: Stage-wise Transferability-aware LoRA. The **STPlanner** profiles
   stage-wise transferability and chooses per-stage LoRA ranks under a LoRA rank
   budget. Both `transfer` and `repair` planning modes are included.

The public training CLI intentionally exposes only the methods used for the main
comparison table:

- `lp`
- `lora8`, `lora16`, `lora32`, `lora64`
- `last_stage`
- `surgical`
- `full_ft`
- `spectra` = BRE-PoG + ST-LoRA

## Installation

```bash
pip install -e .
```

Backbone-specific dependencies such as TerraTorch/Prithvi should be installed
in the active environment. If they are local checkouts, point the runner to
them with:

```bash
export TERRATORCH_ROOT=/path/to/terratorch
export PRITHVI_EO_ROOT=/path/to/Prithvi-EO-2.0
```

## Dataset roots

Set dataset roots with environment variables. No private paths are hard-coded.

```bash
export SPECTRA_FIRE_SCARS_ROOT=/path/to/hls_burn_scars
export SPECTRA_SEN1FLOODS11_ROOT=/path/to/sen1floods11
export SPECTRA_LANDSLIDE4SENSE_ROOT=/path/to/Landslide4Sense
export SPECTRA_GEOBENCH_SA_CROP_TYPE_ROOT=/path/to/Geobench_SA_crop_type
```

## Training examples

All published configs use UPerNet-style decoding, half learning rates from the
main experiments, and either DWA(CE/weighted CE, Dice) or CE/weighted CE+Dice.

LoRA32 baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
  --config configs/prithvi_sen1floods11.yaml \
  --method lora32 \
  --seed 42 \
  --split-seed 42 \
  --loss-mode ce_dice_dwa \
  --save-checkpoints
```

SPECTRA with STPlanner repair mode:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
  --config configs/prithvi_sen1floods11.yaml \
  --method spectra \
  --st-planner repair \
  --seed 42 \
  --split-seed 42 \
  --loss-mode ce_dice_dwa \
  --save-checkpoints
```

SPECTRA with fixed CE/weighted CE + Dice1:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
  --config configs/satmae_sen1floods11.yaml \
  --method spectra \
  --st-planner transfer \
  --loss-mode ce_dice \
  --dice-lambda 1.0 \
  --seed 42 \
  --split-seed 42 \
  --save-checkpoints
```

## Included experiment cells

Configs are provided for Prithvi-EO-2.0 600M, ScaleMAE, and SatMAE on:

- Sen1Floods11
- FireScars
- Landslide4Sense
- GEO-Bench SA Crop Type

For Prithvi + FireScars, where the target input bands already match the
pretrained 6-band patch embedding, `spectra` automatically bypasses BRE and
runs native patch embedding + ST-LoRA.
