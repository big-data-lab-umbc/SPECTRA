# Appendix: Experiment Hyperparameters

This appendix documents the hyperparameters exposed by the publishable SPECTRA
codebase. All values refer to the cleaned public implementation in this
repository.

## Public Methods

| CLI method | Input adapter | Backbone adaptation |
| --- | --- | --- |
| `lp` | closest-band selection | frozen encoder, train task head only |
| `lora8` | closest-band selection | uniform LoRA rank 8, frozen encoder weights |
| `lora16` | closest-band selection | uniform LoRA rank 16, frozen encoder weights |
| `lora32` | closest-band selection | uniform LoRA rank 32, frozen encoder weights |
| `lora64` | closest-band selection | uniform LoRA rank 64, frozen encoder weights |
| `last_stage` | closest-band selection | last stage unfrozen, no LoRA |
| `surgical` | closest-band selection | highest-scoring stage unfrozen, no LoRA |
| `full_ft` | closest-band selection | all stages unfrozen, no LoRA |
| `spectra` | BRE, bypassed when target bands already match pretraining bands | ST-LoRA ranks selected by STPlanner |

Closest-band selection maps the pretraining sensor wavelengths to the nearest
available target bands. BRE starts from the same selected-band anchor and learns
a zero-initialized residual correction from all input bands while keeping the
pretrained patch embedding frozen.

## Optimizer and Training Defaults

| Hyperparameter | Default |
| --- | --- |
| Optimizer | AdamW |
| LoRA adapter learning rate | `5.0e-5` |
| Unfrozen backbone learning rate | `5.0e-6` |
| Head learning rate | `5.0e-4` |
| Weight decay | `0.05` |
| Gradient clipping | global norm, `1.0` |
| Batch size | `8` unless overridden by the cell config |
| Workers | `4` |
| Default crop size | `224` |
| Default evaluation metric | `miou` |

Checkpoints are optional with `--save-checkpoints`. Auto test evaluation is
enabled by default; when active it saves best/final adapted checkpoints and
evaluates them on the test split.

## Loss Defaults

The public CLI exposes two loss modes:

| CLI flag | Formula |
| --- | --- |
| `--loss-mode ce_dice_dwa` | Dynamic-weight average of CE or weighted CE and Dice |
| `--loss-mode ce_dice` | CE or weighted CE plus `dice_lambda * Dice` |

Default loss settings:

| Hyperparameter | Default |
| --- | --- |
| `--loss-mode` | `ce_dice_dwa` |
| `--dice-lambda` | `1.0` |
| `--dwa-temperature` | `2.0` |
| `--minority-boost-cap-ratio` | `8.0` |
| `class_weights` | `auto` |
| Auto class-weighting scheme | inverse frequency |
| Absent class weight | `0.0` |
| Binary minority boost | `2.0` |
| Dice class mode | `default` |
| Dice include background | `false` |

GEO-Bench crop-type configs use `enet_log` class weighting with
`log_smoothing=1.02` and `dice.class_mode=present_train` for Prithvi. Binary
segmentation configs commonly use foreground-only Dice when specified in the
cell config.

## ST-LoRA / STPlanner Defaults

STPlanner first profiles stage-wise transferability with LogME and then assigns
LoRA ranks. The `transfer` strategy allocates rank toward high-transferability
stages; `repair` allocates rank toward high-gap stages.

| Hyperparameter | Default |
| --- | --- |
| `--st-planner` | `transfer` |
| `--st-reference-rank` | `32` |
| `--st-tau` | `0.05` |
| `--st-stage-prior` | `0.8,1.0,1.1,1.2` |
| `--st-budget-candidates` | `32,48,60,72,80,92,96,104,112` |
| `--st-budget-f-min` | `0.40` |
| `--st-budget-f-max` | `0.85` |
| `--st-budget-midpoint` | `0.50` |
| `--st-budget-slope` | `3.0` |
| `--st-budget-override` | unset |
| Rank grid | `4,8,16,32,64` |
| Minimum stage rank | `4` |
| Profiling images | `1000` |
| Profiling purity threshold | `0.8` |

For `spectra`, the default maximum LoRA rank is `64`. For uniform LoRA
baselines, the default maximum rank is the method rank.

## Backbones

| Backbone config key | Layers | Embedding dim | Patch size | Notes |
| --- | ---: | ---: | ---: | --- |
| `prithvi_eo_v2_600` | 32 | 1280 | 14 | TerraTorch Prithvi segmentation path |
| `satmae_sentinel_vitl` | 24 | 1024 | 8 | SatMAE Sentinel-style multispectral path |
| `scalemae_fmow_rgb` | 24 | 1024 | 16 | Scale-MAE RGB-native path |

## Dataset Cells

| Config | Backbone | Dataset | Epochs | Image/crop | Pad multiple | Class weighting | Dice mode |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| `prithvi_fire_scars.yaml` | Prithvi | FireScars | 10 | 224/224 | 14 | inverse frequency | default |
| `prithvi_sen1floods11.yaml` | Prithvi | Sen1Floods11 | 50 | 224/224 | 14 | inverse frequency | default |
| `prithvi_landslide4sense.yaml` | Prithvi | Landslide4Sense | 50 | 128/128 | 14 | inverse frequency | foreground |
| `prithvi_geobench_sa_crop_type.yaml` | Prithvi | GEO-Bench crop type | 50 | 224/224 | 14 | enet log | present train |
| `satmae_fire_scars.yaml` | SatMAE | FireScars | 10 | 128/128 | 8 | inverse frequency | foreground |
| `satmae_sen1floods11.yaml` | SatMAE | Sen1Floods11 | 50 | 128/128 | 8 | inverse frequency | foreground |
| `satmae_landslide4sense.yaml` | SatMAE | Landslide4Sense | 50 | 128/128 | 8 | inverse frequency | foreground |
| `satmae_geobench_sa_crop_type.yaml` | SatMAE | GEO-Bench crop type | 50 | 96/96 | 8 | enet log | default |
| `scalemae_fire_scars.yaml` | Scale-MAE | FireScars | 10 | 224/224 | 16 | inverse frequency | foreground |
| `scalemae_sen1floods11.yaml` | Scale-MAE | Sen1Floods11 | 50 | 224/224 | 16 | inverse frequency | default |
| `scalemae_landslide4sense.yaml` | Scale-MAE | Landslide4Sense | 50 | 128/128 | 16 | inverse frequency | default |
| `scalemae_geobench_sa_crop_type.yaml` | Scale-MAE | GEO-Bench crop type | 50 | 224/224 | 16 | enet log | default |

All listed configs use `batch_size=8` and `eval_metric=miou`.

## Seed Protocol

The default seed is `42`. If a specialized seed is omitted, it falls back to
`--seed`.

| Seed flag | Purpose |
| --- | --- |
| `--split-seed` | train/validation/test split selection |
| `--model-seed` | model initialization and deterministic setup |
| `--residual-seed` | BRE initialization |
| `--lora-seed` | LoRA initialization |
| `--head-seed` | task head reset |
| `--loader-seed` | data loader ordering |
| `--train-shuffle-seed` | BRE training-time diagnostic RNG |
| `--eval-shuffle-seed` | BRE evaluation-time diagnostic RNG |

## Example Commands

Uniform LoRA baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
  --config configs/prithvi_sen1floods11.yaml \
  --method lora32 \
  --seed 42 \
  --split-seed 42 \
  --loss-mode ce_dice_dwa \
  --save-checkpoints
```

SPECTRA with repair planning:

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
