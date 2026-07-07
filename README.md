# SPECTRA

This repository contains the publishable implementation of **SPECTRA** for
segmentation fine-tuning of geospatial foundation models.

SPECTRA combines:

1. **BRE**: Band-Routed Embedding. BRE keeps the pretrained patch embedding
   frozen, selects pretrained-compatible target bands, and learns a
   zero-initialized residual correction from all available source bands.
2. **ST-LoRA**: Stage-wise Transferability-aware LoRA. The **STPlanner** profiles
   stage-wise transferability and chooses per-stage LoRA ranks under a LoRA rank
   budget. Both `transfer` and `repair` planning modes are included.

The public training CLI exposes only the methods used for the main comparison
table:

- `lp`
- `lora8`, `lora16`, `lora32`, `lora64`
- `last_stage`
- `surgical`
- `full_ft`
- `spectra` = BRE + ST-LoRA

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

For a source checkout setup:

```bash
git clone https://github.com/torchgeo/terratorch /path/to/terratorch
git clone https://github.com/NASA-IMPACT/Prithvi-EO-2.0 /path/to/Prithvi-EO-2.0
```

## Dataset roots

Set dataset roots with environment variables. No private paths are hard-coded.

```bash
export SPECTRA_FIRE_SCARS_ROOT=/path/to/hls_burn_scars
export SPECTRA_SEN1FLOODS11_ROOT=/path/to/sen1floods11
export SPECTRA_LANDSLIDE4SENSE_ROOT=/path/to/Landslide4Sense
export SPECTRA_GEOBENCH_SA_CROP_TYPE_ROOT=/path/to/Geobench_SA_crop_type
```

Download the datasets from their upstream sources, then set the environment
variables above to the local directories with the expected layout.

| Dataset | Upstream source | Example download/setup |
| --- | --- | --- |
| FireScars / HLS Burn Scar Scenes | https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars | `huggingface-cli download --repo-type dataset ibm-nasa-geospatial/hls_burn_scars --local-dir /path/to/hls_burn_scars`; extract the downloaded archive so `/path/to/hls_burn_scars/training` contains `*_merged.tif` images and matching `*.mask.tif` masks. |
| Sen1Floods11 | https://github.com/cloudtostreet/Sen1Floods11 | `gsutil -m rsync -r gs://sen1floods11 /path/to/sen1floods11`; SPECTRA reads the hand-labeled Sentinel-2 split under `v1.1/data/flood_events/HandLabeled`. |
| Landslide4Sense | https://www.iarai.ac.at/landslide4sense and https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense | Download from the IARAI challenge page or the Hugging Face mirror; SPECTRA expects `images/{train,validation,test}` and `annotations/{train,validation,test}` with HDF5 files. |
| GEO-Bench SA Crop Type | https://github.com/ServiceNow/geo-bench and https://huggingface.co/datasets/recursix/geo-bench-1.0 | Install GEO-Bench and run `GEO_BENCH_DIR=/path/to/geobench geobench-download`; set `SPECTRA_GEOBENCH_SA_CROP_TYPE_ROOT` to the extracted SA crop-type task directory containing `default_partition.json` and `data/*.hdf5`. |

## Training examples

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

## References

SPECTRA reuses public backbone, decoder, and adaptation implementations instead
of vendoring them into this repository.

| Component | Upstream reference | Used in SPECTRA |
| --- | --- | --- |
| TerraTorch framework | https://github.com/torchgeo/terratorch | Prithvi segmentation task and shared UPerNet decoder dependency |
| Prithvi-EO-2.0 release | https://github.com/NASA-IMPACT/Prithvi-EO-2.0 | `prithvi_eo_v2_600` configs |
| Prithvi-EO-2.0 600M weights | https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M | Pretrained Prithvi checkpoint used by the TerraTorch task |
| TerraTorch segmentation task | https://github.com/torchgeo/terratorch/blob/main/terratorch/tasks/segmentation_tasks.py | Prithvi backbone implementation path in `scripts/finetune.py` and `scripts/timing_pass.py` |
| TerraTorch UPerNet decoder | https://github.com/torchgeo/terratorch/blob/main/terratorch/models/decoders/upernet_decoder.py | Dense prediction heads for SatMAE and ScaleMAE wrappers |
| SatMAE official code | https://github.com/sustainlab-group/SatMAE | `satmae_sentinel_vitl` wrapper in `spectra/backbone/satmae.py` |
| TorchGeo Scale-MAE | https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/scale_mae.py | `scalemae_fmow_rgb` wrapper in `spectra/backbone/scalemae.py` |
| LoRA reference implementation | https://github.com/microsoft/LoRA | LoRA adaptation design reference for `spectra/adapter/nested_lora.py` |
| LogME reference implementation | https://github.com/thuml/LogME | Transferability scoring reference for `spectra/planner/logme_profiler.py` |
