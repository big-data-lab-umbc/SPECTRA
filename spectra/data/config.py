"""Shared paths, artifact locations, and dataset/backbone constants.

Dataset roots are read from environment variables so the repository can be
published without private filesystem paths. See README.md for examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = _REPO_ROOT / "features"
FIXED_SPLITS_DIR = _REPO_ROOT / "fixed_splits"
FINETUNE_RESULTS_DIR = _REPO_ROOT / "finetune_results"
BASELINE_METRICS_FILE = _REPO_ROOT / "baseline_metrics.json"
RESULTS_DIR = _REPO_ROOT / "results"


def _env_path(name: str, default: str = "") -> Path:
    value = os.environ.get(name, default)
    return Path(value).expanduser() if value else Path("__unset__")


FIRE_SCARS_ROOT = _env_path("SPECTRA_FIRE_SCARS_ROOT")
SEN1FLOODS11_ROOT = _env_path("SPECTRA_SEN1FLOODS11_ROOT")
LANDSLIDE4SENSE_ROOT = _env_path("SPECTRA_LANDSLIDE4SENSE_ROOT")
GEOBENCH_SA_CROP_TYPE_ROOT = _env_path("SPECTRA_GEOBENCH_SA_CROP_TYPE_ROOT")

ABI_CLOUD_ROOT = _env_path("SPECTRA_ABI_CLOUD_ROOT")
MULTITEMPORAL_CROP_ROOT = _env_path("SPECTRA_MULTITEMPORAL_CROP_ROOT")
CLOUDSEN12_ROOT = _env_path("SPECTRA_CLOUDSEN12_ROOT")
BIGEARTHNET_ROOT = _env_path("SPECTRA_BIGEARTHNET_ROOT")
LOVEDA_ROOT = _env_path("SPECTRA_LOVEDA_ROOT")

SEN1FLOODS11_HANDLABELED = SEN1FLOODS11_ROOT / "v1.1" / "data" / "flood_events" / "HandLabeled"
FIRE_SCARS_TRAINING = FIRE_SCARS_ROOT / "training"

DATASET_ROOTS: dict[str, Path] = {
    "fire_scars": FIRE_SCARS_ROOT,
    "sen1floods11": SEN1FLOODS11_ROOT,
    "landslide4sense": LANDSLIDE4SENSE_ROOT,
    "geobench_sa_crop_type": GEOBENCH_SA_CROP_TYPE_ROOT,
}
PROSPECTIVE_DATASET_ROOTS: dict[str, Path] = {
    "cloudsen12": CLOUDSEN12_ROOT,
    "bigearthnet": BIGEARTHNET_ROOT,
    "loveda": LOVEDA_ROOT,
}
ALL_DATASET_ROOTS = {**DATASET_ROOTS, **PROSPECTIVE_DATASET_ROOTS}


@dataclass(frozen=True)
class FullBandConfig:
    in_chans: int
    sensor_key: str
    bands: tuple[Any, ...]


FULL_BAND_CONFIGS: dict[str, FullBandConfig] = {
    "fire_scars": FullBandConfig(6, "fire_scars", ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")),
    "sen1floods11": FullBandConfig(13, "sen1floods11", tuple(f"B{i}" for i in range(1, 14))),
    "geobench_sa_crop_type": FullBandConfig(12, "geobench_sa_crop_type", ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")),
    "landslide4sense": FullBandConfig(14, "landslide4sense", tuple(f"B{i}" for i in range(1, 15))),
    "abi_cloud": FullBandConfig(16, "abi_cloud", tuple(range(16))),
    "multitemporal_crop": FullBandConfig(18, "multitemporal_crop", tuple(f"T{t}_{b}" for t in (1, 2, 3) for b in ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"))),
    "cloudsen12": FullBandConfig(13, "cloudsen12", tuple(f"B{i}" for i in range(1, 14))),
    "bigearthnet": FullBandConfig(12, "bigearthnet", tuple(f"B{i}" for i in range(1, 13))),
    "loveda": FullBandConfig(3, "loveda", ("RED", "GREEN", "BLUE")),
}


@dataclass(frozen=True)
class BackboneSpec:
    model_id: str
    n_layers: int
    embed_dim: int
    patch_size: int
    n_stages: int = 4


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "prithvi_eo_v2_600": BackboneSpec("prithvi_eo_v2_600", n_layers=32, embed_dim=1280, patch_size=14),
    "satmae_sentinel_vitl": BackboneSpec("satmae_sentinel_vitl", n_layers=24, embed_dim=1024, patch_size=8),
    "scalemae_fmow_rgb": BackboneSpec("scalemae_fmow_rgb", n_layers=24, embed_dim=1024, patch_size=16),
}

CALIBRATION_DATASETS = ("fire_scars", "sen1floods11", "landslide4sense", "geobench_sa_crop_type")
PROSPECTIVE_DATASETS: tuple[str, ...] = ()
ALL_DATASETS = CALIBRATION_DATASETS
CALIBRATION_BACKBONES = ("prithvi_eo_v2_600", "satmae_sentinel_vitl", "scalemae_fmow_rgb")
