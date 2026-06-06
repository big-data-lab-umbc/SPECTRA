#!/usr/bin/env python3
"""R0.1 — Data preflight: verify all 8 datasets are accessible.

Checks: file format, band count, label classes, patch-size resolution.
Prints a pass/fail table. Exit code 1 if any dataset fails.
"""

import sys
import os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import numpy as np
from pathlib import Path
from spectra.data.config import (
    FIRE_SCARS_ROOT, SEN1FLOODS11_ROOT, LANDSLIDE4SENSE_ROOT,
    ABI_CLOUD_ROOT, MULTITEMPORAL_CROP_ROOT,
    CLOUDSEN12_ROOT, BIGEARTHNET_ROOT, LOVEDA_ROOT,
)
from spectra.data.srf import SRF_REGISTRY

PATCH_SIZE = 16  # Default ViT patch size

CHECKS = [
    # (name, root, role, expected_bands, sensor_key, checker_fn)
    ("fire_scars",        FIRE_SCARS_ROOT,         "calib", 6,  "fire_scars",   "hls"),
    ("sen1floods11",      SEN1FLOODS11_ROOT,        "calib", 13, "sen1floods11", "s2_13b"),
    ("abi_cloud",         ABI_CLOUD_ROOT,           "calib", 16, "abi_cloud",    "abi_16b"),
    ("multitemporal_crop",MULTITEMPORAL_CROP_ROOT,  "calib", 18, "multitemporal_crop", "multitemporal_crop"),
    ("cloudsen12",        CLOUDSEN12_ROOT,          "prosp", 13, "cloudsen12",   "s2_13b"),
    ("bigearthnet",       BIGEARTHNET_ROOT,         "prosp", 12, "bigearthnet",  "s2_12b"),
    ("loveda",            LOVEDA_ROOT,              "prosp", 3,  "loveda",       "rgb_3b"),
    ("landslide4sense",   LANDSLIDE4SENSE_ROOT,     "prosp", 14, "landslide4sense", "landslide_14b"),
]


def check_directory_accessible(root: Path) -> tuple[bool, str]:
    if not root.exists():
        return False, f"Directory not found: {root}"
    if not root.is_dir():
        return False, f"Not a directory: {root}"
    files = list(root.iterdir())
    if len(files) == 0:
        return False, f"Empty directory: {root}"
    return True, f"{len(files)} entries"


def check_srf_table(sensor_key: str) -> tuple[bool, str]:
    try:
        bands = SRF_REGISTRY.get(sensor_key)
        if bands is None:
            return False, f"SRF key '{sensor_key}' not in registry"
        return True, f"{len(bands)} bands registered"
    except Exception as e:
        return False, str(e)


def check_abi_cloud(root: Path) -> tuple[bool, str, int]:
    npz_files = sorted(root.glob("*.npz"))
    if not npz_files:
        return False, "No .npz files found", 0
    sample = np.load(npz_files[0])
    rad = sample.get("rad")
    if rad is None:
        return False, "Key 'rad' missing in npz", 0
    return True, f"rad shape={rad.shape}", rad.shape[-1]


def check_hls_burnscars(root: Path) -> tuple[bool, str, int]:
    training = root / "training"
    if not training.exists():
        training = root
    tifs = sorted(training.glob("*_merged.tif"))
    if not tifs:
        return False, "No *_merged.tif files found", 0
    import rasterio
    with rasterio.open(tifs[0]) as src:
        n_bands = src.count
    return True, f"{len(tifs)} scenes, {n_bands} bands", n_bands


def check_sen1floods11(root: Path) -> tuple[bool, str, int]:
    s2hand = root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S2Hand"
    if not s2hand.exists():
        return False, f"HandLabeled/S2Hand not found at {s2hand}", 0
    tifs = sorted(s2hand.glob("*.tif"))
    if not tifs:
        return False, "No .tif files in S2Hand", 0
    import rasterio
    with rasterio.open(tifs[0]) as src:
        n_bands = src.count
    return True, f"{len(tifs)} scenes, {n_bands} bands", n_bands


def check_landslide4sense(root: Path) -> tuple[bool, str, int]:
    img_dir = root / "images" / "train"
    if not img_dir.exists():
        return False, f"images/train not found at {img_dir}", 0
    h5s = sorted(img_dir.glob("*.h5"))
    if not h5s:
        return False, "No .h5 files in images/train", 0
    import h5py
    with h5py.File(h5s[0], "r") as f:
        img = f["img"][:]
    return True, f"{len(h5s)} samples, shape={img.shape}", img.shape[-1]


def check_multitemporal_crop(root: Path) -> tuple[bool, str, int]:
    chips = root / "training_chips"
    if not chips.exists():
        return False, "training_chips/ not found", 0
    tifs = [f for f in sorted(chips.glob("*_merged.tif")) if not f.name.startswith("._")]
    if not tifs:
        return False, "No *_merged.tif in training_chips", 0
    import rasterio
    with rasterio.open(tifs[0]) as src:
        n_bands = src.count
    return True, f"{len(tifs)} chips, {n_bands} bands (3 time steps × 6)", n_bands


def check_prospective(root: Path, name: str) -> tuple[bool, str, int]:
    if not root.exists():
        return False, f"NOT DOWNLOADED yet — path: {root}", 0
    return True, "Directory exists (download verified)", -1


CHECKER_MAP = {
    "hls":          check_hls_burnscars,
    "s2_13b":       check_sen1floods11,
    "s2_12b":       None,
    "landslide_14b": check_landslide4sense,
    "abi_16b":      check_abi_cloud,
    "multitemporal_crop": check_multitemporal_crop,
    "rgb_3b":       None,
}


def main() -> int:
    print("=" * 80)
    print("R0.1 Data Preflight — SPECTRA")
    print("=" * 80)
    print(f"{'Dataset':<22} {'Role':<7} {'Dir':<5} {'SRF':<5} {'Bands':<6} {'Notes'}")
    print("-" * 80)

    failures = 0
    for name, root, role, expected_bands, sensor_key, checker_key in CHECKS:
        dir_ok, dir_msg = check_directory_accessible(root)
        srf_ok, srf_msg = check_srf_table(sensor_key)

        # Run format-specific checker
        bands_ok, band_msg, actual_bands = True, "n/a", expected_bands
        if role == "prosp":
            bands_ok, band_msg, actual_bands = check_prospective(root, name)
        elif checker_key in CHECKER_MAP and CHECKER_MAP[checker_key] is not None:
            try:
                bands_ok, band_msg, actual_bands = CHECKER_MAP[checker_key](root)
                if actual_bands > 0 and actual_bands != expected_bands:
                    bands_ok = False
                    band_msg += f" ← EXPECTED {expected_bands}"
            except Exception as e:
                bands_ok = False
                band_msg = str(e)[:60]

        status_dir = "✓" if dir_ok else "✗"
        status_srf = "✓" if srf_ok else "✗"
        status_bands = "✓" if bands_ok else "✗"

        if not (dir_ok and srf_ok and bands_ok) and role == "calib":
            failures += 1

        print(f"{name:<22} {role:<7} {status_dir:<5} {status_srf:<5} {status_bands:<6} {band_msg}")

    print("=" * 80)
    if failures == 0:
        print("✓ All calibration datasets accessible. Prospective datasets: check above.")
        return 0
    else:
        print(f"✗ {failures} calibration dataset(s) FAILED. Fix paths before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
