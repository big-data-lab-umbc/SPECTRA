#!/usr/bin/env python3
"""Generate fixed 80-10-10 train/val/test splits from the full dataset.

Each dataset is shuffled once and split proportionally:
  train = 80%   val = 10%   test = remaining (≈10%)

Splits are shared by extract_features.py, compute_metrics.py, and finetune.py.

Dataset sizes and resulting splits:
  fire_scars        :   540 total → 432 / 54 / 54
  sen1floods11      :   446 total → 356 / 44 / 46
  landslide4sense   :  3799 total → 3039 / 379 / 381
  abi_cloud         : 14973 total → 11978 / 1497 / 1498
  multitemporal_crop:  3854 total → 3083 / 385 / 386  (official train+val pooled)

Usage:
    conda run -n geofm4cloud python scripts/generate_splits.py
    python scripts/generate_splits.py --seed 44 --dataset sen1floods11 --tag-seed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

from spectra.data.config import (
    FIXED_SPLITS_DIR, FIRE_SCARS_ROOT, SEN1FLOODS11_HANDLABELED,
    LANDSLIDE4SENSE_ROOT, ABI_CLOUD_ROOT, MULTITEMPORAL_CROP_ROOT,
)

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# test = 1 - TRAIN_RATIO - VAL_RATIO = 0.10

DATASET_ORDER = (
    "fire_scars",
    "sen1floods11",
    "landslide4sense",
    "abi_cloud",
    "multitemporal_crop",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=("all", *DATASET_ORDER), default="all")
    p.add_argument(
        "--tag-seed",
        action="store_true",
        help="Write dataset_s{seed}_{split}.txt instead of overwriting dataset_{split}.txt.",
    )
    return p.parse_args()


def save_split(dataset: str, split: str, items: list, seed_tag: int | None = None) -> None:
    FIXED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    if seed_tag is None:
        path = FIXED_SPLITS_DIR / f"{dataset}_{split}.txt"
    else:
        path = FIXED_SPLITS_DIR / f"{dataset}_s{seed_tag}_{split}.txt"
    path.write_text("\n".join(str(x) for x in items) + "\n")


def split_and_save(dataset: str, items: list, rng: random.Random,
                   seed_tag: int | None = None, save: bool = True) -> None:
    shuffled = list(items)
    rng.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val
    if save:
        save_split(dataset, "train", shuffled[:n_train], seed_tag)
        save_split(dataset, "val",   shuffled[n_train:n_train + n_val], seed_tag)
        save_split(dataset, "test",  shuffled[n_train + n_val:], seed_tag)
        tag = f" s{seed_tag}" if seed_tag is not None else ""
        print(f"  {dataset + tag:<25}: {n:5d} total -> train={n_train}  val={n_val}  test={n_test}")


def collect_items(dataset: str) -> list[str]:
    if dataset == "fire_scars":
        training = FIRE_SCARS_ROOT / "training"
        if not training.exists():
            training = FIRE_SCARS_ROOT
        return [t.name for t in sorted(training.glob("*_merged.tif"))]

    if dataset == "sen1floods11":
        s2hand = SEN1FLOODS11_HANDLABELED / "S2Hand"
        return [t.name for t in sorted(s2hand.glob("*.tif"))]

    if dataset == "landslide4sense":
        img_dir = LANDSLIDE4SENSE_ROOT / "images" / "train"
        return [h.name for h in sorted(img_dir.glob("image_*.h5"))]

    if dataset == "abi_cloud":
        return [f.name for f in sorted(ABI_CLOUD_ROOT.glob("*.npz"))]

    if dataset == "multitemporal_crop":
        train_txt = MULTITEMPORAL_CROP_ROOT / "training_data.txt"
        val_txt   = MULTITEMPORAL_CROP_ROOT / "validation_data.txt"
        if train_txt.exists() and val_txt.exists():
            train_ids = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()]
            val_ids   = [l.strip() for l in val_txt.read_text().splitlines() if l.strip()]
            return train_ids + val_ids
        print("  multitemporal_crop: official split files not found — skipping")
        return []

    raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    seed_tag = args.seed if args.tag_seed else None
    print(f"Generating 80-10-10 splits (seed={args.seed}) -> {FIXED_SPLITS_DIR}")

    for dataset in DATASET_ORDER:
        items = collect_items(dataset)
        if not items:
            continue
        split_and_save(
            dataset,
            items,
            rng,
            seed_tag=seed_tag,
            save=(args.dataset in ("all", dataset)),
        )

    print("Done.")


if __name__ == "__main__":
    main()
