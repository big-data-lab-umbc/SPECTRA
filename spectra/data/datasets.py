"""Dataset loaders for SPECTRA calibration datasets.

Each dataset class:
  - Reads a fixed split file listing filenames/IDs
  - Returns (image, label) tensors of consistent spatial size (crop/resize to patch_size multiple)
  - Images: float32 in [0, 1], shape (C, H, W)
  - Labels: int64, shape (H, W), values in {0, …, n_classes-1}, -1 = ignore

Supported:
    FireScarsDataset         — HLS GeoTIFF 6B, binary semseg
    Sen1Floods11Dataset      — S2 GeoTIFF 13B, binary semseg
    MultiTemporalCropDataset — HLS×3T GeoTIFF 6B (time-step 2 only), 14-class semseg
    ABICloudDataset          — GOES ABI npz 16B, 4-class semseg
    Landslide4SenseDataset   — S2+DEM HDF5 14B, binary semseg (placeholder until labels fixed)

Usage::
    from spectra.data.datasets import build_dataset
    ds = build_dataset("fire_scars", split="train", crop_size=512)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import json
import logging
import random
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from spectra.data.config import (
    FIXED_SPLITS_DIR,
    FIRE_SCARS_ROOT,
    SEN1FLOODS11_HANDLABELED,
    MULTITEMPORAL_CROP_ROOT,
    ABI_CLOUD_ROOT,
    LANDSLIDE4SENSE_ROOT,
    CLOUDSEN12_ROOT,
    LOVEDA_ROOT,
    GEOBENCH_SA_CROP_TYPE_ROOT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_split(dataset: str, split: str, split_seed: int | None = None) -> list[str]:
    if split_seed is None or split_seed == 42:
        path = FIXED_SPLITS_DIR / f"{dataset}_{split}.txt"
    else:
        path = FIXED_SPLITS_DIR / f"{dataset}_s{split_seed}_{split}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run scripts/generate_splits.py first."
        )
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _read_tif(path: Path) -> np.ndarray:
    """Read a GeoTIFF → (C, H, W) float32 array, NaN → 0."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # (C, H, W)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _normalize_percentile(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Per-band percentile normalization to [0, 1]."""
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        band = img[c]
        p_lo = np.percentile(band, lo)
        p_hi = np.percentile(band, hi)
        denom = max(p_hi - p_lo, 1e-6)
        out[c] = np.clip((band - p_lo) / denom, 0.0, 1.0)
    return out


def _random_crop(img: np.ndarray, lbl: np.ndarray, size: int):
    """Random square crop of (C,H,W) image and (H,W) label."""
    _, H, W = img.shape
    if H <= size and W <= size:
        return img, lbl
    top  = np.random.randint(0, max(H - size, 1))
    left = np.random.randint(0, max(W - size, 1))
    return img[:, top:top+size, left:left+size], lbl[top:top+size, left:left+size]


def _center_crop(img: np.ndarray, lbl: np.ndarray, size: int):
    """Centre crop to size×size."""
    _, H, W = img.shape
    top  = max((H - size) // 2, 0)
    left = max((W - size) // 2, 0)
    return img[:, top:top+size, left:left+size], lbl[top:top+size, left:left+size]


def _pad_to_multiple(img: np.ndarray, lbl: np.ndarray, multiple: int = 14):
    """Pad image and label so H and W are multiples of `multiple`."""
    _, H, W = img.shape
    pH = (multiple - H % multiple) % multiple
    pW = (multiple - W % multiple) % multiple
    if pH == 0 and pW == 0:
        return img, lbl
    img = np.pad(img, ((0,0),(0,pH),(0,pW)), mode="reflect")
    lbl = np.pad(lbl, ((0,pH),(0,pW)), mode="constant", constant_values=-1)
    return img, lbl


# ---------------------------------------------------------------------------
# FireScarsDataset
# ---------------------------------------------------------------------------

class FireScarsDataset(Dataset):
    """HLS 6-band burn scar segmentation.

    Images: *_merged.tif (C=6, 512×512)
    Labels: *.mask.tif   (C=1, 512×512), values 0=background 1=burn
    """

    def __init__(self, split: str = "train", crop_size: int = 224,
                 augment: bool = True, split_seed: int | None = None,
                 pad_multiple: int = 14) -> None:
        self.split    = split
        self.crop_size = crop_size
        self.augment  = augment and split == "train"
        self.split_seed = split_seed
        self.pad_multiple = pad_multiple

        filenames = _read_split("fire_scars", split, split_seed)
        img_dir   = FIRE_SCARS_ROOT / "training"

        self.samples: list[tuple[Path, Path]] = []
        for fname in filenames:
            img_path = img_dir / fname
            lbl_path = img_dir / fname.replace("_merged.tif", ".mask.tif")
            if img_path.exists() and lbl_path.exists():
                self.samples.append((img_path, lbl_path))
            else:
                logger.warning("Missing: %s or %s", img_path, lbl_path)
        logger.info("FireScars [%s]: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img = _normalize_percentile(_read_tif(img_path))   # (6, H, W)
        lbl = _read_tif(lbl_path)[0].astype(np.int64)      # (H, W)

        if self.augment:
            img, lbl = _random_crop(img, lbl, self.crop_size)
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
        else:
            img, lbl = _center_crop(img, lbl, self.crop_size)

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# Sen1Floods11Dataset
# ---------------------------------------------------------------------------

class Sen1Floods11Dataset(Dataset):
    """Sentinel-2 13-band flood segmentation (hand-labeled split).

    Images: S2Hand/{region}_{id}_S2Hand.tif   (C=13, variable size)
    Labels: LabelHand/{region}_{id}_LabelHand.tif (C=1), 0=no flood 1=flood -1=ignore
    """

    def __init__(
        self,
        split: str = "train",
        crop_size: int = 224,
        augment: bool = True,
        split_seed: int | None = None,
        positive_crop_max_tries: int = 0,
        pad_multiple: int = 14,
    ) -> None:
        self.split     = split
        self.crop_size = crop_size
        self.augment   = augment and split == "train"
        self.split_seed = split_seed
        self.positive_crop_max_tries = max(0, int(positive_crop_max_tries))
        self.pad_multiple = pad_multiple

        filenames = _read_split("sen1floods11", split, split_seed)
        s2_dir    = SEN1FLOODS11_HANDLABELED / "S2Hand"
        lbl_dir   = SEN1FLOODS11_HANDLABELED / "LabelHand"

        self.samples: list[tuple[Path, Path]] = []
        for fname in filenames:
            img_path = s2_dir  / fname
            lbl_path = lbl_dir / fname.replace("_S2Hand.tif", "_LabelHand.tif")
            if img_path.exists() and lbl_path.exists():
                self.samples.append((img_path, lbl_path))
            else:
                logger.warning("Missing: %s or %s", img_path, lbl_path)
        logger.info("Sen1Floods11 [%s]: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img = _normalize_percentile(_read_tif(img_path))   # (13, H, W)
        lbl = _read_tif(lbl_path)[0].astype(np.int64)      # (H, W)

        # Remap: -1 → ignore (-1), 0 → 0, 1 → 1
        lbl = np.where(lbl < 0, -1, lbl)

        if self.augment:
            if self.positive_crop_max_tries > 0 and np.any(lbl == 1):
                crop_img, crop_lbl = _random_crop(img, lbl, self.crop_size)
                for _ in range(self.positive_crop_max_tries):
                    if np.any(crop_lbl == 1):
                        break
                    crop_img, crop_lbl = _random_crop(img, lbl, self.crop_size)
                img, lbl = crop_img, crop_lbl
            else:
                img, lbl = _random_crop(img, lbl, self.crop_size)
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
        else:
            img, lbl = _center_crop(img, lbl, self.crop_size)

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# MultiTemporalCropDataset
# ---------------------------------------------------------------------------

class MultiTemporalCropDataset(Dataset):
    """HLS multi-temporal (3 time steps × 6 bands = 18 bands) crop type segmentation.

    We use all 18 bands (time steps concatenated) to match FULL_BAND_CONFIGS.
    Files: training_chips/{chip_id}_merged.tif  (C=18, variable size)
           training_chips/{chip_id}.mask.tif    (C=1), 14-class crop type map
    """

    def __init__(self, split: str = "train", crop_size: int = 224,
                 augment: bool = True, split_seed: int | None = None,
                 pad_multiple: int = 14) -> None:
        self.split     = split
        self.crop_size = crop_size
        self.augment   = augment and split == "train"
        self.split_seed = split_seed
        self.pad_multiple = pad_multiple

        chip_ids = _read_split("multitemporal_crop", split, split_seed)
        chips_dir  = MULTITEMPORAL_CROP_ROOT / "training_chips"
        val_dir    = MULTITEMPORAL_CROP_ROOT / "validation" / "validation_chips"

        self.samples: list[tuple[Path, Path]] = []
        for chip_id in chip_ids:
            for base_dir in [chips_dir, val_dir]:
                img_path = base_dir / f"{chip_id}_merged.tif"
                lbl_path = base_dir / f"{chip_id}.mask.tif"
                if img_path.exists() and lbl_path.exists():
                    self.samples.append((img_path, lbl_path))
                    break
            else:
                logger.warning("Missing chip: %s", chip_id)
        logger.info("MultiTemporalCrop [%s]: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img = _normalize_percentile(_read_tif(img_path))   # (18, H, W)
        lbl = _read_tif(lbl_path)[0].astype(np.int64)      # (H, W)

        # Class 0 = background/no-data → remap to ignore
        lbl = np.where(lbl == 0, -1, lbl - 1)   # 1–14 → 0–13, 0 → -1

        if self.augment:
            img, lbl = _random_crop(img, lbl, self.crop_size)
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
        else:
            img, lbl = _center_crop(img, lbl, self.crop_size)

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# ABICloudDataset
# ---------------------------------------------------------------------------

_ABI_PHASE_TO_CLASS = {0: -1, 1: 0, 2: 1, 3: 2, 6: 3}  # cloud phase → class index
# 0=undetermined(ignore), 1=liquid(0), 2=supercooled(1), 3=mixed(2), 6=ice(3)


class ABICloudDataset(Dataset):
    """GOES-16 ABI 16-band cloud phase segmentation.

    Files: {name}.npz with keys:
        rad      (128, 128, 16) float32 — radiance all 16 ABI bands
        l2_cloud_phase (128, 128) float16 — cloud phase class
    """

    def __init__(self, split: str = "train", augment: bool = True,
                 split_seed: int | None = None, pad_multiple: int = 14) -> None:
        self.split   = split
        self.augment = augment and split == "train"
        self.split_seed = split_seed
        self.pad_multiple = pad_multiple

        filenames = _read_split("abi_cloud", split, split_seed)
        self.paths = []
        for fname in filenames:
            p = ABI_CLOUD_ROOT / fname
            if p.exists():
                self.paths.append(p)
            else:
                logger.warning("Missing: %s", p)
        logger.info("ABICloud [%s]: %d samples", split, len(self.paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        data = np.load(self.paths[idx])
        rad  = data["rad"].astype(np.float32)           # (128, 128, 16)
        img  = rad.transpose(2, 0, 1)                   # (16, 128, 128)
        img  = _normalize_percentile(img)

        phase = data["l2_cloud_phase"].astype(np.float32)  # (128, 128)
        lbl   = np.full(phase.shape, -1, dtype=np.int64)
        for raw, cls in _ABI_PHASE_TO_CLASS.items():
            lbl[phase == raw] = cls

        if self.augment and np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# Landslide4SenseDataset
# ---------------------------------------------------------------------------

class Landslide4SenseDataset(Dataset):
    """Sentinel-2 + DEM 14-band landslide segmentation (HDF5).

    Images:      images/{train|validation}/image_{id}.h5  key='img' (128,128,14)
    Annotations: annotations/{train|validation}/image_{id}.h5  key='mask' (128,128)
                 Values: 0=no landslide, 1=landslide

    Images:      images/{train|validation}/image_{id}.h5    key='img'  (128,128,14)
    Annotations: annotations/{train|validation}/mask_{id}.h5 key='mask' (128,128) uint8
                 Values: 0=no landslide, 1=landslide
    """

    def __init__(self, split: str = "train", augment: bool = True,
                 split_seed: int | None = None, pad_multiple: int = 14) -> None:
        import h5py
        self.split   = split
        self.augment = augment and split == "train"
        self._h5py   = h5py
        self.split_seed = split_seed
        self.pad_multiple = pad_multiple

        filenames = _read_split("landslide4sense", split, split_seed)
        split_dir = "train" if split != "validation" else "validation"
        img_dir = LANDSLIDE4SENSE_ROOT / "images"      / split_dir
        ann_dir = LANDSLIDE4SENSE_ROOT / "annotations" / split_dir

        self.samples: list[tuple[Path, Path]] = []
        for fname in filenames:
            # fname = "image_{id}.h5" → mask file = "mask_{id}.h5"
            mask_fname = fname.replace("image_", "mask_")
            img_path = img_dir / fname
            ann_path = ann_dir / mask_fname
            if img_path.exists() and ann_path.exists():
                self.samples.append((img_path, ann_path))
            else:
                logger.warning("Missing: %s or %s", img_path, ann_path)

        logger.info("Landslide4Sense [%s]: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]

        with self._h5py.File(img_path) as hf:
            img = hf["img"][:].astype(np.float32)   # (128, 128, 14)
        img = img.transpose(2, 0, 1)                 # (14, 128, 128)
        img = _normalize_percentile(img)

        with self._h5py.File(ann_path) as hf:
            lbl = hf["mask"][:].astype(np.int64)    # (128, 128)

        if self.augment and np.random.rand() > 0.5:
            img = img[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# GeoBenchSACropTypeDataset
# ---------------------------------------------------------------------------

class GeoBenchSACropTypeDataset(Dataset):
    """GEO-Bench m-SA-crop-type semantic segmentation.

    The downloaded archive stores 5000 HDF5 chips with 12 real Sentinel-2
    spectral bands, an all-zero Cloud Probability channel, and a 10-value label.
    This loader keeps the official GEO-Bench label space 0..9. It drops Cloud
    Probability and uses the official GEO-Bench train/valid/test partition.
    """

    _SPLIT_MAP = {"train": "train", "val": "valid", "valid": "valid", "validation": "valid", "test": "test"}
    _BAND_KEYS = (
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "11 - SWIR",
        "12 - SWIR",
    )

    def __init__(
        self,
        split: str = "train",
        crop_size: int = 224,
        augment: bool = True,
        split_seed: int | None = None,
        pad_multiple: int = 14,
    ) -> None:
        import h5py

        self.split = split
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        self.split_seed = split_seed
        self.pad_multiple = pad_multiple
        self.root = GEOBENCH_SA_CROP_TYPE_ROOT
        self.data_dir = self.root / "data"
        self._h5py = h5py

        partition = self._load_partition()
        split_key = self._SPLIT_MAP.get(split, split)
        if split_key not in partition:
            raise ValueError(f"Unknown GEO-Bench SA crop split {split!r}; available: {list(partition)}")

        self.samples: list[tuple[Path, Path]] = []
        missing = 0
        for sample_id in partition[split_key]:
            sample_path = self.data_dir / f"{sample_id}.hdf5"
            if sample_path.exists():
                self.samples.append((sample_path, sample_path))
            else:
                missing += 1
        if missing:
            logger.warning(
                "GeoBenchSACropType [%s]: missing %d HDF5 files under %s",
                split,
                missing,
                self.data_dir,
            )
        logger.info("GeoBenchSACropType [%s]: %d samples", split, len(self.samples))

    def _load_partition(self) -> dict[str, list[str]]:
        partition_path = self.root / "default_partition.json"
        if partition_path.exists():
            return json.loads(partition_path.read_text())

        archive_path = self.root / "files-archive"
        if archive_path.exists():
            with zipfile.ZipFile(archive_path) as zf:
                return json.loads(zf.read("default_partition.json"))

        raise FileNotFoundError(
            f"Could not find {partition_path} or {archive_path}; GEO-Bench SA crop dataset is not available."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_path, _ = self.samples[idx]
        with self._h5py.File(sample_path, "r") as hf:
            img = np.stack([hf[key][:].astype(np.float32) for key in self._BAND_KEYS], axis=0)
            lbl = hf["label"][:].astype(np.int64)

        lbl = lbl.astype(np.int64)
        img = _normalize_percentile(img)

        if self.augment:
            img, lbl = _random_crop(img, lbl, self.crop_size)
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
        else:
            img, lbl = _center_crop(img, lbl, self.crop_size)

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# CloudSEN12Dataset
# ---------------------------------------------------------------------------

class CloudSEN12Dataset(Dataset):
    """CloudSEN12 L2A cloud segmentation via TACO/TORTILLA format.

    Reads all cloudsen12-l2a.*.part.taco files from CLOUDSEN12_ROOT.
    Only uses samples with label_type='high' (pixel-wise expert labels).
    Uses the dataset's built-in tortilla:data_split for train/validation/test.

    Images: 13 S2 L2A reflectance bands (first 13 of 14), uint16 → [0,1]
    Labels: 0=clear, 1=thick cloud, 2=thin cloud, 3=cloud shadow
    Image size: 2048×2048 → random/center crop to crop_size
    """

    _TACO_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test", "validation": "validation"}

    def __init__(self, split: str = "train", crop_size: int = 224,
                 augment: bool = True, pad_multiple: int = 14) -> None:
        import tacoreader
        import glob
        self.split     = split
        self.crop_size = crop_size
        self.augment   = augment and split == "train"
        self.pad_multiple = pad_multiple

        taco_split = self._TACO_SPLIT_MAP.get(split, split)
        taco_files = sorted(glob.glob(str(CLOUDSEN12_ROOT / "cloudsen12-l2a.*.part.taco")))
        if not taco_files:
            raise FileNotFoundError(f"No cloudsen12-l2a.*.part.taco files found at {CLOUDSEN12_ROOT}")

        # Pre-extract vsisubfile paths — cached to disk to avoid 70s repeated init
        import pickle, hashlib
        cache_key  = hashlib.md5((taco_split + "".join(taco_files)).encode()).hexdigest()[:12]
        cache_file = Path(taco_files[0]).parent / f".cloudsen12_index_{cache_key}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                self.index = pickle.load(f)
        else:
            # Use positional (iloc) indices via np.where; df.read() internally uses iloc.
            self.index: list[tuple[str, str]] = []   # (img_vsi_path, lbl_vsi_path)
            for taco_path in taco_files:
                df = tacoreader.load(taco_path)
                mask = (df["label_type"] == "high") & (df["tortilla:data_split"] == taco_split)
                positions = np.where(mask.values)[0]
                for pos in positions:
                    sample   = df.read(int(pos))
                    self.index.append((sample.read(0), sample.read(1)))
            with open(cache_file, "wb") as f:
                pickle.dump(self.index, f)

        logger.info("CloudSEN12 [%s]: %d high-quality samples", split, len(self.index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        import rasterio
        img_path, lbl_path = self.index[idx]

        with rasterio.open(img_path) as src:
            img = src.read()[:13].astype(np.float32)   # (13, 2048, 2048) uint16
        img = _normalize_percentile(img)

        with rasterio.open(lbl_path) as src:
            lbl = src.read(1).astype(np.int64)          # (2048, 2048) 0-3

        if self.augment:
            img, lbl = _random_crop(img, lbl, self.crop_size)
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
        else:
            img, lbl = _center_crop(img, lbl, self.crop_size)

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# LoveDADataset
# ---------------------------------------------------------------------------

class LoveDADataset(Dataset):
    """LoveDA aerial RGB segmentation (Rural + Urban domains).

    Directory layout:
        {LOVEDA_ROOT}/{split_dir}/{domain}/images_png/{id}.png
        {LOVEDA_ROOT}/{split_dir}/{domain}/masks_png/{id}.png

    split_dir: Train (train), Val (val/test)
    domains:   Rural, Urban

    Images: RGB (3, 1024, 1024), uint8 → [0,1]
    Labels: 1–7 → 0–6, 0 → -1 (ignore). 7 classes:
        0=Background, 1=Building, 2=Road, 3=Water,
        4=Barren, 5=Forest, 6=Agricultural
    Image size: 1024×1024 → crop to crop_size
    """

    def __init__(self, split: str = "train", crop_size: int = 224,
                 augment: bool = True, pad_multiple: int = 14) -> None:
        self.split     = split
        self.crop_size = crop_size
        self.augment   = augment and split == "train"
        self.pad_multiple = pad_multiple

        split_dir = "Train" if split == "train" else "Val"
        domains   = ["Rural", "Urban"]

        self.samples: list[tuple[Path, Path]] = []
        for domain in domains:
            img_dir = LOVEDA_ROOT / split_dir / domain / "images_png"
            msk_dir = LOVEDA_ROOT / split_dir / domain / "masks_png"
            if not img_dir.exists():
                logger.warning("LoveDA: %s not found", img_dir)
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                msk_path = msk_dir / img_path.name
                if msk_path.exists():
                    self.samples.append((img_path, msk_path))
                else:
                    logger.warning("Missing mask: %s", msk_path)

        logger.info("LoveDA [%s]: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        img_path, msk_path = self.samples[idx]

        img = np.array(Image.open(img_path)).astype(np.float32) / 255.0  # (H, W, 3)
        img = img.transpose(2, 0, 1)                                       # (3, H, W)

        msk = np.array(Image.open(msk_path)).astype(np.int64)             # (H, W) 1-7 or 0
        lbl = np.where(msk == 0, -1, msk - 1)                             # 1-7→0-6, 0→-1

        if self.augment:
            img, lbl = _random_crop(img, lbl, self.crop_size)
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
        else:
            img, lbl = _center_crop(img, lbl, self.crop_size)

        img, lbl = _pad_to_multiple(img, lbl, self.pad_multiple)
        return torch.from_numpy(img), torch.from_numpy(lbl)


# ---------------------------------------------------------------------------
# Registry + build_dataset()
# ---------------------------------------------------------------------------

_DATASET_REGISTRY = {
    "fire_scars":         FireScarsDataset,
    "sen1floods11":       Sen1Floods11Dataset,
    "multitemporal_crop": MultiTemporalCropDataset,
    "abi_cloud":          ABICloudDataset,
    "landslide4sense":    Landslide4SenseDataset,
    "geobench_sa_crop_type": GeoBenchSACropTypeDataset,
    "cloudsen12":         CloudSEN12Dataset,
    "loveda":             LoveDADataset,
}


def build_dataset(
    dataset_name: str,
    split: str,
    crop_size: int = 224,
    augment: bool = True,
    split_seed: int | None = None,
    positive_crop_max_tries: int = 0,
    pad_multiple: int = 14,
) -> Dataset:
    cls = _DATASET_REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(_DATASET_REGISTRY)}"
        )
    kwargs: dict = {"split": split}
    if dataset_name not in ("abi_cloud", "landslide4sense"):
        kwargs["crop_size"] = crop_size
    kwargs["pad_multiple"] = pad_multiple
    kwargs["augment"] = augment
    if dataset_name in {
        "fire_scars",
        "sen1floods11",
        "multitemporal_crop",
        "abi_cloud",
        "landslide4sense",
        "geobench_sa_crop_type",
    }:
        kwargs["split_seed"] = split_seed
    if dataset_name == "sen1floods11":
        kwargs["positive_crop_max_tries"] = positive_crop_max_tries if split == "train" else 0
    return cls(**kwargs)  # type: ignore[call-arg]


class PositiveBalancedBatchSampler(Sampler[list[int]]):
    """Batch sampler that oversamples positive-source samples into every batch."""

    def __init__(
        self,
        positive_indices: list[int],
        negative_indices: list[int],
        dataset_size: int,
        batch_size: int,
        min_positive: int,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.positive_indices = list(positive_indices)
        self.negative_indices = list(negative_indices)
        self.all_indices = list(range(dataset_size))
        self.batch_size = int(batch_size)
        self.min_positive = min(max(1, int(min_positive)), self.batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.n_batches = dataset_size // self.batch_size if drop_last else int(np.ceil(dataset_size / self.batch_size))
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        neg_pool = self.negative_indices or self.all_indices
        for _ in range(self.n_batches):
            batch: list[int] = []
            for _ in range(self.min_positive):
                batch.append(self.positive_indices[rng.randrange(len(self.positive_indices))])
            n_remaining = self.batch_size - len(batch)
            for _ in range(n_remaining):
                batch.append(neg_pool[rng.randrange(len(neg_pool))])
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


def _positive_source_indices(ds: Dataset, positive_class: int = 1) -> tuple[list[int], list[int]]:
    """Classify samples by whether their full label image contains positive_class."""
    samples = getattr(ds, "samples", None)
    if samples is None:
        raise ValueError("positive-balanced sampling requires datasets with a .samples list")

    positive: list[int] = []
    negative: list[int] = []
    for idx, sample in enumerate(samples):
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError("positive-balanced sampling expects samples as (image_path, label_path)")
        label_path = sample[1]
        lbl = _read_tif(Path(label_path))[0].astype(np.int64)
        if np.any(lbl == positive_class):
            positive.append(idx)
        else:
            negative.append(idx)
    return positive, negative


def build_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = 224,
    seed: int = 42,
    split_seed: int | None = None,
    positive_batch_min: int = 0,
    positive_crop_max_tries: int = 0,
    pad_multiple: int = 14,
) -> DataLoader:
    augment = split == "train"
    ds = build_dataset(
        dataset_name,
        split,
        crop_size=crop_size,
        augment=augment,
        split_seed=split_seed,
        positive_crop_max_tries=positive_crop_max_tries,
        pad_multiple=pad_multiple,
    )

    def worker_init(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if split == "train" and positive_batch_min > 0:
        positive_indices, negative_indices = _positive_source_indices(ds)
        if positive_indices:
            min_positive = min(int(positive_batch_min), int(batch_size))
            sampler = PositiveBalancedBatchSampler(
                positive_indices,
                negative_indices,
                dataset_size=len(ds),
                batch_size=batch_size,
                min_positive=min_positive,
                seed=seed,
                drop_last=True,
            )
            logger.info(
                "PositiveBalancedBatchSampler [%s/%s]: batch_size=%d min_positive=%d positives=%d negatives=%d batches=%d crop_retries=%d",
                dataset_name,
                split,
                batch_size,
                min_positive,
                len(positive_indices),
                len(negative_indices),
                len(sampler),
                positive_crop_max_tries,
            )
            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=worker_init,
                generator=generator,
            )
        logger.warning(
            "positive_batch_min=%d requested for %s/%s, but no positive-source samples were found; falling back to shuffle=True",
            positive_batch_min,
            dataset_name,
            split,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        worker_init_fn=worker_init,
        generator=generator,
    )
