"""Spectral Response Function (SRF) tables for supported sensors.

Each entry stores per-band: center wavelength (nm), FWHM (nm), and SRF area
(approximated as Gaussian area = FWHM * sqrt(π / (4 * ln2))).

Thermal ABI bands use effective wavelength and placeholder FWHM.
Non-reflective auxiliary bands (DEM, slope) use a sentinel triple (0, 0, 0).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

_GAUSS_AREA_FACTOR = math.sqrt(math.pi / (4 * math.log(2)))  # FWHM → Gaussian integral


@dataclass(frozen=True)
class BandSRF:
    center_nm: float
    fwhm_nm: float
    area: float
    name: str

    @classmethod
    def from_center_fwhm(cls, center_nm: float, fwhm_nm: float, name: str = "") -> "BandSRF":
        return cls(center_nm=center_nm, fwhm_nm=fwhm_nm,
                   area=fwhm_nm * _GAUSS_AREA_FACTOR, name=name)

    def triple(self) -> tuple[float, float, float]:
        return (self.center_nm, self.fwhm_nm, self.area)

    def normalized_triple(self, max_nm: float = 13_300.0, max_fwhm: float = 1500.0, max_area: float = 2000.0) -> tuple[float, float, float]:
        return (self.center_nm / max_nm, self.fwhm_nm / max_fwhm, self.area / max_area)


def _band(c: float, f: float, n: str = "") -> BandSRF:
    return BandSRF.from_center_fwhm(c, f, n)


# ---------------------------------------------------------------------------
# HLS (harmonized Sentinel-2 + Landsat) — 6 bands matching Prithvi pre-train
# ---------------------------------------------------------------------------
HLS_6B: list[BandSRF] = [
    _band(490,  65,  "BLUE"),
    _band(560,  35,  "GREEN"),
    _band(665,  30,  "RED"),
    _band(865,  20,  "NIR_NARROW"),
    _band(1610, 90,  "SWIR_1"),
    _band(2202, 180, "SWIR_2"),
]

# ---------------------------------------------------------------------------
# Sentinel-2 MSI — 13 bands (B1–B12 + B8A)
# ---------------------------------------------------------------------------
S2_13B: list[BandSRF] = [
    _band(443,  20,  "B1_COASTAL"),
    _band(490,  65,  "B2_BLUE"),
    _band(560,  35,  "B3_GREEN"),
    _band(665,  30,  "B4_RED"),
    _band(705,  15,  "B5_RE1"),
    _band(740,  15,  "B6_RE2"),
    _band(783,  20,  "B7_RE3"),
    _band(842,  115, "B8_NIR"),
    _band(865,  20,  "B8A_RE4"),
    _band(945,  20,  "B9_WATER_VAPOR"),
    _band(1375, 30,  "B10_CIRRUS"),
    _band(1610, 90,  "B11_SWIR1"),
    _band(2202, 180, "B12_SWIR2"),
]

# ---------------------------------------------------------------------------
# Sentinel-2 MSI L2A — 12 bands (B10 cirrus absent)
# ---------------------------------------------------------------------------
S2_12B: list[BandSRF] = [b for b in S2_13B if b.name != "B10_CIRRUS"]

# ---------------------------------------------------------------------------
# SatMAE fMoW-Sentinel multispectral pretraining bands.
# Original SatMAE command drops S2 B1, B9, and B10, then groups:
#   [B2, B3, B4, B8], [B5, B6, B7, B8A], [B11, B12]
# ---------------------------------------------------------------------------
SATMAE_S2_10B: list[BandSRF] = [
    S2_13B[1],   # B2 BLUE
    S2_13B[2],   # B3 GREEN
    S2_13B[3],   # B4 RED
    S2_13B[4],   # B5 RE1
    S2_13B[5],   # B6 RE2
    S2_13B[6],   # B7 RE3
    S2_13B[7],   # B8 NIR
    S2_13B[8],   # B8A RE4
    S2_13B[11],  # B11 SWIR1
    S2_13B[12],  # B12 SWIR2
]

# ---------------------------------------------------------------------------
# GOES-R ABI — 16 bands (VIS + NIR + SWIR + MWIR + LWIR + absorption)
# Thermal bands use micrometres → nm conversion; FWHM is approximate.
# ---------------------------------------------------------------------------
ABI_16B: list[BandSRF] = [
    _band(470,    60,   "C01_BLUE"),
    _band(640,    60,   "C02_RED"),
    _band(865,    45,   "C03_NIR"),
    _band(1378,   35,   "C04_CIRRUS"),
    _band(1610,   100,  "C05_SWIR1"),
    _band(2250,   200,  "C06_SWIR2"),
    _band(3900,   800,  "C07_SWIR3"),      # MWIR
    _band(6185,   1000, "C08_WATER"),      # upper troposphere H₂O
    _band(6950,   1000, "C09_WATER"),      # mid-troposphere H₂O
    _band(7340,   700,  "C10_WATER"),      # lower troposphere H₂O
    _band(8500,   500,  "C11_CLOUD_TOP"),
    _band(9610,   500,  "C12_OZONE"),
    _band(10330,  700,  "C13_LWIR"),       # clean window
    _band(11200,  700,  "C14_LWIR"),
    _band(12270,  700,  "C15_CO2"),
    _band(13300,  500,  "C16_CO2"),
]

# ---------------------------------------------------------------------------
# Landslide4Sense — 14 bands (Sentinel-2 L1C + DEM + slope)
# ---------------------------------------------------------------------------
_AUX_BAND = BandSRF(center_nm=0.0, fwhm_nm=0.0, area=0.0, name="AUX")

LANDSLIDE_14B: list[BandSRF] = [
    S2_13B[0],   # B1 COASTAL
    S2_13B[1],   # B2 BLUE
    S2_13B[2],   # B3 GREEN
    S2_13B[3],   # B4 RED
    S2_13B[4],   # B5 RE1
    S2_13B[5],   # B6 RE2
    S2_13B[6],   # B7 RE3
    S2_13B[7],   # B8 NIR
    S2_13B[8],   # B8A RE4
    S2_13B[9],   # B9 WATER_VAPOR
    S2_13B[10],  # B10 CIRRUS
    S2_13B[11],  # B11 SWIR1
    S2_13B[12],  # B12 SWIR2
    _AUX_BAND,   # DEM
]

# ---------------------------------------------------------------------------
# HLS multi-temporal — 18 bands (3 time steps × 6 HLS bands)
# Same spectral response functions repeated per time step; SSC-PE treats each
# band independently so the SRF triples encode only spectral identity, not time.
# ---------------------------------------------------------------------------
HLS_18B: list[BandSRF] = HLS_6B * 3   # T1_B1..T1_B6, T2_B1..T2_B6, T3_B1..T3_B6

# ---------------------------------------------------------------------------
# RGB (LoveDA / GF-2 style aerial) — 3 bands
# ---------------------------------------------------------------------------
RGB_3B: list[BandSRF] = [
    _band(450, 80, "RED"),
    _band(550, 80, "GREEN"),
    _band(650, 80, "BLUE"),
]

# Scale-MAE fMoW-RGB uses standard RGB channel order. These center wavelengths
# drive closest-band selection for multispectral datasets.
FMOW_RGB_3B: list[BandSRF] = [
    _band(665, 80, "RED"),
    _band(560, 80, "GREEN"),
    _band(490, 80, "BLUE"),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SRF_REGISTRY: dict[str, list[BandSRF]] = {
    "hls_6b":        HLS_6B,
    "hls_18b":       HLS_18B,
    "fire_scars":    HLS_6B,
    "multitemporal_crop": HLS_18B,
    "s2_13b":        S2_13B,
    "sen1floods11":  S2_13B,
    "cloudsen12":    S2_13B,
    "s2_12b":        S2_12B,
    "bigearthnet":   S2_12B,
    "geobench_sa_crop_type": S2_12B,
    "satmae_s2_10b": SATMAE_S2_10B,
    "abi_16b":       ABI_16B,
    "abi_cloud":     ABI_16B,
    "landslide_14b": LANDSLIDE_14B,
    "landslide4sense": LANDSLIDE_14B,
    "rgb_3b":        RGB_3B,
    "loveda":        RGB_3B,
    "fmow_rgb":      FMOW_RGB_3B,
}


# ---------------------------------------------------------------------------
# Spyndex variable → approximate center wavelength (nm)
# Variable names follow the awesome-spectral-indices / spyndex convention.
# Only reflective (solar) bands are included; thermal bands have no spyndex names.
# ---------------------------------------------------------------------------
SPYNDEX_WAVELENGTHS_NM: dict[str, float] = {
    "A":    443,    # Coastal Aerosol
    "B":    490,    # Blue
    "G":    560,    # Green
    "R":    665,    # Red
    "RE1":  705,    # Red Edge 1
    "RE2":  740,    # Red Edge 2
    "RE3":  783,    # Red Edge 3
    "N":    842,    # NIR broad (S2 B8)
    "N2":   865,    # NIR narrow (S2 B8A / HLS NIR)
    "WV":   945,    # Water Vapour
    "SWIR": 1375,   # Cirrus / SWIR-cirrus
    "S1":   1610,   # SWIR 1
    "S2":   2190,   # SWIR 2
}

_REFLECTIVE_MAX_NM = 3000.0  # bands above this are thermal — skip for spectral indices


def sensor_to_spyndex_map(sensor_key: str) -> dict[int, str]:
    """Map each reflective band of a sensor to the closest spyndex variable name.

    Returns a dict {band_index: spyndex_var} for bands whose center wavelength
    is > 0 (i.e. not an auxiliary DEM/slope band) and < _REFLECTIVE_MAX_NM.
    Each spyndex variable is assigned to at most one band (closest wins).
    """
    bands = get_srf(sensor_key)
    reflective = {
        i: b.center_nm
        for i, b in enumerate(bands)
        if 0 < b.center_nm < _REFLECTIVE_MAX_NM
    }
    if not reflective:
        return {}

    # For each spyndex variable, find the closest reflective band
    spyndex_to_band: dict[str, tuple[int, float]] = {}  # var → (idx, dist)
    for var, ref_nm in SPYNDEX_WAVELENGTHS_NM.items():
        best_idx, best_dist = min(
            ((i, abs(nm - ref_nm)) for i, nm in reflective.items()),
            key=lambda x: x[1],
        )
        spyndex_to_band[var] = (best_idx, best_dist)

    # Invert: each band gets the spyndex variable it is closest to
    # (a band may be the best match for multiple variables; keep all)
    band_to_spyndex: dict[int, str] = {}
    for var, (idx, _) in spyndex_to_band.items():
        # If multiple variables map to the same band, pick the closest one
        if idx not in band_to_spyndex:
            band_to_spyndex[idx] = var
        else:
            existing_var = band_to_spyndex[idx]
            existing_dist = abs(reflective[idx] - SPYNDEX_WAVELENGTHS_NM[existing_var])
            new_dist = abs(reflective[idx] - SPYNDEX_WAVELENGTHS_NM[var])
            if new_dist < existing_dist:
                band_to_spyndex[idx] = var

    return band_to_spyndex


def select_closest_bands(
    sensor_key: str,
    reference_wavelengths_nm: list[float],
) -> list[int]:
    """Select one band per reference wavelength by nearest-center-wavelength matching.

    Used to build the DEFLECT LR stream: maps the pre-training sensor's N bands
    onto the best-matching N bands in the target sensor.

    Args:
        sensor_key: target sensor (e.g. "abi_cloud", "sen1floods11")
        reference_wavelengths_nm: center wavelengths of the pre-training bands
            (e.g. [490, 560, 665, 865, 1610, 2202] for Prithvi/HLS)

    Returns:
        list of length len(reference_wavelengths_nm) with band indices (may repeat
        if the target sensor has fewer bands than the reference)
    """
    bands = get_srf(sensor_key)
    reflective_indices = [
        i for i, b in enumerate(bands)
        if 0 < b.center_nm < _REFLECTIVE_MAX_NM
    ]
    if not reflective_indices:
        # Fall back to first N bands if no reflective info available
        return list(range(min(len(reference_wavelengths_nm), len(bands))))

    selected = []
    for ref_nm in reference_wavelengths_nm:
        best = min(reflective_indices, key=lambda i: abs(bands[i].center_nm - ref_nm))
        selected.append(best)
    return selected


def get_srf(sensor_key: str) -> list[BandSRF]:
    key = sensor_key.lower()
    if key not in SRF_REGISTRY:
        raise KeyError(f"Unknown sensor '{sensor_key}'. Available: {list(SRF_REGISTRY)}")
    return SRF_REGISTRY[key]


def get_srf_triples(sensor_key: str, normalize: bool = True) -> np.ndarray:
    """Return (C, 3) array of (λ, FWHM, area) triples, optionally normalized."""
    bands = get_srf(sensor_key)
    if normalize:
        triples = [b.normalized_triple() for b in bands]
    else:
        triples = [b.triple() for b in bands]
    return np.array(triples, dtype=np.float32)
