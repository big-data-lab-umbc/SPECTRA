"""Band preprocessing utilities: normalization, resizing, PCA projection, sampling."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple, TypeVar

T = TypeVar("T")


def normalize_multiband_image(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-band min-max normalization to [0, 1]. img: (H, W, C)."""
    img = img.astype(np.float32)
    mn = img.min(axis=(0, 1), keepdims=True)
    mx = img.max(axis=(0, 1), keepdims=True)
    return (img - mn) / (mx - mn + eps)


def resize_channels_last_image(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize (H, W, C) image to (target_size, target_size, C) via area interpolation."""
    import cv2
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)


def sample_sequence(seq: list[T], n: int) -> list[T]:
    """Return evenly spaced n elements from seq, or all elements if len(seq) <= n."""
    if len(seq) <= n:
        return seq
    indices = np.linspace(0, len(seq) - 1, n, dtype=int)
    return [seq[i] for i in indices]


class PCAProjector:
    """Incrementally fits PCA on pixel vectors and projects multi-band images."""

    def __init__(self, n_components: int = 6, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.pca: Optional[PCA] = None

    def fit(self, images: np.ndarray) -> "PCAProjector":
        """images: (N, H, W, C) or (N, C, H, W)."""
        images = _ensure_channels_last(images)
        flat = images.reshape(-1, images.shape[-1])
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(flat.astype(np.float64))
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Call fit() first.")
        images = _ensure_channels_last(images)
        shape = images.shape
        projected = self.pca.transform(images.reshape(-1, shape[-1]).astype(np.float64))
        return projected.reshape(*shape[:-1], self.n_components).astype(np.float32)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        self.fit(images)
        return self.transform(images)


def apply_pca_projection(
    images: np.ndarray,
    n_components: int = 6,
    pca: Optional[PCA] = None,
) -> Tuple[np.ndarray, PCA]:
    """Stateless helper: fit or reuse a PCA and return projected images + fitted PCA."""
    images = _ensure_channels_last(images)
    shape = images.shape
    flat = images.reshape(-1, shape[-1]).astype(np.float64)
    if pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        projected = pca.fit_transform(flat)
    else:
        projected = pca.transform(flat)
    return projected.reshape(*shape[:-1], n_components).astype(np.float32), pca


def _ensure_channels_last(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] < images.shape[-1]:
        return np.transpose(images, (0, 2, 3, 1))
    return images
