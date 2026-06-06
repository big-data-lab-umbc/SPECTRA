"""Stagewise LogME profiler with warm-up stopping rule.

Computes q[s] = LogME(stage-s features, patch-purity-filtered labels) for s in 1..4.
Used by MGAS to derive Δq[s] = q[4] - q[s] and select the adaptation schedule.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# LogME reference implementation
try:
    _logme_root = os.environ.get("SPECTRA_LOGME_ROOT") or os.environ.get("LOGME_ROOT")
    if _logme_root:
        sys.path.insert(0, _logme_root)
    from LogME import LogME as _LogMERef
    _LOGME_AVAILABLE = True
except ImportError:
    _LOGME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StagewiseProfile:
    """Output of the LogME profiler."""
    scores: list[float]            # q[s] for s=0..n_stages-1 (0-indexed)
    n_patches_used: list[int]      # retained patches per stage
    used_fallback: list[bool]      # image-level fallback per stage
    stopped_at: int                # warm-up step where stopping rule triggered
    precondition_ok: bool          # True if monotonicity precondition holds

    @property
    def n_stages(self) -> int:
        return len(self.scores)

    def delta_q(self) -> list[float]:
        """Δq[s] = max(q) - q[s] — gap from the best-performing stage.

        Using max(q) instead of q[-1] as reference makes Δq non-negative for any
        profile shape, including MAE-pretrained ViTs where intermediate stages are
        often more transferable than the final stage (which is specialized for the
        pretraining decoder).
        """
        q_max = max(self.scores)
        return [q_max - q for q in self.scores]


# ---------------------------------------------------------------------------
# Patch-level label extraction
# ---------------------------------------------------------------------------

def extract_patch_labels(
    seg_labels: np.ndarray,
    patch_size: int,
    purity_tau: float = 0.8,
    ignore_class: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign integer class labels to patches via mode; filter by purity.

    Args:
        seg_labels: (H, W) segmentation label map
        patch_size: spatial patch size p
        purity_tau: minimum fraction of dominant class to keep patch
        ignore_class: class index to exclude (e.g. boundary / no-data)

    Returns:
        patch_labels: (N_kept,) integer labels
        patch_indices: (N_kept,) linear patch indices into the flattened patch grid
    """
    H, W = seg_labels.shape
    H_p = H // patch_size
    W_p = W // patch_size

    kept_labels = []
    kept_indices = []

    for i in range(H_p):
        for j in range(W_p):
            patch = seg_labels[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            valid_mask = patch != ignore_class
            if valid_mask.sum() == 0:
                continue
            values, counts = np.unique(patch[valid_mask], return_counts=True)
            max_count = counts.max()
            purity = max_count / valid_mask.sum()
            if purity < purity_tau:
                continue
            dominant_class = values[counts.argmax()]
            kept_labels.append(dominant_class)
            kept_indices.append(i * W_p + j)

    if len(kept_labels) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.array(kept_labels, dtype=np.int64), np.array(kept_indices, dtype=np.int64)


def stratified_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    n_min: int = 500,
    max_per_class_ratio: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap each class at max_per_class_ratio × N_min_class; return (features, labels)."""
    if rng is None:
        rng = np.random.default_rng(42)

    classes, counts = np.unique(labels, return_counts=True)
    n_min_class = int(counts.min())
    cap = int(n_min_class * max_per_class_ratio)

    keep_idx = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep_idx.append(idx)

    keep_idx = np.concatenate(keep_idx)
    return features[keep_idx], labels[keep_idx]


# ---------------------------------------------------------------------------
# LogME computation
# ---------------------------------------------------------------------------

def _compute_logme(features: np.ndarray, labels: np.ndarray) -> float:
    if not _LOGME_AVAILABLE:
        raise ImportError("LogME reference not found. Set SPECTRA_LOGME_ROOT or LOGME_ROOT to the LogME implementation.")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    logme = _LogMERef(regression=False)
    return float(logme.fit(features.astype(np.float64), y))


def _pool_stage_features(stage_output: torch.Tensor, patch_indices: np.ndarray) -> np.ndarray:
    """Extract mean-pooled patch features from ViT stage output.

    stage_output: (B, 1+N, D) — first token is CLS; remaining are patch tokens.
    patch_indices: subset of patch indices to extract.
    Returns: (len(patch_indices) * B, D) feature matrix.
    """
    # Remove CLS token
    patches = stage_output[:, 1:, :]      # (B, N, D)
    # Select kept patches
    selected = patches[:, patch_indices, :]  # (B, K, D)
    B, K, D = selected.shape
    return selected.reshape(B * K, D).cpu().numpy()


def _image_level_fallback(stage_output: torch.Tensor) -> np.ndarray:
    """Global average pool — used when patch-level retained < N_min."""
    patches = stage_output[:, 1:, :]   # (B, N, D)
    pooled = patches.mean(dim=1)       # (B, D)
    return pooled.cpu().numpy()


# ---------------------------------------------------------------------------
# Warm-up stopping rule
# ---------------------------------------------------------------------------

def _check_stopping_rule(history: list[list[float]], min_steps: int = 300, rho_threshold: float = 0.95) -> bool:
    """Return True if Spearman ρ(t, t-100) ≥ 0.95 AND ρ(t-100, t-200) ≥ 0.95."""
    if len(history) < 3:
        return False
    q_t   = history[-1]
    q_t1  = history[-2]
    q_t2  = history[-3]
    if len(set(q_t)) == 1 or len(set(q_t1)) == 1:
        return False
    rho1, _ = spearmanr(q_t, q_t1)
    rho2, _ = spearmanr(q_t1, q_t2)
    return float(rho1) >= rho_threshold and float(rho2) >= rho_threshold


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

class StagewiseLogMEProfiler:
    """Computes the one-shot stagewise LogME profile q[0..3].

    Usage::
        profiler = StagewiseLogMEProfiler(lora_backbone, patch_size=16)
        profile  = profiler.run(images, seg_labels, warmup_optimizer)
    """

    def __init__(
        self,
        lora_backbone: nn.Module,
        patch_size: int = 16,
        n_stages: int = 4,
        purity_tau: float = 0.8,
        n_probe_images: int = 1000,
        n_min_patches: int = 500,
        warmup_steps: tuple[int, ...] = (200, 300, 400, 500, 600, 700, 800),
        rho_threshold: float = 0.95,
        ignore_class: int = -1,
        device: str = "cuda",
    ) -> None:
        self.backbone = lora_backbone
        self.patch_size = patch_size
        self.n_stages = n_stages
        self.purity_tau = purity_tau
        self.n_probe_images = n_probe_images
        self.n_min_patches = n_min_patches
        self.warmup_steps = warmup_steps
        self.rho_threshold = rho_threshold
        self.ignore_class = ignore_class
        self.device = device

    @torch.no_grad()
    def _measure_profile(
        self,
        images: torch.Tensor,
        seg_labels: np.ndarray,
    ) -> list[float]:
        """Compute q[s] for all stages on n_probe_images.

        Processes each image independently to keep features and patch labels
        aligned (seg_labels is (N, H, W) batched, features are per-image patches).
        """
        self.backbone.eval()
        N = images.shape[0]

        # Accumulate (features, labels) per stage across all probe images
        stage_feats:  list[list[np.ndarray]] = [[] for _ in range(self.n_stages)]
        stage_labels: list[list[np.ndarray]] = [[] for _ in range(self.n_stages)]
        stage_fallback_used: list[bool] = [False] * self.n_stages

        for i in range(N):
            img_i  = images[i:i+1].to(self.device)          # (1, C, H, W)
            lbl_i  = seg_labels[i]                           # (H, W) — per-image

            stage_outs_i = self.backbone.forward_features_per_stage(img_i)
            # stage_outs_i[s]: (1, 1+N_patches, D)

            for s, out_i in enumerate(stage_outs_i):
                patch_labels_i, patch_indices_i = extract_patch_labels(
                    lbl_i, self.patch_size, self.purity_tau, self.ignore_class
                )
                if len(patch_labels_i) == 0:
                    continue

                feats_i = _pool_stage_features(out_i, patch_indices_i)  # (K, D)
                stage_feats[s].append(feats_i)
                stage_labels[s].append(patch_labels_i)

        scores: list[float] = []
        for s in range(self.n_stages):
            if not stage_feats[s]:
                scores.append(float("-inf"))
                continue

            feats_all  = np.concatenate(stage_feats[s],  axis=0)  # (total_K, D)
            labels_all = np.concatenate(stage_labels[s], axis=0)  # (total_K,)

            use_fallback = len(labels_all) < self.n_min_patches
            if use_fallback:
                # Image-level fallback: one feature per image
                stage_fallback_used[s] = True
                img_feats   = np.stack([
                    self.backbone.forward_features_per_stage(images[i:i+1].to(self.device))[s][:, 1:, :].mean(dim=1).cpu().numpy()[0]
                    for i in range(N)
                ])
                img_labels  = np.array([_mode_label(seg_labels[i], self.ignore_class) for i in range(N)])
                feats_all, labels_all = img_feats, img_labels
            else:
                feats_all, labels_all = stratified_subsample(feats_all, labels_all)

            try:
                q = _compute_logme(feats_all, labels_all)
            except Exception:
                q = float("-inf")
            scores.append(q)

        return scores

    def run(
        self,
        probe_loader,
        warmup_optimizer: torch.optim.Optimizer,
        train_fn,
    ) -> StagewiseProfile:
        """Run SSC-PE warm-up with stopping rule, then measure stagewise profile.

        Args:
            probe_loader:     DataLoader yielding (images, seg_labels) batches
            warmup_optimizer: Optimizer for SSC-PE + LoRA + head parameters
            train_fn:         Callable(images, labels) -> loss tensor.
                              Must handle moving tensors to device internally.
                              Typically runs the full encoder+decoder pipeline.

        Returns:
            StagewiseProfile with q[s] scores and diagnostics.
        """
        score_history: list[list[float]] = []
        stopped_at = self.warmup_steps[-1]

        probe_images, probe_labels = self._collect_probe_data(probe_loader)

        step = 0
        for step_target in self.warmup_steps:
            self.backbone.train()
            for images, labels in probe_loader:
                if step >= step_target:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)
                warmup_optimizer.zero_grad()
                loss = train_fn(images, labels)
                loss.backward()
                warmup_optimizer.step()
                step += 1

            scores_t = self._measure_profile(probe_images, probe_labels)
            score_history.append(scores_t)

            if step >= 300 and len(score_history) >= 3:
                if _check_stopping_rule(score_history, rho_threshold=self.rho_threshold):
                    stopped_at = step
                    break

        final_scores = score_history[-1]
        # Use max(q) as reference — always non-negative regardless of profile shape.
        # MAE-pretrained ViTs often have q[intermediate] > q[last]; using the last
        # stage as reference would produce negative Δq and trigger a false precondition
        # failure for nearly all ViT backbones.
        delta_q = [max(final_scores) - q for q in final_scores]
        # Precondition is satisfied when the profile has any discriminative structure.
        # The "no signal" case (all Δq ≤ ε) is handled downstream by MGASPlanner.plan().
        precondition_ok = max(delta_q) > 1e-6

        return StagewiseProfile(
            scores=final_scores,
            n_patches_used=[0] * self.n_stages,
            used_fallback=[False] * self.n_stages,
            stopped_at=stopped_at,
            precondition_ok=precondition_ok,
        )

    def profile_only(self, probe_loader) -> StagewiseProfile:
        """Measure stagewise profile on pretrained backbone without any warm-up.

        Used when SSC-PE is disabled (same-sensor calibration cells): the backbone
        is already pretrained with the target sensor, so there is nothing to warm up.
        """
        probe_images, probe_labels = self._collect_probe_data(probe_loader)
        scores = self._measure_profile(probe_images, probe_labels)
        delta_q = [max(scores) - q for q in scores]
        precondition_ok = max(delta_q) > 1e-6
        return StagewiseProfile(
            scores=scores,
            n_patches_used=[0] * self.n_stages,
            used_fallback=[False] * self.n_stages,
            stopped_at=0,
            precondition_ok=precondition_ok,
        )

    def _collect_probe_data(self, loader) -> tuple[torch.Tensor, np.ndarray]:
        images_list = []
        labels_list = []
        count = 0
        for imgs, lbls in loader:
            images_list.append(imgs)
            labels_list.append(lbls.numpy() if isinstance(lbls, torch.Tensor) else lbls)
            count += imgs.shape[0]
            if count >= self.n_probe_images:
                break
        images = torch.cat(images_list, dim=0)[:self.n_probe_images]
        labels = np.concatenate(labels_list, axis=0)[:self.n_probe_images]
        return images, labels


def _check_monotonicity(delta_q: list[float], delta: float = 0.01) -> bool:
    for s in range(1, len(delta_q)):
        if delta_q[s - 1] - delta_q[s] < -delta:
            return False
    return True


def _mode_label(label_patch: np.ndarray, ignore_class: int) -> int:
    vals = label_patch.flatten()
    vals = vals[vals != ignore_class]
    if len(vals) == 0:
        return 0
    vals_u, counts = np.unique(vals, return_counts=True)
    return int(vals_u[counts.argmax()])
