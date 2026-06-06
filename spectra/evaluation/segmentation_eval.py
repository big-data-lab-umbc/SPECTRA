"""Segmentation evaluation metrics and visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _unwrap_outputs(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "output"):
        return outputs.output
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def evaluate_segmentation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    foreground_class: int = 1,
    ignore_index: int = -1,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate semantic segmentation with pooled pixel and image-level metrics."""
    model.eval()

    total_inter: torch.Tensor | None = None
    total_union: torch.Tensor | None = None
    pred_counts: torch.Tensor | None = None
    label_counts: torch.Tensor | None = None
    logit_sums: torch.Tensor | None = None
    logit_count = 0

    tp = fp = fn = tn = 0
    image_count = 0
    positive_image_count = 0
    background_image_count = 0
    predicted_positive_image_count = 0
    detected_positive_images = 0
    false_alarm_background_images = 0
    per_image_fg_iou: list[float] = []
    per_image_fg_dice: list[float] = []
    loss_sum = 0.0
    loss_batches = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = _unwrap_outputs(model(images))
            if criterion is not None:
                loss_sum += float(criterion(outputs, labels).item())
                loss_batches += 1

            preds = outputs.argmax(dim=1)
            n_classes = int(outputs.shape[1])
            valid = labels != ignore_index
            if not bool(valid.any()):
                continue

            if total_inter is None:
                total_inter = torch.zeros(n_classes, device=device)
                total_union = torch.zeros(n_classes, device=device)
                pred_counts = torch.zeros(n_classes, device=device)
                label_counts = torch.zeros(n_classes, device=device)
                logit_sums = torch.zeros(n_classes, device=device)

            preds_m = preds[valid]
            labels_m = labels[valid]
            logits_valid = outputs.permute(0, 2, 3, 1)[valid]
            logit_sums += logits_valid.sum(dim=0)
            logit_count += int(logits_valid.shape[0])

            for cls in range(n_classes):
                pred_c = preds_m == cls
                label_c = labels_m == cls
                total_inter[cls] += (pred_c & label_c).sum()
                total_union[cls] += (pred_c | label_c).sum()
                pred_counts[cls] += pred_c.sum()
                label_counts[cls] += label_c.sum()

            pred_fg = preds_m == foreground_class
            true_fg = labels_m == foreground_class
            tp += int((pred_fg & true_fg).sum().item())
            fp += int((pred_fg & ~true_fg).sum().item())
            fn += int((~pred_fg & true_fg).sum().item())
            tn += int((~pred_fg & ~true_fg).sum().item())

            for b in range(labels.shape[0]):
                valid_b = labels[b] != ignore_index
                if not bool(valid_b.any()):
                    continue
                pred_b = preds[b][valid_b] == foreground_class
                true_b = labels[b][valid_b] == foreground_class
                has_true = bool(true_b.any())
                has_pred = bool(pred_b.any())

                image_count += 1
                positive_image_count += int(has_true)
                background_image_count += int(not has_true)
                predicted_positive_image_count += int(has_pred)
                detected_positive_images += int(has_true and has_pred)
                false_alarm_background_images += int((not has_true) and has_pred)

                inter = int((pred_b & true_b).sum().item())
                pred_sum = int(pred_b.sum().item())
                true_sum = int(true_b.sum().item())
                union = pred_sum + true_sum - inter
                denom = pred_sum + true_sum
                if union > 0:
                    per_image_fg_iou.append(float(inter / union))
                if denom > 0:
                    per_image_fg_dice.append(float((2.0 * inter) / denom))

    if total_inter is None or total_union is None or pred_counts is None or label_counts is None:
        return {
            "miou": 0.0,
            "macro_iou": 0.0,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "mean_class_accuracy": None,
            "per_class_iou": [],
            "per_class_precision": [],
            "per_class_recall": [],
            "per_class_f1": [],
            "foreground_class": foreground_class,
            "foreground_iou": None,
            "foreground_dice": None,
            "pixel_accuracy": None,
            "pred_counts": [],
            "label_counts": [],
            "pred_class_ratios": [],
            "label_class_ratios": [],
            "class_area_bias": [],
            "loss": None,
        }

    iou = total_inter / (total_union + 1e-6)
    present = total_union > 0
    miou = float(iou[present].mean().item()) if bool(present.any()) else 0.0
    class_present = label_counts > 0
    per_class_precision_t = total_inter / pred_counts.clamp(min=1)
    per_class_recall_t = total_inter / label_counts.clamp(min=1)
    per_class_f1_t = (2.0 * total_inter) / (pred_counts + label_counts).clamp(min=1)
    macro_precision = (
        float(per_class_precision_t[class_present].mean().item())
        if bool(class_present.any()) else None
    )
    macro_recall = (
        float(per_class_recall_t[class_present].mean().item())
        if bool(class_present.any()) else None
    )
    macro_f1 = (
        float(per_class_f1_t[class_present].mean().item())
        if bool(class_present.any()) else None
    )
    total_valid = int(label_counts.sum().item())
    total_pred = int(pred_counts.sum().item())
    accuracy = _safe_div(float(total_inter.sum().item()), float(total_valid))
    pred_class_ratios_t = pred_counts / pred_counts.sum().clamp(min=1)
    label_class_ratios_t = label_counts / label_counts.sum().clamp(min=1)
    class_area_bias_t = pred_class_ratios_t - label_class_ratios_t

    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    specificity = _safe_div(float(tn), float(tn + fp))
    foreground_iou = _safe_div(float(tp), float(tp + fp + fn))
    foreground_dice = _safe_div(float(2 * tp), float(2 * tp + fp + fn))
    false_positive_rate = _safe_div(float(fp), float(fp + tn))
    false_negative_rate = _safe_div(float(fn), float(fn + tp))
    balanced_accuracy = (
        None if recall is None or specificity is None else float((recall + specificity) / 2.0)
    )
    pred_positive_ratio = _safe_div(float(tp + fp), float(total_pred))
    true_positive_ratio = _safe_div(float(tp + fn), float(total_valid))
    area_bias = (
        None
        if pred_positive_ratio is None or true_positive_ratio is None
        else float(pred_positive_ratio - true_positive_ratio)
    )
    area_ratio = (
        None
        if pred_positive_ratio is None or true_positive_ratio is None or true_positive_ratio <= 0
        else float(pred_positive_ratio / true_positive_ratio)
    )

    logit_means = None
    if logit_sums is not None and logit_count > 0:
        logit_means = [float(x) for x in (logit_sums / logit_count).detach().cpu().tolist()]

    return {
        "miou": miou,
        "macro_iou": miou,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mean_class_accuracy": macro_recall,
        "per_class_iou": [float(x) for x in iou.detach().cpu().tolist()],
        "per_class_precision": [
            float(x) for x in per_class_precision_t.detach().cpu().tolist()
        ],
        "per_class_recall": [float(x) for x in per_class_recall_t.detach().cpu().tolist()],
        "per_class_f1": [float(x) for x in per_class_f1_t.detach().cpu().tolist()],
        "foreground_class": int(foreground_class),
        "foreground_iou": foreground_iou,
        "foreground_dice": foreground_dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pixel_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "pred_positive_ratio": pred_positive_ratio,
        "true_positive_ratio": true_positive_ratio,
        "area_bias": area_bias,
        "area_ratio": area_ratio,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "pred_counts": [int(x) for x in pred_counts.detach().cpu().tolist()],
        "label_counts": [int(x) for x in label_counts.detach().cpu().tolist()],
        "pred_class_ratios": [float(x) for x in pred_class_ratios_t.detach().cpu().tolist()],
        "label_class_ratios": [float(x) for x in label_class_ratios_t.detach().cpu().tolist()],
        "class_area_bias": [float(x) for x in class_area_bias_t.detach().cpu().tolist()],
        "class_logit_means": logit_means,
        "mean_image_foreground_iou": (
            float(np.mean(per_image_fg_iou)) if per_image_fg_iou else None
        ),
        "mean_image_foreground_dice": (
            float(np.mean(per_image_fg_dice)) if per_image_fg_dice else None
        ),
        "positive_image_recall": _safe_div(
            float(detected_positive_images), float(positive_image_count)
        ),
        "background_image_false_alarm_rate": _safe_div(
            float(false_alarm_background_images), float(background_image_count)
        ),
        "positive_image_count": int(positive_image_count),
        "background_image_count": int(background_image_count),
        "predicted_positive_image_count": int(predicted_positive_image_count),
        "image_count": int(image_count),
        "loss": (float(loss_sum / max(loss_batches, 1)) if criterion is not None else None),
    }


def rgb_indices_for_dataset(dataset_name: str, n_channels: int) -> list[int]:
    """Return channel indices in RGB order for visualization."""
    if n_channels <= 0:
        raise ValueError("Cannot build RGB visualization for an image with no channels")
    if dataset_name in {"sen1floods11", "cloudsen12", "bigearthnet", "landslide4sense", "geobench_sa_crop_type"}:
        indices = [3, 2, 1]  # Sentinel-2 B4/B3/B2 for zero-based B1.. order.
    elif dataset_name == "fire_scars":
        indices = [2, 1, 0]  # HLS RED/GREEN/BLUE.
    elif dataset_name == "multitemporal_crop":
        indices = [2, 1, 0]  # First HLS timestamp RED/GREEN/BLUE.
    elif dataset_name == "loveda":
        indices = [0, 1, 2]  # Already RGB.
    elif dataset_name == "abi_cloud":
        indices = [1, 2, 0]  # Approximate visible red/veg/blue composite.
    else:
        indices = [min(i, n_channels - 1) for i in (0, 1, 2)]
    return [min(max(i, 0), n_channels - 1) for i in indices]


def _rgb_from_tensor(image: torch.Tensor, dataset_name: str) -> tuple[np.ndarray, list[int]]:
    image_np = image.detach().float().cpu().numpy()
    indices = rgb_indices_for_dataset(dataset_name, image_np.shape[0])
    rgb = np.stack([image_np[i] for i in indices], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, indices


def _overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    out = rgb.copy()
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        color_arr = np.asarray(color, dtype=np.float32)
        out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0.0, 1.0)


def _paint_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
) -> np.ndarray:
    out = rgb.copy()
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        out[mask_bool] = np.asarray(color, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def _binary_prediction_overlay(
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    valid: np.ndarray,
    *,
    true_positive_color: tuple[float, float, float],
    false_positive_color: tuple[float, float, float],
    false_negative_color: tuple[float, float, float],
) -> np.ndarray:
    out = rgb.copy()
    valid_bool = valid.astype(bool)
    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)

    true_positive = gt_bool & pred_bool & valid_bool
    false_positive = (~gt_bool) & pred_bool & valid_bool
    false_negative = gt_bool & (~pred_bool) & valid_bool

    out[true_positive] = np.asarray(true_positive_color, dtype=np.float32)
    out[false_positive] = np.asarray(false_positive_color, dtype=np.float32)
    out[false_negative] = np.asarray(false_negative_color, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def _segmentation_palette(n_classes: int) -> np.ndarray:
    """Fixed class palette for multi-class segmentation overlays."""
    base = np.asarray(
        [
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
            (188, 189, 34),
            (23, 190, 207),
            (174, 199, 232),
            (255, 187, 120),
            (152, 223, 138),
            (255, 152, 150),
            (197, 176, 213),
            (196, 156, 148),
            (247, 182, 210),
            (199, 199, 199),
            (219, 219, 141),
            (158, 218, 229),
        ],
        dtype=np.float32,
    ) / 255.0
    if n_classes <= len(base):
        return base[:n_classes]

    extra = []
    for i in range(n_classes - len(base)):
        hue = (i * 0.61803398875) % 1.0
        sector = int(hue * 6.0)
        f = hue * 6.0 - sector
        q = 1.0 - f
        if sector % 6 == 0:
            rgb = (1.0, f, 0.0)
        elif sector == 1:
            rgb = (q, 1.0, 0.0)
        elif sector == 2:
            rgb = (0.0, 1.0, f)
        elif sector == 3:
            rgb = (0.0, q, 1.0)
        elif sector == 4:
            rgb = (f, 0.0, 1.0)
        else:
            rgb = (1.0, 0.0, q)
        extra.append(rgb)
    return np.concatenate([base, np.asarray(extra, dtype=np.float32)], axis=0)


def _multiclass_overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    palette: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out = rgb.copy()
    labels_i = labels.astype(np.int64)
    valid_bool = valid.astype(bool)
    for cls in np.unique(labels_i[valid_bool]):
        cls_int = int(cls)
        if cls_int < 0 or cls_int >= len(palette):
            continue
        mask = (labels_i == cls_int) & valid_bool
        out[mask] = (1.0 - alpha) * out[mask] + alpha * palette[cls_int]
    return np.clip(out, 0.0, 1.0)


def _palette_legend(palette: np.ndarray) -> dict[str, list[float]]:
    return {
        f"class_{idx}": [float(v) for v in color.tolist()]
        for idx, color in enumerate(palette)
    }


def _sample_path(loader: torch.utils.data.DataLoader, index: int) -> str | None:
    ds = getattr(loader, "dataset", None)
    samples = getattr(ds, "samples", None)
    if samples is None or index >= len(samples):
        return None
    sample = samples[index]
    if isinstance(sample, tuple) and sample:
        return str(sample[0])
    return str(sample)


def save_segmentation_visualizations(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    dataset_name: str,
    run_id: str,
    out_dir: Path,
    max_examples: int,
    alpha: float = 0.45,
    foreground_class: int = 1,
    ignore_index: int = -1,
) -> dict[str, Any]:
    """Save RGB | ground-truth | prediction triptychs.

    Binary tasks use solid colors: green for GT foreground, and green/blue/red
    for true positives / false positives / false negatives in the prediction
    panel. Multi-class tasks use a fixed class palette so the same label has
    the same color across all images.
    """
    from PIL import Image

    vis_dir = out_dir / run_id
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    records: list[dict[str, Any]] = []
    sample_index = 0
    binary_gt_color = (0.0, 1.0, 0.0)
    true_positive_color = (0.0, 1.0, 0.0)
    false_positive_color = (0.0, 0.0, 1.0)
    false_negative_color = (1.0, 0.0, 0.0)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels_device = labels.to(device)
            outputs = _unwrap_outputs(model(images))
            preds = outputs.argmax(dim=1).detach().cpu()
            is_binary = int(outputs.shape[1]) == 2

            for b in range(images.shape[0]):
                if len(records) >= max_examples:
                    manifest_path = vis_dir / "manifest.json"
                    manifest_path.write_text(json.dumps(records, indent=2))
                    return {
                        "dir": str(vis_dir),
                        "manifest_path": str(manifest_path),
                        "images": records,
                    }

                rgb, indices = _rgb_from_tensor(images[b], dataset_name)
                label = labels[b].detach().cpu()
                pred = preds[b]
                valid = label != ignore_index
                gt_mask = ((label == foreground_class) & valid).numpy()
                pred_mask = ((pred == foreground_class) & valid).numpy()
                valid_np = valid.numpy()

                if is_binary:
                    gt_overlay = _paint_mask(rgb, gt_mask, binary_gt_color)
                    pred_overlay = _binary_prediction_overlay(
                        rgb,
                        gt_mask,
                        pred_mask,
                        valid_np,
                        true_positive_color=true_positive_color,
                        false_positive_color=false_positive_color,
                        false_negative_color=false_negative_color,
                    )
                    panel_order = [
                        "rgb_input",
                        "ground_truth_solid_positive",
                        "prediction_tp_fp_fn",
                    ]
                    color_legend = {
                        "ground_truth_positive": binary_gt_color,
                        "true_positive": true_positive_color,
                        "false_positive": false_positive_color,
                        "false_negative": false_negative_color,
                    }
                else:
                    palette = _segmentation_palette(int(outputs.shape[1]))
                    gt_overlay = _multiclass_overlay(
                        rgb,
                        label.numpy(),
                        valid_np,
                        palette,
                        alpha,
                    )
                    pred_overlay = _multiclass_overlay(
                        rgb,
                        pred.numpy(),
                        valid_np,
                        palette,
                        alpha,
                    )
                    panel_order = ["rgb_input", "ground_truth_overlay", "prediction_overlay"]
                    color_legend = _palette_legend(palette)

                spacer = np.ones((rgb.shape[0], 4, 3), dtype=np.float32)
                triptych = np.concatenate([rgb, spacer, gt_overlay, spacer, pred_overlay], axis=1)
                png_path = vis_dir / f"test_{len(records):03d}.png"
                Image.fromarray((triptych * 255.0).round().astype(np.uint8)).save(png_path)

                pred_sum = int(pred_mask.sum())
                true_sum = int(gt_mask.sum())
                inter = int(np.logical_and(pred_mask, gt_mask).sum())
                union = pred_sum + true_sum - inter
                denom = pred_sum + true_sum
                records.append({
                    "sample_index": sample_index,
                    "image_path": _sample_path(loader, sample_index),
                    "png_path": str(png_path),
                    "panel_order": panel_order,
                    "rgb_indices": indices,
                    "color_legend_rgb": color_legend,
                    "alpha": float(alpha),
                    "true_positive_ratio": float(true_sum / max(int(valid.sum().item()), 1)),
                    "pred_positive_ratio": float(pred_sum / max(int(valid.sum().item()), 1)),
                    "foreground_iou": (float(inter / union) if union > 0 else None),
                    "foreground_dice": (float((2.0 * inter) / denom) if denom > 0 else None),
                })
                sample_index += 1

    manifest_path = vis_dir / "manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2))
    return {
        "dir": str(vis_dir),
        "manifest_path": str(manifest_path),
        "images": records,
    }
