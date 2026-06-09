#!/usr/bin/env python3
"""SPECTRA public fine-tuning pipeline.

Supported methods:
  lp, lora8, lora16, lora32, lora64, last_stage, surgical, full_ft, spectra

SPECTRA = Band-Routed Embedding (BRE) + ST-LoRA. ST-LoRA uses the
STPlanner in either transfer or repair mode to choose per-stage LoRA ranks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _env_name in ("TERRATORCH_ROOT", "PRITHVI_EO_ROOT"):
    _env_path = os.environ.get(_env_name)
    if _env_path:
        sys.path.insert(0, _env_path)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("finetune")

import torch.nn as nn

from spectra.data.config import (
    RESULTS_DIR, FIXED_SPLITS_DIR, BACKBONE_SPECS, FULL_BAND_CONFIGS,
    GEOBENCH_SA_CROP_TYPE_ROOT,
)
from spectra.data.srf import get_srf_triples, get_srf
from spectra.tokenizer.band_selector  import BandSelector
from spectra.tokenizer.bre import BRE
from spectra.adapter.nested_lora import NestedLoRABackbone
from spectra.planner.stplanner import STPlanner, STPlannerConfig
from spectra.planner.logme_profiler import StagewiseLogMEProfiler
from spectra.baselines.schedules import (
    lora1_schedule, lora2_schedule, lora4_schedule,
    lora8_schedule, lora16_schedule, lora32_schedule, lora64_schedule,
    last_stage_full_ft_schedule,
    surgical_ft_schedule, linear_probe_schedule, full_ft_schedule,
)
from spectra.evaluation.segmentation_eval import (
    evaluate_segmentation,
    save_segmentation_visualizations,
)


METHODS = {
    "lp": linear_probe_schedule,
    "lora8": lora8_schedule,
    "lora16": lora16_schedule,
    "lora32": lora32_schedule,
    "lora64": lora64_schedule,
    "last_stage": last_stage_full_ft_schedule,
    "surgical": surgical_ft_schedule,
    "full_ft": full_ft_schedule,
    "lp_bandsel": linear_probe_schedule,
    "lora8_bandsel": lora8_schedule,
    "lora16_bandsel": lora16_schedule,
    "lora32_bandsel": lora32_schedule,
    "lora64_bandsel": lora64_schedule,
    "last_stage_bandsel": last_stage_full_ft_schedule,
    "surgical_bandsel": surgical_ft_schedule,
    "full_ft_bandsel": full_ft_schedule,
    "spectra_bre": None,
}

PUBLIC_METHOD_ALIASES = {
    "lp": "lp_bandsel",
    "lora8": "lora8_bandsel",
    "lora16": "lora16_bandsel",
    "lora32": "lora32_bandsel",
    "lora64": "lora64_bandsel",
    "last_stage": "last_stage_bandsel",
    "surgical": "surgical_bandsel",
    "full_ft": "full_ft_bandsel",
    "spectra": "spectra_bre",
}

# Pre-training sensor key per backbone
BACKBONE_PRETRAIN_SENSOR: dict[str, str] = {
    "prithvi_eo_v2_600": "fire_scars",   # HLS 6B (490,560,665,865,1610,2202 nm)
    "satmae":            "fire_scars",   # also HLS-based
    "satmae_sentinel_vitl": "satmae_s2_10b",  # fMoW-Sentinel groups: B2/B3/B4/B8, red-edge, SWIR
    "scalemae_fmow_rgb": "fmow_rgb",     # fMoW RGB / ImageNet-normalized visual bands
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune geospatial foundation models with published baselines or SPECTRA."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", default="spectra", choices=list(PUBLIC_METHOD_ALIASES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--model-seed", "--model-init-seed", dest="model_seed", type=int, default=None)
    p.add_argument("--residual-seed", "--residual-init-seed", dest="residual_seed", type=int, default=None)
    p.add_argument("--lora-seed", "--lora-init-seed", dest="lora_seed", type=int, default=None)
    p.add_argument("--head-seed", "--head-init-seed", dest="head_seed", type=int, default=None)
    p.add_argument("--loader-seed", type=int, default=None)
    p.add_argument("--train-shuffle-seed", type=int, default=None)
    p.add_argument("--eval-shuffle-seed", type=int, default=None)

    p.add_argument("--loss-mode", choices=("ce_dice_dwa", "ce_dice"), default="ce_dice_dwa")
    p.add_argument("--dice-lambda", type=float, default=1.0)
    p.add_argument("--dwa-temperature", type=float, default=2.0)
    p.add_argument("--minority-boost-cap-ratio", type=float, default=8.0)

    p.add_argument("--st-planner", "--stlora-planner", dest="star_planner",
                   choices=("transfer", "repair"), default="transfer")
    p.add_argument("--st-reference-rank", dest="star_reference_rank", metavar="ST_REFERENCE_RANK",
                   type=int, default=32)
    p.add_argument("--st-tau", dest="star_tau", metavar="ST_TAU", type=float, default=0.05)
    p.add_argument("--st-stage-prior", dest="star_stage_prior", metavar="ST_STAGE_PRIOR",
                   type=str, default="0.8,1.0,1.1,1.2")
    p.add_argument("--st-budget-candidates", dest="star_budget_candidates", metavar="ST_BUDGET_CANDIDATES",
                   type=str, default="32,48,60,72,80,92,96,104,112")
    p.add_argument("--st-budget-f-min", dest="star_budget_f_min", metavar="ST_BUDGET_F_MIN",
                   type=float, default=0.40)
    p.add_argument("--st-budget-f-max", dest="star_budget_f_max", metavar="ST_BUDGET_F_MAX",
                   type=float, default=0.85)
    p.add_argument("--st-budget-midpoint", dest="star_budget_midpoint", metavar="ST_BUDGET_MIDPOINT",
                   type=float, default=0.50)
    p.add_argument("--st-budget-slope", dest="star_budget_slope", metavar="ST_BUDGET_SLOPE",
                   type=float, default=3.0)
    p.add_argument("--st-budget-override", dest="star_budget_override", metavar="ST_BUDGET_OVERRIDE",
                   type=int, default=None)
    p.add_argument("--st-q-bank-csv", dest="star_q_bank_csv", metavar="ST_Q_BANK_CSV",
                   type=Path, default=RESULTS_DIR / "spectral_mismatch_transferability.csv")
    p.add_argument("--st-q-min", dest="star_q_min", metavar="ST_Q_MIN", type=float, default=None)
    p.add_argument("--st-q-max", dest="star_q_max", metavar="ST_Q_MAX", type=float, default=None)

    p.add_argument("--epochs-override", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--auto-test", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-test-vis-examples", type=int, default=8)
    p.add_argument("--auto-test-vis-alpha", type=float, default=None)
    p.add_argument("--auto-test-vis-dir", type=Path, default=RESULTS_DIR / "visualizations")
    p.add_argument("--load-adapted-checkpoint", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--use-wandb", action="store_true")

    args = p.parse_args()
    args.public_method = args.method
    args.method = PUBLIC_METHOD_ALIASES[args.public_method]
    if args.max_rank is None:
        if args.public_method.startswith("lora"):
            args.max_rank = int(args.public_method.replace("lora", ""))
        elif args.public_method == "spectra":
            args.max_rank = 64

    # Internal compatibility defaults for the development runner. These options
    # are intentionally not exposed in the public CLI.
    args.consume_residual_rng = False
    args.force_residual_zero = False
    args.freeze_residual = False
    args.lr_residual_override = None
    args.lr_adapter_override = None
    args.lr_backbone_override = None
    args.lr_head_override = None
    args.train_patch_embed = False
    args.lr_patch_embed_override = None
    args.weight_decay_patch_embed_override = None
    args.max_grad_norm_override = None
    args.grad_clip_mode = "global"
    args.weight_decay_residual_override = None
    args.dice_include_background = False
    args.positive_batch_min = 0
    args.positive_crop_max_tries = 0
    args.stage1_bandsel_epochs = 0
    args.stage2_lr_residual = None
    args.residual_gamma_ramp_epochs = 0
    args.eval_delta_off = False
    args.rank_grid = None
    args.staircase_thresholds = None
    args.ranks = None
    args.unfrozen = None
    return args


def load_config(path: Path) -> dict:
    base_path = path.parent.parent / "configs" / "base.yaml"
    cfg: dict = {}
    if base_path.exists():
        cfg = yaml.safe_load(base_path.read_text()) or {}
    override = yaml.safe_load(path.read_text()) or {}
    cfg.update(override)
    return cfg


def resolve_seed(specific: int | None, fallback: int) -> int:
    return fallback if specific is None else specific


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int) -> None:
    """Backward-compatible alias for older local helpers."""
    set_all_seeds(seed, deterministic=False)


def prithvi_upernet_feature_indices(backbone_name: str) -> list[int]:
    """Benchmark-style multi-scale feature taps for Prithvi ViT backbones."""
    if "prithvi_eo_v2_600" in backbone_name:
        return [7, 15, 23, 31]
    if "prithvi_eo_v2_300" in backbone_name or "prithvi_eo_v2_300_tl" in backbone_name:
        return [5, 11, 17, 23]
    if "prithvi_eo_v1_100" in backbone_name:
        return [2, 5, 8, 11]
    raise ValueError(
        f"No default UPerNet feature indices are defined for backbone={backbone_name!r}. "
        "Add the backbone-specific indices before running benchmark-comparable fine-tuning."
    )


def prithvi_model_architecture(backbone_name: str) -> dict:
    """Return the paper-comparable Prithvi segmentation architecture defaults."""
    feature_indices = prithvi_upernet_feature_indices(backbone_name)
    return {
        "decoder": "UperNetDecoder",
        "feature_indices": feature_indices,
        "necks": [
            {"name": "SelectIndices", "indices": feature_indices},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        "decoder_channels": 256,
    }


def model_architecture_defaults(backbone_name: str) -> dict:
    """Return segmentation architecture metadata for supported backbones."""
    if backbone_name == "satmae_sentinel_vitl":
        return {
            "decoder": "UperNetDecoder",
            "feature_indices": [5, 11, 17, 23],
            "necks": [{"name": "ScaleModulesInUPerNet"}],
            "decoder_channels": 256,
        }
    if backbone_name == "scalemae_fmow_rgb":
        return {
            "decoder": "UperNetDecoder",
            "feature_indices": [7, 11, 15, 23],
            "necks": [{"name": "ScaleModulesInUPerNet"}],
            "decoder_channels": 256,
        }
    return prithvi_model_architecture(backbone_name)


def setup_file_logging(run_id: str) -> Path:
    """Mirror Python logging output to results/logs/{run_id}.log."""
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"

    root_logger = logging.getLogger()
    resolved = str(log_path.resolve())
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved:
            return log_path

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)
    return log_path


def adapted_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Save trained/adapted components.

    This keeps LoRA/residual/head checkpoint compatibility while also saving
    unfrozen encoder weights for full_ft, last_stage, and surgical runs.
    """
    trainable_names = {
        name for name, param in model.named_parameters()
        if param.requires_grad
    }
    state = {}
    for name, tensor in model.state_dict().items():
        legacy_adapter_or_head = (
            not name.startswith("encoder.")
            or ".A" in name
            or ".B" in name
            or "patch_embed.R." in name
        )
        if name in trainable_names or legacy_adapter_or_head:
            state[name] = tensor.detach().cpu().clone()
    return state


def save_adapted_checkpoint(
    path: Path,
    model: nn.Module,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": adapted_state_dict(model), "meta": meta}, path)


def load_adapted_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    logger.info(
        "Loaded adapted checkpoint %s: loaded=%d missing=%d unexpected=%d",
        path,
        len(state),
        len(missing),
        len(unexpected),
    )
    return {
        "path": str(path),
        "n_loaded": len(state),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "missing_sample": list(missing[:10]),
        "unexpected_sample": list(unexpected[:10]),
    }


def first_lora_norms(lora_backbone: NestedLoRABackbone) -> dict[str, float | None]:
    for stage in lora_backbone.stages:
        if stage.lora_layers:
            layer = stage.lora_layers[0]
            return {
                "first_lora_A_norm": float(layer.A.detach().norm().item()),
                "first_lora_B_norm": float(layer.B.detach().norm().item()),
            }
    return {"first_lora_A_norm": None, "first_lora_B_norm": None}


def head_weight_norm(model: nn.Module) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if not name.startswith("encoder.") and p.detach().numel() > 0:
            total += float(p.detach().pow(2).sum().item())
    return total ** 0.5


def reset_task_head(model: nn.Module, seed: int) -> int:
    set_all_seeds(seed)
    n_reset = 0
    for child_name, child in model.named_children():
        if child_name == "encoder":
            continue
        for module in child.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
                n_reset += 1
    return n_reset


def optimizer_param_occurrences(optimizer: torch.optim.Optimizer, params: list[nn.Parameter]) -> int:
    ids = {id(p) for p in params}
    return sum(
        1
        for group in optimizer.param_groups
        for p in group["params"]
        if id(p) in ids
    )


def tensor_norm(x: torch.Tensor) -> float:
    return float(x.detach().float().norm().cpu())


def param_count(params) -> int:
    return int(sum(p.numel() for p in params))


def unwrap_pretrained_patch_embed(module: nn.Module | None) -> nn.Module | None:
    """Return the pretrained patch embedding inside adapter wrappers."""
    seen: set[int] = set()
    current = module
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "original"):
            current = getattr(current, "original")
            continue
        if hasattr(current, "inner"):
            current = getattr(current, "inner")
            continue
        break
    return current


def pretrained_patch_embed_params(model: nn.Module) -> tuple[nn.Module | None, list[nn.Parameter]]:
    encoder = getattr(model, "encoder", None)
    patch_embed = getattr(encoder, "patch_embed", None) if encoder is not None else None
    patch_embed = unwrap_pretrained_patch_embed(patch_embed)
    if patch_embed is None:
        return None, []
    return patch_embed, list(patch_embed.parameters())


def active_lora_param_count(lora_backbone: NestedLoRABackbone, ranks: list[int] | None) -> int | None:
    """Count active LoRA slice parameters implied by a stage-wise rank schedule."""
    if ranks is None:
        return None
    total = 0
    for stage, rank in zip(lora_backbone.stages, ranks):
        r = int(rank)
        if r <= 0:
            continue
        for layer in stage.lora_layers:
            k = min(r, layer.A.shape[0], layer.B.shape[1])
            total += k * layer.A.shape[1]
            total += layer.B.shape[0] * k
    return int(total)


def residual_adapter_params(bre: nn.Module | None) -> list[nn.Parameter]:
    if bre is None:
        return []
    adapter_parameters = getattr(bre, "adapter_parameters", None)
    if callable(adapter_parameters):
        return list(adapter_parameters())
    return list(bre.R.parameters())


def residual_init_norms(bre: nn.Module | None) -> dict[str, float | None]:
    if bre is None:
        return {
            "R0_weight_norm": None,
            "R2_weight_norm": None,
            "R4_weight_norm_after_zero_init": None,
            "router_logits_norm": None,
            "router_entropy_mean": None,
        }
    info = {
        "R0_weight_norm": tensor_norm(bre.R[0].weight),
        "R2_weight_norm": tensor_norm(bre.R[2].weight),
        "R4_weight_norm_after_zero_init": tensor_norm(bre.R[4].weight),
    }
    gate_logits = getattr(bre, "gate_logits", None)
    if gate_logits is not None:
        info["gate_logits_norm"] = tensor_norm(gate_logits)
        info["router_logits_norm"] = None
        info["router_entropy_mean"] = None
        gate_values = getattr(bre, "_gate_values", None)
        if callable(gate_values):
            gates = gate_values().detach().float()
            info.update({
                "router_gate_mean": float(gates.mean().item()),
                "router_gate_min": float(gates.min().item()),
                "router_gate_max": float(gates.max().item()),
            })
            if gates.dim() >= 2:
                info.update({
                    "router_gate_per_target_mean": [float(v) for v in gates.mean(dim=1).cpu().tolist()],
                    "router_gate_per_target_min": [float(v) for v in gates.min(dim=1).values.cpu().tolist()],
                    "router_gate_per_target_max": [float(v) for v in gates.max(dim=1).values.cpu().tolist()],
                })
    else:
        info["gate_logits_norm"] = None

    router_logits = getattr(bre, "router_logits", None)
    if router_logits is not None:
        info["router_logits_norm"] = tensor_norm(router_logits)
        if router_logits.dim() >= 2:
            weights = torch.softmax(router_logits.detach().float(), dim=-1)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
            info["router_entropy_mean"] = float(entropy.mean().item())
        else:
            gate_values = getattr(bre, "_gate_values", None)
            if callable(gate_values):
                gates = gate_values().detach().float()
                info.update({
                    "router_entropy_mean": None,
                    "router_gate_mean": float(gates.mean().item()),
                    "router_gate_min": float(gates.min().item()),
                    "router_gate_max": float(gates.max().item()),
                })
            else:
                info["router_entropy_mean"] = None
    else:
        info.update({
            "router_logits_norm": None,
            "router_entropy_mean": None,
        })
    return info


def count_split_labels(
    dataset_name: str,
    split: str,
    cfg: dict,
    n_classes: int,
    split_seed: int,
) -> torch.Tensor:
    if dataset_name == "geobench_sa_crop_type":
        root = GEOBENCH_SA_CROP_TYPE_ROOT
        split_key = {"val": "valid", "validation": "valid"}.get(split, split)
        partition_path = root / "default_partition.json"
        label_stats_path = root / "label_stats.json"

        if partition_path.exists() and label_stats_path.exists():
            partition = json.loads(partition_path.read_text())
            label_stats = json.loads(label_stats_path.read_text())
        else:
            archive_path = root / "files-archive"
            with zipfile.ZipFile(archive_path) as zf:
                partition = json.loads(zf.read("default_partition.json"))
                label_stats = json.loads(zf.read("label_stats.json"))

        counts = torch.zeros(n_classes)
        # GEO-Bench label_stats stores full-chip per-class pixel fractions in
        # the official label space. Keep labels 0..9 for benchmark-compatible
        # training/evaluation.
        pixels_per_chip = float(256 * 256)
        for sample_id in partition[split_key]:
            ratios = label_stats[sample_id]
            for c, ratio in enumerate(ratios[:n_classes]):
                counts[c] += float(ratio) * pixels_per_chip
        return counts

    from spectra.data.datasets import build_dataset as _build_dataset

    ds = _build_dataset(
        dataset_name,
        split,
        crop_size=cfg.get("crop_size", 224),
        augment=False,
        split_seed=split_seed,
        pad_multiple=cfg.get("pad_multiple", 14),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=16,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )
    counts = torch.zeros(n_classes)
    for _, labels in loader:
        valid = labels != -1
        for c in range(n_classes):
            counts[c] += (labels[valid] == c).sum()
    return counts


def positive_ratio(counts: torch.Tensor) -> float | None:
    if counts.numel() < 2:
        return None
    total = counts.sum().item()
    if total <= 0:
        return None
    return float((counts[1] / counts.sum()).item())


def class_ratios_from_counts(counts: torch.Tensor) -> list[float]:
    total = float(counts.sum().item())
    if total <= 0:
        return [0.0 for _ in range(int(counts.numel()))]
    return [float(x) for x in (counts.float() / total).tolist()]


def load_backbone(backbone_name: str, n_classes: int, device: torch.device):
    if backbone_name == "scalemae_fmow_rgb":
        from spectra.backbone.scalemae import load_scalemae
        return load_scalemae(backbone_name, n_classes, device)
    if backbone_name == "satmae_sentinel_vitl":
        from spectra.backbone.satmae import load_satmae
        return load_satmae(backbone_name, n_classes, device)

    # Prithvi / SatMAE: load via TerraTorch
    from terratorch.tasks import SemanticSegmentationTask
    arch = prithvi_model_architecture(backbone_name)
    task = SemanticSegmentationTask(
        model_args=dict(
            backbone_pretrained=True,
            backbone=backbone_name,
            backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            backbone_num_frames=1,
            necks=arch["necks"],
            decoder=arch["decoder"],
            decoder_channels=arch["decoder_channels"],
            num_classes=n_classes,
        ),
        plot_on_val=False, lr=1e-4,
        model_factory="EncoderDecoderFactory",
    )
    logger.info(
        "Loaded %s with %s feature_indices=%s",
        backbone_name,
        arch["decoder"],
        arch["feature_indices"],
    )
    return task.model.to(device)


def wire_band_selector(model: nn.Module, selector: BandSelector) -> None:
    """Prepend a parameter-free BandSelector to encoder.forward so the pretrained
    patch_embed receives a pretrained-compatible slice instead of all C_in bands.
    """
    encoder = model.encoder
    orig_fwd = encoder.forward_features

    def patched_forward(x, **kwargs):
        return orig_fwd(selector(x), **kwargs)

    encoder.forward_features = patched_forward
    encoder.forward = patched_forward
    logger.info("BandSelector wired: %d-band → %d-band (indices=%s)",
                selector.in_chans, selector.out_chans, selector.indices.tolist())


def build_dataloader(
    dataset_name: str,
    split: str,
    cfg: dict,
    seed: int,
    split_seed: int,
    positive_batch_min: int = 0,
    positive_crop_max_tries: int = 0,
):
    from spectra.data.datasets import build_dataloader as _build
    return _build(
        dataset_name,
        split,
        batch_size  = cfg.get("batch_size", 8),
        num_workers = cfg.get("num_workers", 4),
        crop_size   = cfg.get("crop_size", 224),
        seed        = seed,
        split_seed  = split_seed,
        pad_multiple = cfg.get("pad_multiple", 14),
        positive_batch_min = positive_batch_min if split == "train" else 0,
        positive_crop_max_tries = positive_crop_max_tries if split == "train" else 0,
    )


def _load_stage_costs(backbone_name: str, spec) -> tuple:
    """Load per-stage timing from timing_pass.py JSON for ST-LoRA budget planning.

    Returns (stage_dims, stage_n_params, t_lora, t_full).
    t_lora and t_full are derived from aggregate measurements (not per-stage sums)
    to avoid overcounting the shared backward pass.
    """
    timing_path = RESULTS_DIR / "timing" / f"timing_{backbone_name}.json"
    if timing_path.exists():
        import json as _json
        td = _json.loads(timing_path.read_text())
        agg = td.get("aggregate", {})
        # Use aggregate-derived per-stage uniform estimates (in seconds)
        t_lora_ms = agg.get("t_lora_per_stage_ms", 11.6)
        t_full_ms = agg.get("t_full_per_stage_ms", 30.9)
        t_lora = [t_lora_ms / 1000] * spec.n_stages
        t_full = [t_full_ms / 1000] * spec.n_stages
        logger.info("Loaded timing from %s: t_lora=%.1fms/stage  t_full=%.1fms/stage",
                    timing_path.name, t_lora_ms, t_full_ms)
    else:
        logger.warning("Timing JSON not found at %s — using fallback values", timing_path)
        t_lora = [0.012] * spec.n_stages
        t_full = [0.031] * spec.n_stages

    stage_dims     = [spec.embed_dim] * spec.n_stages
    # Approximate: 4 QKV linear layers per block × blocks_per_stage × embed_dim²
    n_layers_per_stage = spec.n_layers // spec.n_stages
    stage_n_params = [n_layers_per_stage * 4 * spec.embed_dim * spec.embed_dim] * spec.n_stages
    return stage_dims, stage_n_params, t_lora, t_full


def apply_baseline_schedule(method: str, lora_backbone: NestedLoRABackbone, cfg: dict):
    n_stages = cfg["lora"]["n_stages"]
    schedule_fn = METHODS.get(method)
    if schedule_fn is None:
        raise ValueError(f"Unknown or non-schedule method: {method}")
    sched = schedule_fn(n_stages=n_stages)   # keyword to handle surgical_ft_schedule's logme_scores-first signature
    # fix_train_rank=True: use a fixed rank per layer during training instead of stochastic
    # sampling. Baselines are standard (non-nested) LoRA — each layer must see the same rank
    # so head gradients are stable across the forward pass.
    lora_backbone.apply_schedule(sched.ranks, sched.unfrozen, fix_train_rank=True)
    logger.info("Applied baseline '%s': %s", method, sched.schedule_str())
    return sched


class DiceLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -1,
        include_background: bool = False,
        class_mask: torch.Tensor | list[bool] | None = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth
        self.class_mask = class_mask

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid = target != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0

        n_classes = logits.shape[1]
        if self.class_mask is None:
            class_mask = torch.ones(n_classes, dtype=torch.bool, device=logits.device)
        else:
            class_mask = torch.as_tensor(
                self.class_mask, dtype=torch.bool, device=logits.device
            ).clone()
            if class_mask.numel() != n_classes:
                raise ValueError(
                    f"Dice class_mask has {class_mask.numel()} entries, expected {n_classes}"
                )
        if not self.include_background and n_classes > 1:
            class_mask[0] = False

        losses = []
        valid_f = valid.float()
        for cls in class_mask.nonzero(as_tuple=False).flatten().tolist():
            pred = probs[:, cls] * valid_f
            truth = ((target == cls) & valid).float()
            inter = (pred * truth).sum()
            denom = pred.sum() + truth.sum()
            dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
            losses.append(1.0 - dice)

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_loss: nn.Module, dice_loss: nn.Module, dice_lambda: float) -> None:
        super().__init__()
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.dice_lambda = float(dice_lambda)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(logits, target) + self.dice_lambda * self.dice_loss(logits, target)


class DynamicWeightAverageLoss(nn.Module):
    """DWA weighting for a CE term and a Dice term.

    We keep the first two epochs at equal weights. From epoch 3 onward, the
    weights are computed from the previous two epoch-average component losses:
    w_i = 2 * softmax((L_i[t-1] / L_i[t-2]) / T).
    """

    def __init__(
        self,
        ce_loss: nn.Module,
        dice_loss: nn.Module,
        temperature: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("DWA temperature must be > 0")
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.register_buffer("weights", torch.ones(2, dtype=torch.float32))
        self.history: list[dict[str, float]] = []
        self._collect = False
        self._epoch_sums = {"ce_loss": 0.0, "dice_loss": 0.0, "total_loss": 0.0}
        self._epoch_batches = 0
        self._last_epoch_summary: dict[str, float] | None = None

    def _compute_weights(self) -> None:
        if len(self.history) < 2:
            self.weights.fill_(1.0)
            return
        prev = self.history[-1]
        prev2 = self.history[-2]
        ratios = torch.tensor(
            [
                prev["ce_loss"] / max(prev2["ce_loss"], self.eps),
                prev["dice_loss"] / max(prev2["dice_loss"], self.eps),
            ],
            dtype=torch.float32,
            device=self.weights.device,
        )
        self.weights.copy_(2.0 * torch.softmax(ratios / self.temperature, dim=0))

    def begin_epoch(self) -> None:
        self._compute_weights()
        self._collect = True
        self._epoch_sums = {"ce_loss": 0.0, "dice_loss": 0.0, "total_loss": 0.0}
        self._epoch_batches = 0
        self._last_epoch_summary = None

    def end_epoch(self) -> dict[str, float]:
        self._collect = False
        denom = max(self._epoch_batches, 1)
        summary = {
            "ce_loss": self._epoch_sums["ce_loss"] / denom,
            "dice_loss": self._epoch_sums["dice_loss"] / denom,
            "total_loss": self._epoch_sums["total_loss"] / denom,
            "dwa_ce_weight": float(self.weights[0].detach().cpu().item()),
            "dwa_dice_weight": float(self.weights[1].detach().cpu().item()),
            "dwa_temperature": self.temperature,
        }
        if self._epoch_batches > 0:
            self.history.append(summary)
        self._last_epoch_summary = summary
        return summary

    def latest_epoch_summary(self) -> dict[str, float] | None:
        return self._last_epoch_summary

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        weights = self.weights.to(device=logits.device, dtype=logits.dtype)
        total = weights[0] * ce + weights[1] * dice
        if self._collect:
            self._epoch_sums["ce_loss"] += float(ce.detach().item())
            self._epoch_sums["dice_loss"] += float(dice.detach().item())
            self._epoch_sums["total_loss"] += float(total.detach().item())
            self._epoch_batches += 1
        return total


def build_auto_class_weights(
    counts: torch.Tensor,
    n_classes: int,
    *,
    is_backbone_frozen: bool,
    args: argparse.Namespace,
    cfg: dict,
) -> tuple[torch.Tensor, dict]:
    """Build CE class weights with safe absent-class handling.

    The generic rules are:
      - absent classes get a configurable weight, default 0
      - present classes use the configured static weighting scheme
      - optional max_ratio caps the present-class weight range when requested
      - binary segmentation keeps the existing minority boost behavior
    """
    weight_cfg = cfg.get("class_weighting", {}) or {}
    scheme = str(weight_cfg.get("scheme", "inverse_frequency")).lower()
    counts = counts.detach().float().cpu()
    if counts.numel() != n_classes:
        raise ValueError(f"class-count length {counts.numel()} does not match n_classes={n_classes}")

    present = counts > 0
    absent = ~present
    absent_weight = float(weight_cfg.get("absent_class_weight", 0.0))
    weights = torch.full((n_classes,), absent_weight, dtype=torch.float32)

    if bool(present.any()):
        present_count = int(present.sum().item())
        total_present = counts[present].sum().clamp(min=1.0)

        if scheme == "inverse_frequency":
            weights[present] = total_present / (present_count * counts[present].clamp(min=1.0))
        elif scheme in {"enet_log", "log_inverse"}:
            probs = counts[present] / total_present
            log_smoothing = float(weight_cfg.get("log_smoothing", 1.02))
            if log_smoothing <= 1.0:
                raise ValueError("class_weighting.log_smoothing must be > 1.0 for enet_log")
            weights[present] = 1.0 / torch.log(log_smoothing + probs.clamp(min=1e-12))
            weights[present] = weights[present] / weights[present].mean().clamp(min=1e-12)
        elif scheme == "median_frequency":
            probs = counts[present] / total_present
            weights[present] = probs.median() / probs.clamp(min=1e-12)
            weights[present] = weights[present] / weights[present].mean().clamp(min=1e-12)
        else:
            raise ValueError(
                f"Unknown class_weighting.scheme={scheme!r}; "
                "use inverse_frequency, enet_log, log_inverse, or median_frequency"
            )

        max_ratio = weight_cfg.get("max_ratio", None)
        if max_ratio is not None and float(max_ratio) > 0 and present_count > 1:
            min_present = weights[present].min()
            weights[present] = torch.minimum(weights[present], min_present * float(max_ratio))

    binary_boost = float(weight_cfg.get("binary_minority_boost", 2.0))
    if n_classes == 2 and bool(present.all()) and not is_backbone_frozen and binary_boost > 0:
        minority = int(counts.argmin().item())
        majority = 1 - minority
        boosted = weights[minority] * binary_boost
        if args.minority_boost_cap_ratio > 0:
            boosted = min(boosted, args.minority_boost_cap_ratio * weights[majority])
        weights[minority] = boosted

    info = {
        "mode": "auto",
        "scheme": scheme,
        "present_classes": [int(i) for i in present.nonzero(as_tuple=False).flatten().tolist()],
        "absent_classes": [int(i) for i in absent.nonzero(as_tuple=False).flatten().tolist()],
        "absent_class_weight": absent_weight,
        "max_ratio": weight_cfg.get("max_ratio", None),
        "log_smoothing": (
            float(weight_cfg.get("log_smoothing", 1.02))
            if scheme in {"enet_log", "log_inverse"} else None
        ),
        "binary_minority_boost": binary_boost if n_classes == 2 else None,
        "binary_minority_boost_cap_ratio": (
            float(args.minority_boost_cap_ratio) if n_classes == 2 else None
        ),
        "train_class_counts": [int(x) for x in counts.long().tolist()],
        "train_class_ratios": class_ratios_from_counts(counts),
    }
    return weights, info


def build_dice_class_mask(
    counts: torch.Tensor,
    n_classes: int,
    cfg: dict,
) -> tuple[torch.Tensor | None, str, bool]:
    dice_cfg = cfg.get("dice", {}) or {}
    class_mode = str(dice_cfg.get("class_mode", "default"))
    include_background = bool(dice_cfg.get("include_background", False))

    if class_mode in {"present", "present_train"}:
        class_mask = (counts.detach().cpu() > 0).bool()
    elif class_mode == "all":
        class_mask = torch.ones(n_classes, dtype=torch.bool)
    elif class_mode in {"default", "foreground"}:
        class_mask = None
    else:
        raise ValueError(
            f"Unknown dice.class_mode={class_mode!r}; use default, foreground, all, or present_train"
        )
    return class_mask, class_mode, include_background


def set_optimizer_group_lr(optimizer: torch.optim.Optimizer, group_name: str, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            group["lr"] = lr


def optimizer_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float | None:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return None


def configure_residual_stage(
    epoch_idx: int,
    bre: BRE | None,
    optimizer: torch.optim.Optimizer,
    force_residual_zero: bool,
    stage1_epochs: int,
    stage2_lr_residual: float | None,
    default_residual_lr: float,
    ramp_epochs: int,
) -> dict:
    if bre is None:
        return {}

    if epoch_idx < stage1_epochs:
        bre.force_delta_zero = True
        bre.residual_scale = 0.0
        set_optimizer_group_lr(optimizer, "residual_R", 0.0)
        return {
            "residual_stage": "bandsel",
            "residual_scale": 0.0,
            "residual_lr_active": optimizer_group_lr(optimizer, "residual_R"),
        }

    stage2_epoch = epoch_idx - stage1_epochs
    bre.force_delta_zero = force_residual_zero
    if force_residual_zero:
        gamma = 0.0
    elif ramp_epochs > 0:
        gamma = min(1.0, float(stage2_epoch + 1) / float(ramp_epochs))
    else:
        gamma = 1.0
    bre.residual_scale = gamma

    active_lr = stage2_lr_residual if stage2_lr_residual is not None else default_residual_lr
    set_optimizer_group_lr(optimizer, "residual_R", active_lr)
    return {
        "residual_stage": "residual",
        "residual_scale": gamma,
        "residual_lr_active": optimizer_group_lr(optimizer, "residual_R"),
    }


def train_epoch(model, loader, optimizer, criterion, device: torch.device,
                max_grad_norm: float = 1.0,
                grad_clip_mode: str = "global") -> tuple[float, int]:
    model.train()
    if hasattr(criterion, "begin_epoch"):
        criterion.begin_epoch()
    total_loss = 0.0
    n_batches  = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if hasattr(outputs, "output"):      # TerraTorch ModelOutput
            outputs = outputs.output
        elif isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]
        loss = criterion(outputs, labels)
        loss.backward()
        if max_grad_norm > 0:
            if grad_clip_mode == "per_group":
                for group in optimizer.param_groups:
                    group_params = [p for p in group["params"] if p.grad is not None]
                    if group_params:
                        torch.nn.utils.clip_grad_norm_(group_params, max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None],
                    max_grad_norm,
                )
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    if hasattr(criterion, "end_epoch"):
        criterion.end_epoch()
    return total_loss / max(n_batches, 1), n_batches


def evaluate(model, loader, metric_name: str, device: torch.device) -> float:
    """Backward-compatible wrapper: returns mIoU as a scalar."""
    return evaluate_full(model, loader, device, criterion=None)["miou"]


def evaluate_full(model, loader, device: torch.device,
                  criterion=None) -> dict:
    """Compute val mIoU, per-class IoU, positive-prediction ratio, and (optionally)
    average loss. Returns a dict so the training loop can store all of them per epoch.
    """
    model.eval()
    total_inter = None
    total_union = None
    pred_counts  = None
    label_counts = None
    loss_sum     = 0.0
    n_batches    = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if hasattr(outputs, "output"):
                outputs = outputs.output
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]

            if criterion is not None:
                loss_sum += float(criterion(outputs, labels).item())
                n_batches += 1

            preds = outputs.argmax(dim=1)
            n_classes = outputs.shape[1]

            mask = labels != -1
            preds_m  = preds[mask]
            labels_m = labels[mask]

            if total_inter is None:
                total_inter  = torch.zeros(n_classes, device=device)
                total_union  = torch.zeros(n_classes, device=device)
                pred_counts  = torch.zeros(n_classes, device=device)
                label_counts = torch.zeros(n_classes, device=device)

            for c in range(n_classes):
                pred_c  = preds_m  == c
                label_c = labels_m == c
                total_inter[c] += (pred_c & label_c).sum()
                total_union[c] += (pred_c | label_c).sum()
                pred_counts[c]  += pred_c.sum()
                label_counts[c] += label_c.sum()

    if total_inter is None:
        return {"miou": 0.0, "per_class_iou": [], "pos_pred_ratio": float("nan"),
                "loss": float("nan"), "pred_dist": [], "pred_counts": [],
                "label_counts": [], "pred_class_ratios": [],
                "label_class_ratios": [], "class_area_bias": []}

    iou = total_inter / (total_union + 1e-6)
    per_class_iou = iou.detach().cpu().tolist()
    miou = iou[total_union > 0].mean().item()

    total_preds = pred_counts.sum().clamp(min=1)
    total_labels = label_counts.sum().clamp(min=1)
    pred_frac = (pred_counts / total_preds).detach().cpu().tolist()
    label_frac = (label_counts / total_labels).detach().cpu().tolist()
    class_area_bias = ((pred_counts / total_preds) - (label_counts / total_labels)).detach().cpu().tolist()
    # "Positive-prediction ratio" defaults to the share of the LAST class (the
    # minority class for binary segmentation; the convention matches sen1floods11
    # where class 1 = flood). For >2-class segmentation, this is just pred_frac[-1].
    pos_pred_ratio = float(pred_frac[-1])
    if pred_counts[0] / total_preds > 0.98:
        logger.warning("  STUCK: %.1f%% of predictions are class 0 (trivial predictor)",
                       100 * pred_counts[0].item() / total_preds.item())

    avg_loss = (loss_sum / max(n_batches, 1)) if criterion is not None else float("nan")
    return {
        "miou": miou,
        "per_class_iou": per_class_iou,
        "pos_pred_ratio": pos_pred_ratio,
        "pred_dist": pred_frac,
        "pred_counts": [int(x) for x in pred_counts.detach().cpu().tolist()],
        "label_counts": [int(x) for x in label_counts.detach().cpu().tolist()],
        "pred_class_ratios": [float(x) for x in pred_frac],
        "label_class_ratios": [float(x) for x in label_frac],
        "class_area_bias": [float(x) for x in class_area_bias],
        "loss": avg_loss,
    }


def run_post_training_test_eval(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    checkpoint_kind: str,
    run_id: str,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    loader_seed: int,
    split_seed: int,
    n_classes: int,
) -> dict:
    """Evaluate a saved best/final checkpoint on test and save visualizations."""
    eval_run_id = f"{run_id}_{checkpoint_kind}_test"
    checkpoint_load_info = load_adapted_checkpoint(model, checkpoint_path, device)
    test_loader = build_dataloader(
        cfg["dataset"],
        "test",
        cfg,
        loader_seed,
        split_seed,
    )
    metrics = evaluate_segmentation(
        model,
        test_loader,
        device,
        foreground_class=1,
        ignore_index=-1,
    )
    vis_alpha = (
        float(args.auto_test_vis_alpha)
        if args.auto_test_vis_alpha is not None
        else (0.7 if n_classes > 2 else 0.45)
    )
    visualizations = None
    if int(args.auto_test_vis_examples) > 0:
        visualizations = save_segmentation_visualizations(
            model,
            test_loader,
            device,
            dataset_name=cfg["dataset"],
            run_id=eval_run_id,
            out_dir=args.auto_test_vis_dir,
            max_examples=int(args.auto_test_vis_examples),
            alpha=vis_alpha,
            foreground_class=1,
            ignore_index=-1,
        )

    result = {
        "run_id": eval_run_id,
        "dataset": cfg["dataset"],
        "backbone": cfg["backbone"],
        "method": getattr(args, "public_method", args.method),
        "split": "test",
        "seed": args.seed,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint": str(checkpoint_path),
        "checkpoint_load": checkpoint_load_info,
        "metrics": metrics,
        "test_miou": metrics["miou"],
        "test_macro_iou": metrics.get("macro_iou"),
        "test_macro_precision": metrics.get("macro_precision"),
        "test_macro_recall": metrics.get("macro_recall"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_mean_class_accuracy": metrics.get("mean_class_accuracy"),
        "test_foreground_iou": metrics.get("foreground_iou"),
        "test_pred_positive_ratio": metrics.get("pred_positive_ratio"),
        "test_true_positive_ratio": metrics.get("true_positive_ratio"),
        "test_per_class_iou": metrics.get("per_class_iou"),
        "test_per_class_precision": metrics.get("per_class_precision"),
        "test_per_class_recall": metrics.get("per_class_recall"),
        "test_per_class_f1": metrics.get("per_class_f1"),
        "test_pred_counts": metrics.get("pred_counts"),
        "test_label_counts": metrics.get("label_counts"),
        "test_pred_class_ratios": metrics.get("pred_class_ratios"),
        "test_label_class_ratios": metrics.get("label_class_ratios"),
        "test_class_area_bias": metrics.get("class_area_bias"),
    }
    if visualizations is not None:
        result["visualizations"] = visualizations

    out_dir = RESULTS_DIR / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{eval_run_id}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(
        "Auto test %s: mIoU=%.4f macroF1=%s -> %s",
        checkpoint_kind,
        metrics["miou"],
        metrics.get("macro_f1"),
        out_path,
    )
    return {
        "run_id": eval_run_id,
        "json_path": str(out_path),
        "checkpoint": str(checkpoint_path),
        "test_miou": metrics["miou"],
        "test_macro_f1": metrics.get("macro_f1"),
        "test_mean_class_accuracy": metrics.get("mean_class_accuracy"),
        "visualization_dir": visualizations.get("dir") if visualizations is not None else None,
    }


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    seed = args.seed
    if args.epochs_override is not None:
        cfg["epochs"] = args.epochs_override

    split_seed = resolve_seed(args.split_seed, seed)
    model_seed = resolve_seed(args.model_seed, seed)
    residual_seed = resolve_seed(args.residual_seed, seed)
    lora_seed = resolve_seed(args.lora_seed, seed)
    head_seed = resolve_seed(args.head_seed, seed)
    loader_seed = resolve_seed(args.loader_seed, seed)
    train_shuffle_seed = resolve_seed(args.train_shuffle_seed, seed)
    eval_shuffle_seed = resolve_seed(args.eval_shuffle_seed, seed)
    resolved_seed_protocol = {
        "seed": seed,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "residual_seed": residual_seed,
        "lora_seed": lora_seed,
        "head_seed": head_seed,
        "loader_seed": loader_seed,
        "train_shuffle_seed": train_shuffle_seed,
        "eval_shuffle_seed": eval_shuffle_seed,
    }
    set_all_seeds(model_seed)

    dataset  = cfg["dataset"]
    backbone_name = cfg["backbone"]
    run_id = args.run_id or f"{dataset}_{backbone_name}_{args.method}_s{seed}"
    log_path = setup_file_logging(run_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Log file: %s", log_path)
    logger.info("Device: %s  Method: %s  Seed: %d", device, args.method, seed)
    logger.info(
        "Seeds: split=%s model=%s residual=%s lora=%s head=%s loader=%s train_shuffle=%s eval_shuffle=%s",
        split_seed,
        model_seed,
        residual_seed,
        lora_seed,
        head_seed,
        loader_seed,
        train_shuffle_seed,
        eval_shuffle_seed,
    )
    logger.info("Config: %s", args.config)
    if args.auto_test and not args.save_checkpoints:
        args.save_checkpoints = True
        logger.info("Auto-test is enabled; enabling --save-checkpoints for best/final test evaluation.")

    spec     = BACKBONE_SPECS[backbone_name]
    band_cfg = FULL_BAND_CONFIGS[dataset]
    srf_triples = torch.from_numpy(get_srf_triples(band_cfg.sensor_key)).to(device)

    # Number of classes for this dataset
    DATASET_N_CLASSES = {
        "fire_scars": 2, "sen1floods11": 2, "abi_cloud": 4,
        "multitemporal_crop": 14, "cloudsen12": 4,
        "bigearthnet": 19, "loveda": 7, "landslide4sense": 2,
        "geobench_sa_crop_type": 10,
    }
    n_classes = DATASET_N_CLASSES.get(dataset, 2)
    model_architecture = model_architecture_defaults(backbone_name)

    # Build model components
    model = load_backbone(backbone_name, n_classes, device)

    # Public methods use native input, band selection baselines, or BRE.
    if args.method.endswith("_bandsel"):
        use_band_select = True
        use_bre = False
        _sched_method = args.method[:-len("_bandsel")]
    elif args.method.endswith("_bre"):
        use_band_select = False
        use_bre = True
        _sched_method = args.method[:-len("_bre")]
    else:
        use_band_select = False
        use_bre = False
        _sched_method = args.method

    band_selector = None
    band_selector = None
    bandsel_indices = None
    if use_band_select or use_bre:
        # Pick the C_pretrain bands of the target sensor whose central wavelengths
        # are closest to the pre-training sensor's bands.
        from spectra.data.srf import select_closest_bands, get_srf
        pretrain_sensor = BACKBONE_PRETRAIN_SENSOR.get(backbone_name, "fire_scars")
        pretrain_bands = get_srf(pretrain_sensor)
        ref_wavelengths = [b.center_nm for b in pretrain_bands]
        bandsel_indices = select_closest_bands(band_cfg.sensor_key, ref_wavelengths)
        if (
            use_bre
            and len(bandsel_indices) == band_cfg.in_chans
            and list(bandsel_indices) == list(range(band_cfg.in_chans))
        ):
            use_bre = False
            use_band_select = False
            bre_mode = "native"
            logger.info("BRE bypassed: target bands exactly match the pretrained patch embedding input; using native patch_embed + ST-LoRA.")

    if use_band_select:
        band_selector = BandSelector(bandsel_indices).to(device)
        logger.info("BandSelector: pretrain_sensor=%s ref_λ=%s → target_sensor=%s indices=%s",
                    pretrain_sensor, ref_wavelengths, band_cfg.sensor_key, bandsel_indices)

    bre = None
    bre_mode = "bre" if use_bre else "native"
    if use_bre:
        set_all_seeds(residual_seed)
        original_pe = model.encoder.patch_embed
        extra_idx = [i for i in range(band_cfg.in_chans) if i not in bandsel_indices]
        bre = BRE(
            original_patch_embed=original_pe,
            selected_idx=bandsel_indices,
            in_chans_full=band_cfg.in_chans,
            hidden_dim=cfg.get("vres_hidden_dim", 32),
            gate_max=float(cfg.get("bre_gate_max", 2.0)),
        ).to(device)
        bre.force_delta_zero = args.force_residual_zero
        bre.eval_shuffle_seed = int(eval_shuffle_seed)
        bre.set_train_shuffle_seed(int(train_shuffle_seed))
        model.encoder.patch_embed = bre
        n_r_params = sum(p.numel() for p in residual_adapter_params(bre))
        logger.info(
            "BRE wired: adapter params=%d selected=%s extra=%s force_delta_zero=%s train_shuffle_seed=%s eval_shuffle_seed=%s",
            n_r_params, bandsel_indices, extra_idx, bre.force_delta_zero,
            bre.train_shuffle_seed, bre.eval_shuffle_seed,
        )

    # --- : separate code path (no adapter, no nested LoRA) ---
    # --- NestedLoRA path ---


    if use_band_select:
        wire_band_selector(model, band_selector)
    elif use_bre:
        logger.info("BRE already replaced encoder.patch_embed; adapter params will be added to optimizer below")
    else:
        logger.info("No input adapter — using native backbone patch_embed")

    lora_cfg = cfg.get("lora", {})
    if args.max_rank is not None:
        lora_cfg["max_rank"] = args.max_rank
    effective_max_rank = lora_cfg.get("max_rank", 16)
    if args.lora_alpha is not None:
        lora_alpha = args.lora_alpha
    else:
        # Default: keep scaling = 1/16 regardless of max_rank, so rank-grid sweeps isolate
        # the effect of available capacity rather than confounding it with scaling shrinkage.
        lora_alpha = effective_max_rank / 16.0

    if args.rank_grid is not None:
        rank_grid = [int(x.strip()) for x in args.rank_grid.split(",")]
        from spectra.adapter.nested_lora import set_rank_grid
        set_rank_grid(rank_grid)
        logger.info("Stochastic rank grid set to %s", rank_grid)
    else:
        rank_grid = None

    set_all_seeds(lora_seed)

    lora_backbone = NestedLoRABackbone(
        model.encoder,
        blocks_attr = "blocks",
        n_stages    = lora_cfg.get("n_stages", 4),
        max_rank    = effective_max_rank,
        freeze_all  = True,
        alpha       = lora_alpha,
    ).to(device)   # LoRA A/B params initialized on CPU; move to GPU
    logger.info("NestedLoRA built: max_rank=%d  alpha=%.3f  scaling=%.5f",
                effective_max_rank, lora_alpha, lora_alpha / effective_max_rank)

    n_head_reset = reset_task_head(model, head_seed)
    logger.info("Task head reset with head_seed=%d (%d modules)", head_seed, n_head_reset)

    if use_bre and bre is not None:
        r_trainable = not args.freeze_residual
        adapter_params_for_grad = residual_adapter_params(bre)
        for p in adapter_params_for_grad:
            p.requires_grad_(r_trainable)
        logger.info(
            "%s: adapter trainable=%s (%d params)",
            bre.__class__.__name__,
            r_trainable,
            sum(p.numel() for p in adapter_params_for_grad),
        )

    patch_embed_module, patch_embed_all_params = pretrained_patch_embed_params(model)
    patch_embed_params: list[nn.Parameter] = []
    if args.train_patch_embed:
        if not patch_embed_all_params:
            logger.warning("--train-patch-embed requested, but no pretrained patch_embed parameters were found")
        else:
            for p in patch_embed_all_params:
                p.requires_grad_(True)
            patch_embed_params = [p for p in patch_embed_all_params if p.requires_grad]
            logger.info(
                "Pretrained patch_embed trainable: module=%s params=%d tensors=%d",
                patch_embed_module.__class__.__name__ if patch_embed_module is not None else None,
                sum(p.numel() for p in patch_embed_params),
                len(patch_embed_params),
            )

    # Head params needed by SPECTRA warmup optimizer and Phase 4 — compute once here
    head_params = []
    for name, p in model.named_parameters():
        if not name.startswith("encoder."):
            head_params.append(p)

    # Data loaders
    train_loader = build_dataloader(
        dataset,
        "train",
        cfg,
        loader_seed,
        split_seed,
        positive_batch_min=args.positive_batch_min,
        positive_crop_max_tries=args.positive_crop_max_tries,
    )
    val_loader   = build_dataloader(dataset, "val",   cfg, loader_seed, split_seed)
    split_label_counts = {
        split_name: count_split_labels(dataset, split_name, cfg, n_classes, split_seed)
        for split_name in ("train", "val", "test")
    }
    split_class_counts = {
        split_name: [int(x) for x in counts.long().tolist()]
        for split_name, counts in split_label_counts.items()
    }
    split_class_ratios = {
        split_name: class_ratios_from_counts(counts)
        for split_name, counts in split_label_counts.items()
    }
    split_positive_ratios = {
        split_name: positive_ratio(counts)
        for split_name, counts in split_label_counts.items()
    }

    t_start = time.time()

    if _sched_method == "spectra" and args.ranks is not None:
        # Manual-plan override: skip STPlanner profiling and apply the provided ranks/unfrozen.
        # Use case: replay the plan from a previous SPECTRA run with a different adapter,
        # or sidestep planner failure modes (e.g., random MLP front-end producing useless Δq).
        plan_ranks = [int(r.strip()) for r in args.ranks.split(",")]
        if args.unfrozen is not None:
            plan_unfrozen = [u.strip().upper() == "T" for u in args.unfrozen.split(",")]
        else:
            plan_unfrozen = [False] * len(plan_ranks)
        if max(plan_ranks, default=0) > effective_max_rank:
            raise ValueError(
                f"Manual planned rank {max(plan_ranks)} exceeds LoRA max_rank={effective_max_rank}. "
                "Use --max-rank at least as large as the planned rank."
            )
        logger.info("Manual-plan override (skipping STPlanner): ranks=%s unfrozen=%s",
                    plan_ranks, plan_unfrozen)
        lora_backbone.apply_schedule(plan_ranks, plan_unfrozen, fix_train_rank=True)
        schedule_info = {"method": getattr(args, "public_method", args.method), "ranks": plan_ranks, "unfrozen": plan_unfrozen,
                         "plan_source": "manual_override"}

    elif _sched_method == "spectra":
        planner_cfg = cfg.get("stlora", {})
        profiler = StagewiseLogMEProfiler(
            lora_backbone=lora_backbone,
            patch_size=spec.patch_size,
            purity_tau=planner_cfg.get("purity_tau", 0.8),
            n_probe_images=planner_cfg.get("n_probe_images", 1000),
        )
        logger.info("Profiling pretrained backbone for STPlanner...")
        profile = profiler.profile_only(train_loader)
        logger.info("LogME profile q[s]: %s", [f"{q:.4f}" for q in profile.scores])
        logger.info("Δq[s]: %s",              [f"{dq:.4f}" for dq in profile.delta_q()])
        logger.info("Precondition OK: %s", profile.precondition_ok)

        stage_dims, stage_n_params, t_lora, t_full = _load_stage_costs(backbone_name, spec)
        star_stage_prior = tuple(float(x.strip()) for x in args.star_stage_prior.split(","))
        star_budget_candidates = tuple(int(x.strip()) for x in args.star_budget_candidates.split(","))
        star_config = STPlannerConfig(
            strategy=args.star_planner,
            reference_rank=args.star_reference_rank,
            tau=args.star_tau,
            stage_prior=star_stage_prior,
            rank_grid=(4, 8, 16, 32, 64),
            candidate_budgets=star_budget_candidates,
            min_rank=4,
            budget_f_min=args.star_budget_f_min,
            budget_f_max=args.star_budget_f_max,
            budget_midpoint=args.star_budget_midpoint,
            budget_slope=args.star_budget_slope,
            q_bank_csv=args.star_q_bank_csv,
            q_min_bank=args.star_q_min,
            q_max_bank=args.star_q_max,
            budget_override=args.star_budget_override,
            n_stages=lora_cfg.get("n_stages", 4),
        )
        planner = STPlanner(config=star_config)
        plan = planner.plan(profile, spec.embed_dim, stage_dims, stage_n_params, t_lora, t_full)
        if max(plan.ranks, default=0) > effective_max_rank:
            raise ValueError(
                f"ST-LoRA planned rank {max(plan.ranks)} exceeds LoRA max_rank={effective_max_rank}. "
                "Use --max-rank at least as large as the planned rank."
            )
        lora_backbone.apply_schedule(plan.ranks, plan.unfrozen, fix_train_rank=True)
        schedule_info = {
            "method": getattr(args, "public_method", args.method),
            "schedule_method": "spectra",
            "plan_source": f"stlora_{args.star_planner}",
            "ranks": plan.ranks,
            "unfrozen": plan.unfrozen,
            "precondition_ok": profile.precondition_ok,
            "profile_scores": [round(q, 6) for q in profile.scores],
            "delta_q": [round(dq, 6) for dq in plan.delta_q],
            "profile_n_patches_used": profile.n_patches_used,
            "profile_used_fallback": profile.used_fallback,
            "profile_stopped_at": profile.stopped_at,
            "stlora_param_fraction": plan.param_fraction,
            "stlora_gpu_fraction": plan.gpu_fraction,
            "stlora": {
                "strategy": plan.strategy,
                "reference_rank": star_config.reference_rank,
                "max_budget": star_config.reference_rank * star_config.n_stages,
                "budget": plan.budget,
                "budget_raw": plan.budget_raw,
                "budget_fraction": plan.budget_fraction,
                "budget_override": star_config.budget_override,
                "candidate_budgets": list(star_config.candidate_budgets),
                "q_overall": plan.q_overall,
                "q_norm": plan.q_norm,
                "q_min_bank": plan.q_min_bank,
                "q_max_bank": plan.q_max_bank,
                "tau": star_config.tau,
                "stage_prior": plan.stage_prior,
                "rank_grid": list(star_config.rank_grid),
                "min_rank": star_config.min_rank,
                "logits": [round(x, 6) for x in plan.logits],
                "weights": [round(w, 6) for w in plan.weights],
                "continuous_ranks": [round(r, 6) for r in plan.continuous_ranks],
                "budget_f_min": star_config.budget_f_min,
                "budget_f_max": star_config.budget_f_max,
                "budget_midpoint": star_config.budget_midpoint,
                "budget_slope": star_config.budget_slope,
                "q_bank_csv": str(star_config.q_bank_csv) if star_config.q_bank_csv else None,
            },
        }

    else:
        sched = apply_baseline_schedule(_sched_method, lora_backbone, cfg)
        schedule_info = {"method": getattr(args, "public_method", args.method), "ranks": sched.ranks, "unfrozen": list(sched.unfrozen)}

    checkpoint_load_info = None
    if args.load_adapted_checkpoint is not None:
        checkpoint_load_info = load_adapted_checkpoint(model, args.load_adapted_checkpoint, device)

    # Phase 4: Fine-tune — head always trainable; adapter/backbone at lower LR

    # Split backbone trainable params: LoRA A/B adapters (lr_adapter) vs unfrozen backbone
    # weights (lr_backbone).
    lora_params = lora_backbone.lora_adapter_params()
    backbone_params = lora_backbone.unfrozen_backbone_params()
    if use_bre and bre is not None:
        bre_param_ids = {id(p) for p in residual_adapter_params(bre)}
        backbone_params = [p for p in backbone_params if id(p) not in bre_param_ids]
    if patch_embed_params:
        patch_embed_param_ids = {id(p) for p in patch_embed_params}
        backbone_params = [p for p in backbone_params if id(p) not in patch_embed_param_ids]

    residual_lr = args.lr_residual_override
    if residual_lr is None:
        residual_lr = cfg.get("lr_residual", cfg.get("lr_head", 1e-3))
    if args.stage2_lr_residual is not None:
        residual_lr = args.stage2_lr_residual
    adapter_lr = args.lr_adapter_override if args.lr_adapter_override is not None else cfg.get("lr_adapter", 1e-4)
    backbone_lr = args.lr_backbone_override if args.lr_backbone_override is not None else cfg.get("lr_backbone", 1e-5)
    head_lr = args.lr_head_override if args.lr_head_override is not None else cfg.get("lr_head", 1e-3)
    patch_embed_lr = args.lr_patch_embed_override if args.lr_patch_embed_override is not None else backbone_lr
    residual_weight_decay = (
        args.weight_decay_residual_override
        if args.weight_decay_residual_override is not None
        else cfg.get("weight_decay", 0.05)
    )
    patch_embed_weight_decay = (
        args.weight_decay_patch_embed_override
        if args.weight_decay_patch_embed_override is not None
        else 0.0
    )

    param_groups = []
    if use_bre and bre is not None:
        residual_params = [p for p in residual_adapter_params(bre) if p.requires_grad]
        if residual_params:
            param_groups.append({
                "params": residual_params,
                "lr": residual_lr,
                "weight_decay": residual_weight_decay,
                "name": "residual_R",
            })
    if patch_embed_params:
        param_groups.append({
            "params": patch_embed_params,
            "lr": patch_embed_lr,
            "weight_decay": patch_embed_weight_decay,
            "name": "patch_embed",
        })
    if lora_params:
        param_groups.append({"params": lora_params, "lr": adapter_lr, "name": "lora"})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 0.05))

    r_params_all = residual_adapter_params(bre) if (use_bre and bre is not None) else []
    r_optimizer_occurrences = optimizer_param_occurrences(optimizer, r_params_all) if r_params_all else 0
    patch_embed_optimizer_occurrences = (
        optimizer_param_occurrences(optimizer, patch_embed_params) if patch_embed_params else 0
    )
    logger.info(
        "Param groups: bre=%d lora=%d patch_embed=%d backbone=%d head=%d",
        sum(p.numel() for p in r_params_all),
        len(lora_params),
        len(patch_embed_params),
        len(backbone_params),
        len(head_params),
    )
    logger.info(
        "BRE optimizer audit: adapter params=%d adapter lr=%s optimizer occurrences=%d freeze=%s force_zero=%s",
        sum(p.numel() for p in r_params_all),
        residual_lr if r_params_all and not args.freeze_residual else None,
        r_optimizer_occurrences,
        args.freeze_residual,
        args.force_residual_zero,
    )
    logger.info(
        "PatchEmbed optimizer audit: trainable=%s params=%d lr=%s weight_decay=%s optimizer occurrences=%d",
        bool(patch_embed_params),
        sum(p.numel() for p in patch_embed_params),
        patch_embed_lr if patch_embed_params else None,
        patch_embed_weight_decay if patch_embed_params else None,
        patch_embed_optimizer_occurrences,
    )
    loss_mode = args.loss_mode
    loss_mode = args.loss_mode
    cw_cfg = cfg.get("class_weights", None) if loss_mode in {"weighted_ce", "ce_dice", "ce_dice_dwa"} else None
    class_weights_for_log = None
    class_weight_info = {"mode": "none" if loss_mode != "ce" else "disabled_by_loss_mode"}
    if loss_mode == "ce":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif cw_cfg == "auto":
        counts = split_label_counts["train"]
        is_backbone_frozen = (not lora_params) and (not backbone_params)
        cw, class_weight_info = build_auto_class_weights(
            counts,
            n_classes,
            is_backbone_frozen=is_backbone_frozen,
            args=args,
            cfg=cfg,
        )
        logger.info("Class weights (auto): %s", cw.tolist())
        logger.info("Class weight info: %s", class_weight_info)
        class_weights_for_log = [float(x) for x in cw.tolist()]
        criterion = torch.nn.CrossEntropyLoss(weight=cw.to(device), ignore_index=-1)
    elif isinstance(cw_cfg, list):
        cw = torch.tensor(cw_cfg, dtype=torch.float32)
        logger.info("Class weights (manual): %s", cw.tolist())
        class_weights_for_log = [float(x) for x in cw.tolist()]
        class_weight_info = {
            "mode": "manual",
            "present_classes": [int(i) for i in (split_label_counts["train"] > 0).nonzero(as_tuple=False).flatten().tolist()],
            "absent_classes": [int(i) for i in (split_label_counts["train"] <= 0).nonzero(as_tuple=False).flatten().tolist()],
            "train_class_counts": [int(x) for x in split_label_counts["train"].long().tolist()],
            "train_class_ratios": split_class_ratios["train"],
        }
        criterion = torch.nn.CrossEntropyLoss(weight=cw.to(device), ignore_index=-1)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        class_weight_info = {
            "mode": "unweighted_ce",
            "reason": "class_weights config is not auto/list",
        }
    dice_class_mask, dice_class_mode, dice_include_background_cfg = build_dice_class_mask(
        split_label_counts["train"],
        n_classes,
        cfg,
    )
    dice_include_background = bool(args.dice_include_background or dice_include_background_cfg)
    ce_weighted_active = bool(
        loss_mode in {"weighted_ce", "ce_dice", "ce_dice_dwa"} and (cw_cfg == "auto" or isinstance(cw_cfg, list))
    )
    ce_formula = "CE_weighted" if ce_weighted_active else "CE"
    loss_config = {
        "loss_mode": loss_mode,
        "loss_formula": (
            "CE"
            if loss_mode == "ce"
            else ce_formula
            if loss_mode == "weighted_ce"
            else f"DWA({ce_formula}, DiceLoss)"
            if loss_mode == "ce_dice_dwa"
            else f"{ce_formula} + dice_lambda * DiceLoss"
        ),
        "ce_weighted": ce_weighted_active,
        "dice_lambda": float(args.dice_lambda) if loss_mode == "ce_dice" else 0.0,
        "dwa_temperature": float(args.dwa_temperature) if loss_mode == "ce_dice_dwa" else None,
        "dice_include_background": dice_include_background,
        "dice_class_mode": dice_class_mode,
        "dice_classes": (
            [int(i) for i in dice_class_mask.nonzero(as_tuple=False).flatten().tolist()]
            if dice_class_mask is not None else None
        ),
        "class_weighting": class_weight_info,
    }
    if loss_mode == "ce_dice" and args.dice_lambda > 0:
        criterion = CombinedLoss(
            criterion,
            DiceLoss(
                ignore_index=-1,
                include_background=dice_include_background,
                class_mask=dice_class_mask,
            ),
            args.dice_lambda,
        )
    elif loss_mode == "ce_dice_dwa":
        criterion = DynamicWeightAverageLoss(
            criterion,
            DiceLoss(
                ignore_index=-1,
                include_background=dice_include_background,
                class_mask=dice_class_mask,
            ),
            temperature=args.dwa_temperature,
        )
    logger.info("Loss config: %s", loss_config)
    staged_residual_config = {
        "stage1_bandsel_epochs": int(args.stage1_bandsel_epochs),
        "stage2_lr_residual": args.stage2_lr_residual,
        "residual_gamma_ramp_epochs": int(args.residual_gamma_ramp_epochs),
    }
    logger.info("Staged residual config: %s", staged_residual_config)

    input_adapter_name = ("bre" if use_bre else "bandsel" if use_band_select else "native")
    r_param_count = sum(p.numel() for p in r_params_all)
    r_optimizer_lr = residual_lr if r_param_count and r_optimizer_occurrences > 0 else None
    patch_embed_param_count = sum(p.numel() for p in patch_embed_params)
    patch_embed_optimizer_lr = (
        patch_embed_lr if patch_embed_param_count and patch_embed_optimizer_occurrences > 0 else None
    )
    patch_embed_optimizer_weight_decay = (
        patch_embed_weight_decay if patch_embed_optimizer_lr is not None else None
    )
    selected_idx_for_log = list(bandsel_indices) if bandsel_indices is not None else None
    extra_idx_for_log = (
        [i for i in range(band_cfg.in_chans) if i not in bandsel_indices]
        if bandsel_indices is not None else None
    )
    lora_init_fingerprint = first_lora_norms(lora_backbone)
    residual_fingerprint = residual_init_norms(bre if use_bre else None)
    schedule_ranks_for_log = schedule_info.get("ranks") if isinstance(schedule_info, dict) else None
    effective_lora_param_count = active_lora_param_count(lora_backbone, schedule_ranks_for_log)
    actual_lora_tensor_param_count = param_count(lora_params)
    seed_protocol = dict(resolved_seed_protocol)
    if not use_bre:
        seed_protocol["residual_seed"] = None
        seed_protocol["train_shuffle_seed"] = None
        seed_protocol["eval_shuffle_seed"] = None
    fingerprints = {
        "lora_param_count": actual_lora_tensor_param_count,
        "actual_lora_tensor_param_count": actual_lora_tensor_param_count,
        "effective_active_lora_param_count": effective_lora_param_count,
        "lora_rank_schedule": schedule_ranks_for_log,
        "head_param_count": param_count(head_params),
        "residual_param_count": r_param_count,
        "residual_lr": r_optimizer_lr,
        "residual_params_in_optimizer_once": (
            (r_optimizer_occurrences == len(r_params_all)) if r_params_all else None
        ),
        "patch_embed_trainable": bool(patch_embed_params),
        "patch_embed_param_count": patch_embed_param_count,
        "patch_embed_lr": patch_embed_optimizer_lr,
        "patch_embed_weight_decay": patch_embed_optimizer_weight_decay,
        "patch_embed_params_in_optimizer_once": (
            (patch_embed_optimizer_occurrences == len(patch_embed_params)) if patch_embed_params else None
        ),
        "selected_idx": selected_idx_for_log,
        "extra_idx": extra_idx_for_log,
        "class_weights": class_weights_for_log or [],
        "class_weighting": class_weight_info,
        "model_architecture": model_architecture,
        "decoder": model_architecture.get("decoder"),
        "feature_indices": model_architecture.get("feature_indices"),
        "split_class_counts": split_class_counts,
        "split_class_ratios": split_class_ratios,
        "train_class_counts": split_class_counts["train"],
        "val_class_counts": split_class_counts["val"],
        "test_class_counts": split_class_counts["test"],
        "train_class_ratios": split_class_ratios["train"],
        "val_class_ratios": split_class_ratios["val"],
        "test_class_ratios": split_class_ratios["test"],
        "train_positive_pixel_ratio": split_positive_ratios["train"],
        "val_positive_pixel_ratio": split_positive_ratios["val"],
        "test_positive_pixel_ratio": split_positive_ratios["test"],
        "positive_batch_min": int(args.positive_batch_min),
        "positive_crop_max_tries": int(args.positive_crop_max_tries),
        "head_weight_norm": head_weight_norm(model),
        **lora_init_fingerprint,
        **residual_fingerprint,
    }
    if use_bre and bre is not None and hasattr(bre, "contribution_matrix_full"):
        fingerprints["router_candidate_idx"] = [int(x) for x in bre.candidate_idx.detach().cpu().tolist()]
        fingerprints["router_initial_contribution_matrix"] = bre.contribution_matrix_full()
        fingerprints["router_initial_top3"] = bre.top_contributions(k=3)
    diagnostic_fingerprints = {
        "seed_protocol": seed_protocol,
        "fingerprints": fingerprints,
        "loss_config": loss_config,
        "staged_residual": staged_residual_config,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "residual_seed": residual_seed if use_bre else None,
        "lora_seed": lora_seed,
        "head_seed": head_seed,
        "model_init_seed": model_seed,
        "residual_init_seed": residual_seed if use_bre else None,
        "lora_init_seed": lora_seed,
        "head_init_seed": head_seed,
        "loader_seed": loader_seed,
        "head_weight_norm": fingerprints["head_weight_norm"],
        "R_param_count": r_param_count,
        "R_optimizer_lr": r_optimizer_lr,
        "R_optimizer_weight_decay": residual_weight_decay if r_optimizer_lr is not None else None,
        "R_optimizer_occurrences": r_optimizer_occurrences,
        "R_optimizer_exactly_once": (r_optimizer_occurrences == len(r_params_all)) if r_params_all else None,
        "patch_embed_trainable": bool(patch_embed_params),
        "patch_embed_param_count": patch_embed_param_count,
        "patch_embed_optimizer_lr": patch_embed_optimizer_lr,
        "patch_embed_optimizer_weight_decay": patch_embed_optimizer_weight_decay,
        "patch_embed_optimizer_occurrences": patch_embed_optimizer_occurrences,
        "patch_embed_optimizer_exactly_once": (
            (patch_embed_optimizer_occurrences == len(patch_embed_params)) if patch_embed_params else None
        ),
        "lora_optimizer_lr": adapter_lr if lora_params else None,
        "lora_param_count": actual_lora_tensor_param_count,
        "actual_lora_tensor_param_count": actual_lora_tensor_param_count,
        "effective_active_lora_param_count": effective_lora_param_count,
        "backbone_optimizer_lr": backbone_lr if backbone_params else None,
        "head_optimizer_lr": head_lr,
        "max_grad_norm": args.max_grad_norm_override if args.max_grad_norm_override is not None else cfg.get("max_grad_norm", 1.0),
        "grad_clip_mode": args.grad_clip_mode,
        "force_residual_zero": args.force_residual_zero,
        "freeze_residual": args.freeze_residual,
        "consume_residual_rng": args.consume_residual_rng,
        "train_shuffle_seed": (bre.train_shuffle_seed if use_bre and bre is not None else None),
        "eval_shuffle_seed":  (bre.eval_shuffle_seed  if use_bre and bre is not None else None),
        **lora_init_fingerprint,
        **residual_fingerprint,
    }
    logger.info("Diagnostic fingerprints: %s", diagnostic_fingerprints)

    if args.eval_only:
        logger.info("Running eval-only path")
        normal_val = evaluate_full(model, val_loader, device, criterion=criterion)
        normal_train = evaluate_full(model, train_loader, device, criterion=None)
        result = {
            "run_id": run_id,
            "dataset": dataset,
            "backbone": backbone_name,
            "method": getattr(args, "public_method", args.method),
            "input_adapter": input_adapter_name,
            "seed": seed,
            "eval_only": True,
            "normal_val": normal_val,
            "normal_train": normal_train,
            "normal_train_miou": normal_train["miou"],
            cfg.get("eval_metric", "miou"): normal_val["miou"],
            "log_path": str(log_path),
            **diagnostic_fingerprints,
            **schedule_info,
        }
        if checkpoint_load_info is not None:
            result["loaded_checkpoint"] = checkpoint_load_info
        if args.eval_delta_off:
            if not (use_bre and bre is not None):
                raise ValueError("--eval-delta-off requires BRE")
            old_force = bre.force_delta_zero
            bre.force_delta_zero = True
            delta_off_val = evaluate_full(model, val_loader, device, criterion=criterion)
            delta_off_train = evaluate_full(model, train_loader, device, criterion=None)
            bre.force_delta_zero = old_force
            result["delta_off_val"] = delta_off_val
            result["delta_off_train_miou"] = delta_off_train["miou"]
        out_dir = RESULTS_DIR / "finetune"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info("Eval-only result saved -> %s", out_path)
        return

    loaded_step0_val_stats = None
    if checkpoint_load_info is not None:
        loaded_step0_val_stats = evaluate_full(model, val_loader, device, criterion=criterion)
        logger.info(
            "Loaded-checkpoint step-0 val: mIoU=%.4f pos=%.4f loss=%.4f",
            loaded_step0_val_stats["miou"],
            loaded_step0_val_stats["pos_pred_ratio"],
            loaded_step0_val_stats["loss"],
        )

    if args.use_wandb or cfg.get("use_wandb", False):
        import wandb
        wandb.init(project=cfg.get("wandb_project", "SPECTRA"),
                   config={**cfg, **schedule_info, "seed": seed, "method": args.method})

    best_metric = 0.0
    n_steps_total = 0
    max_grad_norm = args.max_grad_norm_override if args.max_grad_norm_override is not None else cfg.get("max_grad_norm", 1.0)
    logger.info("Gradient clipping: mode=%s max_norm=%s", args.grad_clip_mode, max_grad_norm)
    epoch_trajectory: list[dict] = [] 
    checkpoint_dir = RESULTS_DIR / "checkpoints"
    best_checkpoint_path = checkpoint_dir / f"{run_id}_best.pt"
    final_checkpoint_path = checkpoint_dir / f"{run_id}_final.pt"
    for epoch in range(cfg.get("epochs", 10)):
        residual_stage_state = configure_residual_stage(
            epoch,
            bre if use_bre else None,
            optimizer,
            force_residual_zero=args.force_residual_zero,
            stage1_epochs=args.stage1_bandsel_epochs,
            stage2_lr_residual=args.stage2_lr_residual,
            default_residual_lr=residual_lr,
            ramp_epochs=args.residual_gamma_ramp_epochs,
        )
        if residual_stage_state and (
            epoch == 0
            or epoch == args.stage1_bandsel_epochs
            or (
                args.residual_gamma_ramp_epochs > 0
                and residual_stage_state.get("residual_scale") == 1.0
            )
        ):
            logger.info("Residual stage epoch %d: %s", epoch + 1, residual_stage_state)
        loss, n_steps_epoch = train_epoch(model, train_loader, optimizer, criterion, device,
                                          max_grad_norm=max_grad_norm,
                                          grad_clip_mode=args.grad_clip_mode)
        dwa_epoch_summary = (
            criterion.latest_epoch_summary()
            if hasattr(criterion, "latest_epoch_summary")
            else None
        )
        n_steps_total += n_steps_epoch

        # ---- Full val pass: mIoU, per-class IoU, positive-pred ratio, val loss ----
        val_stats = evaluate_full(model, val_loader, device, criterion=criterion)
        metric = val_stats["miou"]
        if metric > best_metric:
            best_metric = metric
            if args.save_checkpoints:
                save_adapted_checkpoint(
                    best_checkpoint_path,
                    model,
                    {
                        "run_id": run_id,
                        "epoch": epoch + 1,
                        "val_miou": metric,
                        "kind": "best",
                        "fingerprints": diagnostic_fingerprints,
                        "schedule": schedule_info,
                    },
                )
                logger.info("Saved best adapted checkpoint -> %s", best_checkpoint_path)

        # ---- Train-set eval pass (cheap on small datasets) ----
        train_stats = evaluate_full(model, train_loader, device, criterion=None)
        train_miou_epoch = train_stats["miou"]

        # ---- Residual telemetry (only when a virtual residual is wired) ----
        residual_epoch_stats = None
        if use_bre and bre is not None:
            try:
                _x_sample, _ = next(iter(val_loader))
                _x_sample = _x_sample.to(device)
                residual_epoch_stats = bre.measure_residual(_x_sample)
            except StopIteration:
                pass

        # ---- Build per-epoch record ----
        rec = {
            "epoch":            epoch + 1,
            "train_loss":       float(loss),
            "val_loss":         float(val_stats["loss"]) if not (val_stats["loss"] != val_stats["loss"]) else None,
            "val_miou":         float(metric),
            "train_miou":       float(train_miou_epoch),
            "val_per_class_iou":[float(x) for x in val_stats["per_class_iou"]],
            "val_pos_pred_ratio": float(val_stats["pos_pred_ratio"]),
            "val_pred_dist":    [float(x) for x in val_stats["pred_dist"]],
            "val_pred_counts":   [int(x) for x in val_stats["pred_counts"]],
            "val_label_counts":  [int(x) for x in val_stats["label_counts"]],
            "val_pred_class_ratios": [float(x) for x in val_stats["pred_class_ratios"]],
            "val_label_class_ratios": [float(x) for x in val_stats["label_class_ratios"]],
            "val_class_area_bias": [float(x) for x in val_stats["class_area_bias"]],
            "train_pred_counts": [int(x) for x in train_stats["pred_counts"]],
            "train_label_counts": [int(x) for x in train_stats["label_counts"]],
            "train_pred_class_ratios": [float(x) for x in train_stats["pred_class_ratios"]],
            "train_label_class_ratios": [float(x) for x in train_stats["label_class_ratios"]],
            "train_class_area_bias": [float(x) for x in train_stats["class_area_bias"]],
        }
        if residual_stage_state:
            rec.update(residual_stage_state)
        if dwa_epoch_summary is not None:
            rec["train_ce_loss"] = float(dwa_epoch_summary["ce_loss"])
            rec["train_dice_loss"] = float(dwa_epoch_summary["dice_loss"])
            rec["dwa_ce_weight"] = float(dwa_epoch_summary["dwa_ce_weight"])
            rec["dwa_dice_weight"] = float(dwa_epoch_summary["dwa_dice_weight"])
            rec["dwa_temperature"] = float(dwa_epoch_summary["dwa_temperature"])
        if residual_epoch_stats is not None:
            rec["delta_to_xsel_ratio"] = float(residual_epoch_stats["delta_to_xsel_ratio"])
            rec["token_shift_ratio"]   = float(residual_epoch_stats["token_shift_ratio"])
            rec["R_final_layer_l2"]    = float(residual_epoch_stats["R_final_layer_l2"])
            if "router_entropy_mean" in residual_epoch_stats:
                rec["router_entropy_mean"] = float(residual_epoch_stats["router_entropy_mean"])
        epoch_trajectory.append(rec)

        extra = ""
        if False and None is not None:
            gate_l2 = None.gate_l2
            gate_mean_abs = None.gate_mean_abs
            gate_trajectory.append({"epoch": epoch + 1,
                                    "gate_l2": gate_l2,
                                    "gate_mean_abs": gate_mean_abs})
            extra = f"  gate_l2={gate_l2:.4f}  gate_mean|·|={gate_mean_abs:.4f}"
        if residual_epoch_stats is not None:
            extra += f"  gamma={rec.get('residual_scale', 1.0):.3f}  rLR={rec.get('residual_lr_active', residual_lr)}  d/x={rec['delta_to_xsel_ratio']:.3f}  tok={rec['token_shift_ratio']:.3f}  Rl2={rec['R_final_layer_l2']:.3f}"
        if dwa_epoch_summary is not None:
            extra += f"  wCE={rec['dwa_ce_weight']:.3f}  wDice={rec['dwa_dice_weight']:.3f}  ceL={rec['train_ce_loss']:.4f}  diceL={rec['train_dice_loss']:.4f}"
        logger.info("Epoch %d/%d  trL=%.4f  vL=%.4f  v_mIoU=%.4f  tr_mIoU=%.4f  pos=%.3f%s",
                    epoch + 1, cfg["epochs"], loss,
                    rec["val_loss"] if rec["val_loss"] is not None else float('nan'),
                    metric, train_miou_epoch, rec["val_pos_pred_ratio"], extra)

    # Final eval on the training set (so we can tell collapse-on-train apart from
    # overfit-but-good-train). Uses the same train_loader (with augmentation enabled,
    # so values are a slight under-estimate vs. an un-augmented eval, but adequate).
    final_train_stats = evaluate_full(model, train_loader, device, criterion=None)
    final_train_metric = final_train_stats["miou"]
    final_val_stats = evaluate_full(model, val_loader, device, criterion=criterion)
    logger.info("Final train %s = %.4f  (val best = %.4f)",
                cfg["eval_metric"], final_train_metric, best_metric)
    if args.save_checkpoints:
        save_adapted_checkpoint(
            final_checkpoint_path,
            model,
            {
                "run_id": run_id,
                "epoch": cfg.get("epochs", 10),
                "val_miou": final_val_stats["miou"],
                "kind": "final",
                "fingerprints": diagnostic_fingerprints,
                "schedule": schedule_info,
            },
        )
        logger.info("Saved final adapted checkpoint -> %s", final_checkpoint_path)

    gpu_h = (time.time() - t_start) / 3600

    # Save results
    result = {
        "run_id": run_id,
        "dataset": dataset,
        "backbone": backbone_name,
        "model_architecture": model_architecture,
        "method": getattr(args, "public_method", args.method),
        "input_adapter": input_adapter_name,
        "seed": seed,
        cfg.get("eval_metric", "miou"): best_metric,
        "final_val_" + cfg.get("eval_metric", "miou"): round(final_val_stats["miou"], 4),
        "final_val_pos_pred_ratio": round(final_val_stats["pos_pred_ratio"], 6),
        "final_train_" + cfg.get("eval_metric", "miou"): round(final_train_metric, 4),
        "final_val_pred_counts": final_val_stats["pred_counts"],
        "final_val_label_counts": final_val_stats["label_counts"],
        "final_val_pred_class_ratios": final_val_stats["pred_class_ratios"],
        "final_val_label_class_ratios": final_val_stats["label_class_ratios"],
        "final_val_class_area_bias": final_val_stats["class_area_bias"],
        "final_train_pred_counts": final_train_stats["pred_counts"],
        "final_train_label_counts": final_train_stats["label_counts"],
        "final_train_pred_class_ratios": final_train_stats["pred_class_ratios"],
        "final_train_label_class_ratios": final_train_stats["label_class_ratios"],
        "final_train_class_area_bias": final_train_stats["class_area_bias"],
        "gpu_h": round(gpu_h, 3),
        "log_path": str(log_path),
        "n_steps": n_steps_total,
        **diagnostic_fingerprints,
        **schedule_info,
    }
    if checkpoint_load_info is not None:
        result["loaded_checkpoint"] = checkpoint_load_info
    if loaded_step0_val_stats is not None:
        result["loaded_step0_val"] = loaded_step0_val_stats
    if args.save_checkpoints:
        result["best_checkpoint_path"] = str(best_checkpoint_path)
        result["final_checkpoint_path"] = str(final_checkpoint_path)
    if False and None is not None:
        result["dual_final_gate_l2"]       = None.gate_l2
        result["dual_final_gate_mean_abs"] = None.gate_mean_abs
        result["dual_gate_trajectory"]     = gate_trajectory
    # Always attach the per-epoch trajectory (loss/mIoU/per-class IoU/pos-pred ratio, plus
    # residual diagnostics when applicable). Lets callers reconstruct full training curves.
    result["epoch_trajectory"] = epoch_trajectory
    if isinstance(criterion, DynamicWeightAverageLoss):
        result["dwa_history"] = criterion.history
    if use_bre and bre is not None:
        result["bre_mode"]          = bre_mode
        result["bre_final_layer_l2"]     = bre.delta_l2
        if hasattr(bre, "contribution_matrix_full"):
            result["router_contribution_matrix"] = bre.contribution_matrix_full()
            result["router_top3"] = bre.top_contributions(k=3)
        if val_metric_zero is not None:
            result["val_miou_extra_zero"]    = round(val_metric_zero, 4)
        if val_metric_shuffle is not None:
            result["val_miou_extra_shuffle"] = round(val_metric_shuffle, 4)
        if per_band_corruption:
            result["per_band_corruption"]    = {
                mode: {int(k): round(v, 4) for k, v in d.items()}
                for mode, d in per_band_corruption.items()
            }
        if residual_measurements is not None:
            # Round floats for JSON readability
            rm = residual_measurements
            result["residual_measurements"] = {
                "delta_l2":             round(rm["delta_l2"], 6),
                "x_sel_l2":             round(rm["x_sel_l2"], 6),
                "delta_to_xsel_ratio":  round(rm["delta_to_xsel_ratio"], 6),
                "token_shift_ratio":    round(rm["token_shift_ratio"], 6),
                "per_out_band_delta_l2":[round(v, 6) for v in rm["per_out_band_delta_l2"]],
                "per_in_band_R0_w_l2":  [round(v, 6) for v in rm["per_in_band_R0_w_l2"]],
                "R_final_layer_l2":     round(rm["R_final_layer_l2"], 6),
            }
            if "router_entropy_mean" in rm:
                result["residual_measurements"]["router_entropy_mean"] = round(rm["router_entropy_mean"], 6)
                result["residual_measurements"]["router_entropy_per_target"] = [
                    round(v, 6) for v in rm.get("router_entropy_per_target", [])
                ]
                result["residual_measurements"]["router_top3"] = rm.get("router_top3")
            for key in ("router_gate_mean", "router_gate_min", "router_gate_max", "gated_input_l2"):
                if key in rm:
                    result["residual_measurements"][key] = round(rm[key], 6)
            if "router_top3" in rm and "router_top3" not in result["residual_measurements"]:
                result["residual_measurements"]["router_top3"] = rm.get("router_top3")
    if args.auto_test:
        if not args.save_checkpoints:
            raise RuntimeError("Internal error: auto-test requires saved best/final checkpoints")
        result["auto_test"] = {
            "best": run_post_training_test_eval(
                model=model,
                checkpoint_path=best_checkpoint_path,
                checkpoint_kind="best",
                run_id=run_id,
                cfg=cfg,
                args=args,
                device=device,
                loader_seed=loader_seed,
                split_seed=split_seed,
                n_classes=n_classes,
            ),
            "final": run_post_training_test_eval(
                model=model,
                checkpoint_path=final_checkpoint_path,
                checkpoint_kind="final",
                run_id=run_id,
                cfg=cfg,
                args=args,
                device=device,
                loader_seed=loader_seed,
                split_seed=split_seed,
                n_classes=n_classes,
            ),
        }
    out_dir = RESULTS_DIR / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Result saved → %s", out_path)
    logger.info("Best %s = %.4f  GPU-h = %.3f", cfg["eval_metric"], best_metric, gpu_h)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error during finetune run")
        raise
