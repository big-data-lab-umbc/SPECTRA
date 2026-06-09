#!/usr/bin/env python3
"""Evaluate a saved fine-tuning checkpoint on a held-out split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import (  # noqa: E402
    BACKBONE_PRETRAIN_SENSOR,
    METHODS,
    PUBLIC_METHOD_ALIASES,
    apply_baseline_schedule,
    head_weight_norm,
    load_adapted_checkpoint,
    load_backbone,
    load_config,
    reset_task_head,
    resolve_seed,
    residual_init_norms,
    set_all_seeds,
    setup_file_logging,
    wire_band_selector,
)
from spectra.adapter.nested_lora import NestedLoRABackbone, set_rank_grid  # noqa: E402
from spectra.data.config import BACKBONE_SPECS, FULL_BAND_CONFIGS, RESULTS_DIR  # noqa: E402
from spectra.data.datasets import build_dataloader  # noqa: E402
from spectra.data.srf import get_srf, get_srf_triples, select_closest_bands  # noqa: E402
from spectra.evaluation.segmentation_eval import (  # noqa: E402
    evaluate_segmentation,
    save_segmentation_visualizations,
)
from spectra.tokenizer.band_selector import BandSelector  # noqa: E402
from spectra.tokenizer.bre import BRE  # noqa: E402


logger = logging.getLogger("evaluate_checkpoint")


DATASET_N_CLASSES = {
    "fire_scars": 2,
    "sen1floods11": 2,
    "abi_cloud": 4,
    "multitemporal_crop": 14,
    "cloudsen12": 4,
    "bigearthnet": 19,
    "loveda": 7,
    "landslide4sense": 2,
    "geobench_sa_crop_type": 10,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", default="spectra", choices=list(PUBLIC_METHOD_ALIASES))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--model-seed", "--model-init-seed", dest="model_seed", type=int, default=None)
    p.add_argument("--residual-seed", "--residual-init-seed", dest="residual_seed", type=int, default=None)
    p.add_argument("--lora-seed", "--lora-init-seed", dest="lora_seed", type=int, default=None)
    p.add_argument("--head-seed", "--head-init-seed", dest="head_seed", type=int, default=None)
    p.add_argument("--loader-seed", type=int, default=None)
    p.add_argument("--train-shuffle-seed", type=int, default=None)
    p.add_argument("--eval-shuffle-seed", type=int, default=None)
    p.add_argument("--max-rank", type=int, default=None)
    p.add_argument("--rank-grid", type=str, default=None)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--ranks", type=str, default=None,
                   help="Comma-separated per-stage ranks. Defaults to checkpoint meta or method schedule.")
    p.add_argument("--unfrozen", type=str, default=None,
                   help="Comma-separated per-stage unfreeze flags, e.g. F,F,F,F.")
    p.add_argument("--force-residual-zero", action="store_true",
                   help="For residual checkpoints, force delta_6=0 during evaluation.")
    p.add_argument("--foreground-class", type=int, default=1)
    p.add_argument("--ignore-index", type=int, default=-1)
    p.add_argument("--vis-examples", type=int, default=0,
                   help="Save this many RGB | GT overlay | prediction overlay examples.")
    p.add_argument("--vis-alpha", type=float, default=0.45)
    p.add_argument("--vis-dir", type=Path, default=RESULTS_DIR / "visualizations")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "test_eval")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()
    args.public_method = args.method
    args.method = PUBLIC_METHOD_ALIASES[args.public_method]
    return args


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_bool_list(value: str, n: int) -> list[bool]:
    if value is None:
        return [False] * n
    result = []
    for item in value.split(","):
        token = item.strip().upper()
        if token in {"T", "TRUE", "1", "Y", "YES"}:
            result.append(True)
        elif token in {"F", "FALSE", "0", "N", "NO"}:
            result.append(False)
        else:
            raise ValueError(f"Invalid unfreeze flag: {item}")
    return result


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
        return payload["meta"]
    return {}


def _method_flags(method: str, cfg: dict) -> dict[str, Any]:
    if method.endswith("_bandsel"):
        return {"use_band_select": True, "use_bre": False, "sched_method": method[:-len("_bandsel")]}
    if method.endswith("_bre"):
        return {"use_band_select": False, "use_bre": True, "sched_method": method[:-len("_bre")]}
    return {"use_band_select": False, "use_bre": False, "sched_method": method}


def _schedule_model(
    args: argparse.Namespace,
    cfg: dict,
    lora_backbone: NestedLoRABackbone,
    sched_method: str,
    checkpoint_meta: dict[str, Any],
) -> dict[str, Any]:
    n_stages = cfg["lora"]["n_stages"]
    if args.ranks is not None:
        ranks = _parse_int_list(args.ranks)
        unfrozen = _parse_bool_list(args.unfrozen, n_stages)
        if len(ranks) != n_stages or len(unfrozen) != n_stages:
            raise ValueError(f"--ranks/--unfrozen must have {n_stages} entries")
        lora_backbone.apply_schedule(ranks, unfrozen, fix_train_rank=True)
        return {"method": getattr(args, "public_method", args.method), "ranks": ranks, "unfrozen": unfrozen, "plan_source": "cli"}

    meta_schedule = checkpoint_meta.get("schedule", {})
    if isinstance(meta_schedule, dict) and "ranks" in meta_schedule and "unfrozen" in meta_schedule:
        ranks = [int(x) for x in meta_schedule["ranks"]]
        unfrozen = [bool(x) for x in meta_schedule["unfrozen"]]
        lora_backbone.apply_schedule(ranks, unfrozen, fix_train_rank=True)
        return {**meta_schedule, "plan_source": "checkpoint_meta"}

    if sched_method in METHODS and METHODS.get(sched_method) is not None:
        sched = apply_baseline_schedule(sched_method, lora_backbone, cfg)
        return {"method": getattr(args, "public_method", args.method), "ranks": sched.ranks, "unfrozen": list(sched.unfrozen)}

    raise ValueError(
        f"Cannot infer schedule for method={args.method}. Pass --ranks and --unfrozen, "
        "or evaluate a checkpoint that includes schedule metadata."
    )


def build_model_for_eval(
    args: argparse.Namespace,
    cfg: dict,
    device: torch.device,
    checkpoint_meta: dict[str, Any],
    seed_protocol: dict[str, int],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    dataset = cfg["dataset"]
    backbone_name = cfg["backbone"]
    n_classes = DATASET_N_CLASSES.get(dataset, 2)
    spec = BACKBONE_SPECS[backbone_name]
    band_cfg = FULL_BAND_CONFIGS[dataset]
    flags = _method_flags(args.method, cfg)

    set_all_seeds(seed_protocol["model_seed"])
    model = load_backbone(backbone_name, n_classes, device)

    srf_triples = torch.from_numpy(get_srf_triples(band_cfg.sensor_key)).to(device)
    bandsel_indices = None
    selected_idx = None
    extra_idx = None

    if flags["use_band_select"] or flags["use_bre"]:
        pretrain_sensor = BACKBONE_PRETRAIN_SENSOR.get(backbone_name, "fire_scars")
        pretrain_bands = get_srf(pretrain_sensor)
        ref_wavelengths = [b.center_nm for b in pretrain_bands]
        bandsel_indices = select_closest_bands(band_cfg.sensor_key, ref_wavelengths)
        selected_idx = list(bandsel_indices)
        extra_idx = [i for i in range(band_cfg.in_chans) if i not in bandsel_indices]

    band_selector = None
    if flags["use_band_select"]:
        if bandsel_indices is None:
            raise RuntimeError("bandsel_indices not initialized")
        band_selector = BandSelector(bandsel_indices).to(device)


    if flags["use_band_select"]:
        wire_band_selector(model, band_selector)

    lora_cfg = cfg.get("lora", {})
    if args.max_rank is not None:
        lora_cfg["max_rank"] = args.max_rank
    effective_max_rank = int(lora_cfg.get("max_rank", 16))
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else effective_max_rank / 16.0
    if args.rank_grid is not None:
        set_rank_grid(_parse_int_list(args.rank_grid))

    set_all_seeds(seed_protocol["lora_seed"])
    lora_backbone = NestedLoRABackbone(
        model.encoder,
        blocks_attr="blocks",
        n_stages=lora_cfg.get("n_stages", 4),
        max_rank=effective_max_rank,
        freeze_all=True,
        alpha=lora_alpha,
    ).to(device)

    reset_task_head(model, seed_protocol["head_seed"])
    schedule_info = _schedule_model(
        args,
        cfg,
        lora_backbone,
        flags["sched_method"],
        checkpoint_meta,
    )

    build_info = {
        "input_adapter": (
            "bandsel" if flags["use_band_select"]
            else "dual" if False
            else "bre" if flags["use_bre"]
            else "native"
        ),
        "schedule": schedule_info,
        "selected_idx": selected_idx,
        "extra_idx": extra_idx,
        "head_weight_norm_before_load": head_weight_norm(model),
        "first_lora_A_norm_before_load": None,
        "first_lora_B_norm_before_load": None,
        **residual_init_norms(bre),
    }
    if bre is not None and hasattr(bre, "contribution_matrix_full"):
        build_info["router_candidate_idx"] = [int(x) for x in bre.candidate_idx.detach().cpu().tolist()]
        build_info["router_contribution_matrix_before_load"] = bre.contribution_matrix_full()
        build_info["router_top3_before_load"] = bre.top_contributions(k=3)
    for stage in lora_backbone.stages:
        if stage.lora_layers:
            build_info["first_lora_A_norm_before_load"] = float(stage.lora_layers[0].A.detach().norm().item())
            build_info["first_lora_B_norm_before_load"] = float(stage.lora_layers[0].B.detach().norm().item())
            break
    return model, build_info


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed
    seed_protocol = {
        "seed": seed,
        "split_seed": resolve_seed(args.split_seed, seed),
        "model_seed": resolve_seed(args.model_seed, seed),
        "residual_seed": resolve_seed(args.residual_seed, seed),
        "lora_seed": resolve_seed(args.lora_seed, seed),
        "head_seed": resolve_seed(args.head_seed, seed),
        "loader_seed": resolve_seed(args.loader_seed, seed),
        "train_shuffle_seed": resolve_seed(args.train_shuffle_seed, seed),
        "eval_shuffle_seed": resolve_seed(args.eval_shuffle_seed, seed),
    }
    run_id = args.run_id or f"{cfg['dataset']}_{args.method}_s{seed}_{args.split}_eval"
    log_path = setup_file_logging(run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Log file: %s", log_path)
    logger.info("Device: %s  Method: %s  Split: %s  Checkpoint: %s",
                device, args.method, args.split, args.checkpoint)
    logger.info("Seeds: %s", seed_protocol)

    checkpoint_meta = _checkpoint_meta(args.checkpoint)

    # Safety: refuse to evaluate on a split that could contain training images.
    # The checkpoint stores its training-time split_seed in
    # meta['fingerprints']['split_seed']. If the user's --seed / --split-seed
    # resolves to something different, the held-out test file would be a
    # *different* random partition of the dataset, whose test images may have
    # been in the training set of this checkpoint's split. Older checkpoints
    # without the field bypass the check so they remain usable.
    ckpt_split_seed = (
        checkpoint_meta.get("fingerprints", {}).get("split_seed")
        if isinstance(checkpoint_meta.get("fingerprints"), dict)
        else None
    )
    if ckpt_split_seed is not None and int(seed_protocol["split_seed"]) != int(ckpt_split_seed):
        raise ValueError(
            f"split_seed mismatch: this script resolves split_seed={seed_protocol['split_seed']} "
            f"(from --split-seed or the default --seed), but the checkpoint at {args.checkpoint} "
            f"was trained with split_seed={ckpt_split_seed}. Evaluating on a different "
            f"split would risk testing on images that appeared in the training set of this "
            f"checkpoint. Pass --split-seed {ckpt_split_seed} (or change --seed) and retry."
        )

    model, build_info = build_model_for_eval(args, cfg, device, checkpoint_meta, seed_protocol)
    checkpoint_load_info = load_adapted_checkpoint(model, args.checkpoint, device)
    patch_embed = model.encoder.patch_embed
    if hasattr(patch_embed, "contribution_matrix_full"):
        build_info["router_contribution_matrix_after_load"] = patch_embed.contribution_matrix_full()
        build_info["router_top3_after_load"] = patch_embed.top_contributions(k=3)

    loader = build_dataloader(
        cfg["dataset"],
        args.split,
        batch_size=cfg.get("batch_size", 8),
        num_workers=cfg.get("num_workers", 4),
        crop_size=cfg.get("crop_size", 224),
        seed=seed_protocol["loader_seed"],
        split_seed=seed_protocol["split_seed"],
        pad_multiple=cfg.get("pad_multiple", 14),
    )

    t_start = time.time()
    metrics = evaluate_segmentation(
        model,
        loader,
        device,
        foreground_class=args.foreground_class,
        ignore_index=args.ignore_index,
    )
    gpu_h = (time.time() - t_start) / 3600.0
    logger.info(
        "%s metrics: mIoU=%.4f fgIoU=%s fgDice=%s pred_pos=%s true_pos=%s",
        args.split,
        metrics["miou"],
        metrics.get("foreground_iou"),
        metrics.get("foreground_dice"),
        metrics.get("pred_positive_ratio"),
        metrics.get("true_positive_ratio"),
    )

    visualizations = None
    if args.vis_examples > 0:
        visualizations = save_segmentation_visualizations(
            model,
            loader,
            device,
            dataset_name=cfg["dataset"],
            run_id=run_id,
            out_dir=args.vis_dir,
            max_examples=args.vis_examples,
            alpha=args.vis_alpha,
            foreground_class=args.foreground_class,
            ignore_index=args.ignore_index,
        )
        logger.info("Saved %d visualization(s) under %s",
                    len(visualizations["images"]), visualizations["dir"])

    result = {
        "run_id": run_id,
        "dataset": cfg["dataset"],
        "backbone": cfg["backbone"],
        "method": getattr(args, "public_method", args.method),
        "split": args.split,
        "seed": seed,
        "seed_protocol": seed_protocol,
        "checkpoint": str(args.checkpoint),
        "checkpoint_meta": checkpoint_meta,
        "checkpoint_load": checkpoint_load_info,
        "log_path": str(log_path),
        "gpu_h": round(gpu_h, 4),
        "build_info": build_info,
        "metrics": metrics,
        f"{args.split}_miou": metrics["miou"],
        f"{args.split}_macro_iou": metrics.get("macro_iou"),
        f"{args.split}_macro_precision": metrics.get("macro_precision"),
        f"{args.split}_macro_recall": metrics.get("macro_recall"),
        f"{args.split}_macro_f1": metrics.get("macro_f1"),
        f"{args.split}_mean_class_accuracy": metrics.get("mean_class_accuracy"),
        f"{args.split}_foreground_iou": metrics.get("foreground_iou"),
        f"{args.split}_pred_positive_ratio": metrics.get("pred_positive_ratio"),
        f"{args.split}_true_positive_ratio": metrics.get("true_positive_ratio"),
        f"{args.split}_per_class_iou": metrics.get("per_class_iou"),
        f"{args.split}_per_class_precision": metrics.get("per_class_precision"),
        f"{args.split}_per_class_recall": metrics.get("per_class_recall"),
        f"{args.split}_per_class_f1": metrics.get("per_class_f1"),
        f"{args.split}_pred_counts": metrics.get("pred_counts"),
        f"{args.split}_label_counts": metrics.get("label_counts"),
        f"{args.split}_pred_class_ratios": metrics.get("pred_class_ratios"),
        f"{args.split}_label_class_ratios": metrics.get("label_class_ratios"),
        f"{args.split}_class_area_bias": metrics.get("class_area_bias"),
    }
    if visualizations is not None:
        result["visualizations"] = visualizations

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Result saved -> %s", out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error during checkpoint evaluation")
        raise
