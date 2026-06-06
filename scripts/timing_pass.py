#!/usr/bin/env python3
"""R0.7 — Timing passes: measure t_s^LoRA and t_s^full per backbone per stage.

These timings feed the cost model (D.2) and STPlanner budget constraints.
Results saved to results/timing/timing_{backbone}.json.

Per-stage measurements capture the positional backward-pass cost (earlier stages
require propagating gradients through more blocks). Aggregate measurements give
actual full-schedule costs for the cost model γ fit.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n geofm4cloud python scripts/timing_pass.py \
        --backbone prithvi_eo_v2_600 --n-warmup 10 --n-measure 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _env_name in ("TERRATORCH_ROOT", "PRITHVI_EO_ROOT"):
    _env_path = os.environ.get(_env_name)
    if _env_path:
        sys.path.insert(0, _env_path)

import torch
import torch.nn as nn

from spectra.data.config import RESULTS_DIR, BACKBONE_SPECS
from spectra.adapter.nested_lora import NestedLoRABackbone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",   default="prithvi_eo_v2_600", choices=list(BACKBONE_SPECS))
    p.add_argument("--n-warmup",   type=int, default=10)
    p.add_argument("--n-measure",  type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-size",   type=int, default=224)
    p.add_argument("--out-dir",    type=Path, default=RESULTS_DIR / "timing")
    return p.parse_args()


def load_backbone(backbone_name: str, device: torch.device) -> nn.Module:
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask(
        model_args=dict(
            backbone_pretrained=True,
            backbone=backbone_name,
            backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            backbone_num_frames=1,
            decoder="FCNDecoder",
            num_classes=2,
        ),
        plot_on_val=False, lr=1e-4,
        model_factory="EncoderDecoderFactory",
    )
    return task.model.encoder.to(device)


def _make_dummy_input(backbone_name: str, batch_size: int, img_size: int, device: torch.device) -> torch.Tensor:
    n_chans = 6  # 6-band HLS, matches backbone_bands loaded above
    if "prithvi" in backbone_name.lower():
        return torch.randn(batch_size, n_chans, 1, img_size, img_size, device=device)
    return torch.randn(batch_size, n_chans, img_size, img_size, device=device)


def _forward_only(model: nn.Module, x: torch.Tensor, n_warmup: int, n_measure: int, device: torch.device) -> float:
    """Time frozen forward pass (no backward, no step)."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure):
            model(x)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / n_measure


def _time_schedule(
    model: nn.Module,
    x: torch.Tensor,
    trainable_params: list,
    n_warmup: int,
    n_measure: int,
    device: torch.device,
) -> float:
    """Measure mean seconds per forward+backward+step for the given param list."""
    opt = torch.optim.AdamW(trainable_params, lr=1e-4)
    model.train()

    for _ in range(n_warmup):
        opt.zero_grad()
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        out.float().mean().backward()
        opt.step()

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        opt.zero_grad()
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        out.float().mean().backward()
        opt.step()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / n_measure


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BACKBONE_SPECS[args.backbone]

    print(f"Backbone: {args.backbone}  n_layers={spec.n_layers}  embed_dim={spec.embed_dim}")
    print(f"Device: {device}  batch_size={args.batch_size}  img_size={args.img_size}")

    x = _make_dummy_input(args.backbone, args.batch_size, args.img_size, device)
    backbone = load_backbone(args.backbone, device)
    # Use max_rank=16 so we can measure any rank in {1,4,8,16} via apply_schedule
    lora_backbone = NestedLoRABackbone(
        backbone, blocks_attr="blocks", n_stages=spec.n_stages,
        max_rank=16, freeze_all=True,
    )

    n_stages = spec.n_stages
    results = {"backbone": args.backbone, "n_stages": n_stages, "stages": {}}

    # -------------------------------------------------------------------------
    # Aggregate measurements: actual cost of common schedule configurations.
    # These are used by the cost model γ fit (R0.8) and STPlanner budget planning.
    # The per-stage sum formula Σ_s(r*t_lora + u*t_full) overcounts because
    # PyTorch does ONE backward through all blocks when multiple stages are active.
    # Aggregate timings give the true cost of each schedule type.
    # -------------------------------------------------------------------------
    print("\n--- Aggregate measurements ---")

    # t_base: frozen backbone forward (no grad, no backward)
    lora_backbone.apply_schedule([0]*n_stages, [False]*n_stages, fix_train_rank=True)
    t_base = _forward_only(lora_backbone, x, args.n_warmup, args.n_measure, device)
    t_base_ms = round(t_base * 1000, 2)
    print(f"  t_base (frozen forward only) = {t_base_ms:.1f} ms")

    # t_all_lora_r1: all stages rank-1 LoRA, all backbone frozen
    lora_backbone.apply_schedule([1]*n_stages, [False]*n_stages, fix_train_rank=True)
    trainable_lora_all = lora_backbone.lora_adapter_params()
    t_all_lora_r1 = _time_schedule(lora_backbone, x, trainable_lora_all, args.n_warmup, args.n_measure, device)
    t_all_lora_r1_ms = round(t_all_lora_r1 * 1000, 2)
    print(f"  t_all_lora_r1 (all stages rank-1) = {t_all_lora_r1_ms:.1f} ms")

    # t_all_full: all stages unfrozen, rank=0 (no LoRA)
    lora_backbone.apply_schedule([0]*n_stages, [True]*n_stages, fix_train_rank=True)
    trainable_full_all = lora_backbone.unfrozen_backbone_params()
    t_all_full = _time_schedule(lora_backbone, x, trainable_full_all, args.n_warmup, args.n_measure, device)
    t_all_full_ms = round(t_all_full * 1000, 2)
    print(f"  t_all_full (all stages full FT)   = {t_all_full_ms:.1f} ms")

    # Derived per-stage-per-rank constants (uniform approximation)
    # These represent the MARGINAL cost of activating one stage at rank-1 LoRA
    # or full FT, as measured by the aggregate overhead divided by n_stages.
    lora_overhead_ms = t_all_lora_r1_ms - t_base_ms
    full_overhead_ms = t_all_full_ms - t_base_ms
    t_lora_per_stage_ms = round(lora_overhead_ms / n_stages, 2)
    t_full_per_stage_ms = round(full_overhead_ms / n_stages, 2)
    print(f"  Derived t_lora/stage = {t_lora_per_stage_ms:.1f} ms  (overhead {lora_overhead_ms:.1f} / {n_stages} stages)")
    print(f"  Derived t_full/stage = {t_full_per_stage_ms:.1f} ms  (overhead {full_overhead_ms:.1f} / {n_stages} stages)")

    # Rank scaling test: does LoRA cost scale linearly with rank?
    # If yes: r × t_lora_per_rank formula is valid.
    # If no (rank doesn't matter much): use binary on/off flag in cost model.
    print("\n--- Rank-scaling test (all-stage LoRA, varying rank) ---")
    rank_scale: dict[int, float] = {}
    for test_rank in [1, 4, 8, 16]:
        lora_backbone.apply_schedule([test_rank]*n_stages, [False]*n_stages, fix_train_rank=True)
        t_lora_params = lora_backbone.lora_adapter_params()
        t_r = _time_schedule(lora_backbone, x, t_lora_params, args.n_warmup, args.n_measure, device)
        t_r_ms = round(t_r * 1000, 2)
        rank_scale[test_rank] = t_r_ms
        print(f"  rank={test_rank:2d}: {t_r_ms:.1f} ms  (overhead vs base: {t_r_ms - t_base_ms:.1f} ms)")

    # Save rank-scaling ratios (relative to rank-1 overhead)
    rank1_overhead = rank_scale[1] - t_base_ms
    rank_scaling_ratios = {r: round((rank_scale[r] - t_base_ms) / max(rank1_overhead, 0.1), 3)
                           for r in rank_scale}
    print(f"  Scaling ratios (overhead_r / overhead_1): {rank_scaling_ratios}")

    results["aggregate"] = {
        "t_base_ms":           t_base_ms,
        "t_all_lora_r1_ms":   t_all_lora_r1_ms,
        "t_all_full_ms":       t_all_full_ms,
        "lora_overhead_ms":    round(lora_overhead_ms, 2),
        "full_overhead_ms":    round(full_overhead_ms, 2),
        "t_lora_per_stage_ms": t_lora_per_stage_ms,   # use in finetune.py STPlanner planner
        "t_full_per_stage_ms": t_full_per_stage_ms,
        "rank_scale_ms":       rank_scale,
        "rank_scaling_ratios": rank_scaling_ratios,
    }

    # -------------------------------------------------------------------------
    # Per-stage measurements: document positional backward-pass cost.
    # Stage 0 costs more than stage 3 because backprop must propagate through
    # more blocks to reach earlier stages. Used for non-uniform schedule analysis.
    # -------------------------------------------------------------------------
    print("\n--- Per-stage measurements (absolute, not marginal) ---")

    for s in range(n_stages):
        # Full FT: unfreeze only stage s backbone weights, rank=0
        full_ranks   = [0] * n_stages
        full_unfrozen = [s_ == s for s_ in range(n_stages)]
        lora_backbone.apply_schedule(full_ranks, full_unfrozen, fix_train_rank=True)
        trainable_full = lora_backbone.unfrozen_backbone_params()

        if not trainable_full:
            print(f"  Stage {s}: WARNING — no trainable backbone params")
            t_full_s_ms = None
        else:
            t_full_s = _time_schedule(lora_backbone, x, trainable_full, args.n_warmup, args.n_measure, device)
            t_full_s_ms = round(t_full_s * 1000, 2)

        # LoRA rank-1: only stage s, all backbone frozen
        lora_ranks = [1 if s_ == s else 0 for s_ in range(n_stages)]
        lora_backbone.apply_schedule(lora_ranks, [False]*n_stages, fix_train_rank=True)
        trainable_lora = lora_backbone.lora_adapter_params()

        if not trainable_lora:
            print(f"  Stage {s}: WARNING — no LoRA params")
            t_lora_s_ms = None
        else:
            t_lora_s = _time_schedule(lora_backbone, x, trainable_lora, args.n_warmup, args.n_measure, device)
            t_lora_s_ms = round(t_lora_s * 1000, 2)

        print(f"  Stage {s}: t_full={t_full_s_ms:.1f} ms  t_lora={t_lora_s_ms:.1f} ms  "
              f"t_full_marginal={round(t_full_s_ms - t_base_ms, 1) if t_full_s_ms else '?'} ms  "
              f"t_lora_marginal={round(t_lora_s_ms - t_base_ms, 1) if t_lora_s_ms else '?'} ms")

        results["stages"][str(s)] = {
            "t_full_ms":         t_full_s_ms,
            "t_lora_ms":         t_lora_s_ms,
            "t_full_marginal_ms": round(t_full_s_ms - t_base_ms, 2) if t_full_s_ms else None,
            "t_lora_marginal_ms": round(t_lora_s_ms - t_base_ms, 2) if t_lora_s_ms else None,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"timing_{args.backbone}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nTiming results saved to: {out_path}")


if __name__ == "__main__":
    main()
