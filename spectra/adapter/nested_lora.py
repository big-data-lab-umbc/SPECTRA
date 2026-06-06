"""Nested / Slimmable LoRA adapter for ViT backbones.

Training: each minibatch samples rank k ~ Uniform({4,8,16}), rank-0 with prob 1/4.
Eval:     truncate A_s[:k,:], B_s[:,:k] to the MGAS-selected rank.

The adapter wraps nn.Linear layers inside the backbone. A NestedLoRABackbone
wraps all 4 stages and exposes per-stage rank control.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


RANK_CHOICES = (0, 4, 8, 16)
MAX_RANK = 16


# ---------------------------------------------------------------------------
# Single-layer nested LoRA
# ---------------------------------------------------------------------------

class NestedLoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a nested LoRA adapter.

    Forward at train time: sample rank k, apply W_0 + B[:,:k] @ A[:k,:].
    Forward at eval time:  use self.eval_rank set by the planner.

    fixed_train_rank: when set (not None), skip stochastic sampling and use this
    rank during training. Use for standard LoRA baselines (lora8, etc.) so all
    layers apply the same rank per forward pass, giving stable head gradients.
    """

    def __init__(self, linear: nn.Linear, max_rank: int = MAX_RANK, alpha: float = 1.0) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.weight = linear.weight   # kept frozen
        self.bias   = linear.bias

        dev = linear.weight.device
        self.A = nn.Parameter(torch.empty(max_rank, d_in, device=dev))
        self.B = nn.Parameter(torch.zeros(d_out, max_rank, device=dev))
        self.scaling = alpha / max_rank
        self.eval_rank: int = max_rank  # set by MGASPlanner / schedule
        self.fixed_train_rank: Optional[int] = None  # None → stochastic (SPECTRA); int → fixed (baselines)

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def _effective_rank(self) -> int:
        if not self.training:
            return self.eval_rank
        # During training: sample random rank, but respect eval_rank=0 (LP — no LoRA)
        if self.eval_rank == 0:
            return 0
        if self.fixed_train_rank is not None:
            return self.fixed_train_rank
        return _sample_rank()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self._effective_rank()
        out = F.linear(x, self.weight, self.bias)
        if k > 0:
            lora_out = F.linear(x, self.A[:k, :])          # (*, k)
            lora_out = F.linear(lora_out, self.B[:, :k])   # (*, d_out)
            out = out + lora_out * self.scaling
        return out


RANK_GRID: list[int] = [0, 4, 8, 16]


def set_rank_grid(grid) -> None:
    """Override the discrete rank set used for stochastic rank sampling during SPECTRA warmup."""
    global RANK_GRID
    RANK_GRID = [int(r) for r in grid]


def _sample_rank() -> int:
    p = torch.rand(1).item()
    n = len(RANK_GRID)
    idx = min(int(p * n), n - 1)
    return RANK_GRID[idx]


# ---------------------------------------------------------------------------
# Stage-level LoRA wrapping
# ---------------------------------------------------------------------------

class NestedLoRAStage(nn.Module):
    """Wraps all attention + MLP Linear layers in one ViT stage with nested LoRA."""

    def __init__(self, blocks: nn.ModuleList, max_rank: int = MAX_RANK, alpha: float = 1.0) -> None:
        super().__init__()
        self.blocks = blocks
        self.lora_layers: list[NestedLoRALinear] = []
        self._wrap_blocks(max_rank, alpha)

    def _wrap_blocks(self, max_rank: int, alpha: float = 1.0) -> None:
        for name, module in self.blocks.named_modules():
            if isinstance(module, nn.Linear) and module.weight.requires_grad is False:
                parent, attr = _get_parent(self.blocks, name)
                lora = NestedLoRALinear(module, max_rank=max_rank, alpha=alpha)
                setattr(parent, attr, lora)
                self.lora_layers.append(lora)

    def set_eval_rank(self, rank: int) -> None:
        for layer in self.lora_layers:
            layer.eval_rank = rank
            # When rank=0 (LP), disable A/B gradients — no LoRA at all
            lora_on = rank > 0
            layer.A.requires_grad_(lora_on)
            layer.B.requires_grad_(lora_on)

    def set_fixed_train_rank(self, rank: Optional[int]) -> None:
        """Set fixed training rank (None = stochastic sampling for SPECTRA, int = fixed for baselines)."""
        for layer in self.lora_layers:
            layer.fixed_train_rank = rank

    def set_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze ALL parameters in the stage (weights, biases, LayerNorm, etc.).
        LoRA A/B trainability is controlled by eval_rank: active when eval_rank > 0.
        """
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad_(not frozen)
        # Re-apply LoRA trainability based on eval_rank
        for layer in self.lora_layers:
            lora_on = layer.eval_rank > 0
            layer.A.requires_grad_(lora_on)
            layer.B.requires_grad_(lora_on)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _get_parent(root: nn.Module, dotted_name: str):
    parts = dotted_name.rsplit(".", 1)
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for p in parts[0].split("."):
        parent = getattr(parent, p)
    return parent, parts[1]


# ---------------------------------------------------------------------------
# Full backbone adapter
# ---------------------------------------------------------------------------

class NestedLoRABackbone(nn.Module):
    """Wraps a ViT backbone's transformer blocks into 4 stages with nested LoRA.

    Stage partitioning: blocks [0..n//4-1], [n//4..n//2-1], ... for n total blocks.

    Args:
        backbone:      The ViT encoder (e.g. prithvi_eo_v2_600's backbone)
        blocks_attr:   Attribute name that holds the nn.ModuleList of transformer blocks
        n_stages:      Number of stages to divide blocks into (default 4)
        max_rank:      Maximum LoRA rank (default 16)
        freeze_all:    Start with all backbone weights frozen (lora params always trainable)
    """

    def __init__(
        self,
        backbone: nn.Module,
        blocks_attr: str = "blocks",
        n_stages: int = 4,
        max_rank: int = MAX_RANK,
        freeze_all: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.blocks_attr = blocks_attr
        self.n_stages = n_stages

        if freeze_all:
            for p in backbone.parameters():
                p.requires_grad_(False)

        blocks: nn.ModuleList = getattr(backbone, blocks_attr)
        n = len(blocks)
        stage_size = n // n_stages
        self.stages: nn.ModuleList = nn.ModuleList()
        for s in range(n_stages):
            start = s * stage_size
            end = (s + 1) * stage_size if s < n_stages - 1 else n
            stage_blocks = nn.ModuleList(list(blocks)[start:end])
            self.stages.append(NestedLoRAStage(stage_blocks, max_rank=max_rank, alpha=alpha))

        # Replace backbone's block list with our staged version
        merged = nn.ModuleList([b for stage in self.stages for b in stage.blocks])
        setattr(backbone, blocks_attr, merged)

    def apply_schedule(self, ranks: list[int], unfrozen: list[bool],
                       fix_train_rank: bool = False) -> None:
        """Apply MGAS-selected (r_s, u_s) schedule to all stages.

        Args:
            ranks:          list of per-stage LoRA rank [r_0, r_1, r_2, r_3]
            unfrozen:       list of per-stage unfreeze flags [u_0, u_1, u_2, u_3]
            fix_train_rank: when True, disable stochastic rank sampling during training
                            and use each stage's eval_rank as fixed train rank. Use
                            for standard LoRA baselines (lora8, etc.).
        """
        assert len(ranks) == self.n_stages and len(unfrozen) == self.n_stages
        for s, (stage, r, u) in enumerate(zip(self.stages, ranks, unfrozen)):
            stage.set_eval_rank(r)
            stage.set_frozen(not u)
            stage.set_fixed_train_rank(r if fix_train_rank else None)

    def set_fixed_train_rank(self, rank: Optional[int]) -> None:
        """Propagate fixed training rank to all stages/layers."""
        for stage in self.stages:
            stage.set_fixed_train_rank(rank)

    def lora_adapter_params(self) -> list[nn.Parameter]:
        """Return only LoRA A/B parameters (requires_grad=True)."""
        lora_ids: set[int] = set()
        result: list[nn.Parameter] = []
        for stage in self.stages:
            for layer in stage.lora_layers:
                for p in (layer.A, layer.B):
                    if p.requires_grad and id(p) not in lora_ids:
                        lora_ids.add(id(p))
                        result.append(p)
        return result

    def unfrozen_backbone_params(self) -> list[nn.Parameter]:
        """Return backbone (non-LoRA) parameters that are unfrozen (requires_grad=True)."""
        lora_ids: set[int] = set()
        for stage in self.stages:
            for layer in stage.lora_layers:
                lora_ids.add(id(layer.A))
                lora_ids.add(id(layer.B))
        result: list[nn.Parameter] = []
        seen: set[int] = set()
        for p in self.parameters():
            if p.requires_grad and id(p) not in lora_ids and id(p) not in seen:
                seen.add(id(p))
                result.append(p)
        return result

    def trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def forward_features_per_stage(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract CLS+patch tokens after each stage for LogME profiling.

        Uses the backbone's own forward_features to handle backbone-specific
        pre-processing (e.g. Prithvi's 3D patch embed, temporal encoding).
        Relies on forward_features returning a list of one tensor per block.
        """
        with torch.no_grad():
            all_block_outputs = self.backbone.forward_features(x)
        # all_block_outputs[i]: (B, 1+N, D) after block i
        n = len(all_block_outputs)
        stage_size = n // self.n_stages
        stage_outputs = []
        for s in range(self.n_stages):
            end_idx = (s + 1) * stage_size - 1 if s < self.n_stages - 1 else n - 1
            stage_outputs.append(all_block_outputs[end_idx].detach())
        return stage_outputs
