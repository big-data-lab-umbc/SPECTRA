"""Dual-path patch embedding for the diagnostic 'do extra bands help?' study.

Architecture:
    Main path: x[:, band_indices] → pretrained patch_embed (frozen) → z_main
    Aux  path: x[:, all bands]    → new Conv2d patch_embed (random init) → z_aux
    Output:    z_main + gate ⊙ z_aux

The per-channel `gate` is initialized to zero, so at step 0 the module is exactly
equivalent to `BandSelector → pretrained patch_embed`. Any improvement over the
bandsel-only baseline is therefore attributable to the model learning to use the
auxiliary path. The final gate L2-norm is logged so we can read off whether the
model wanted to use the extra bands or not.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class DualPatchEmbed(nn.Module):
    def __init__(
        self,
        original_patch_embed: nn.Module,
        band_indices: list[int],
        in_chans_full: int,
        embed_dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.original = original_patch_embed
        # Freeze the wrapped original; only aux_proj + gate train.
        for p in self.original.parameters():
            p.requires_grad_(False)
        self.register_buffer("band_indices", torch.tensor(band_indices, dtype=torch.long))
        self.aux_proj = nn.Conv2d(
            in_chans_full, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        nn.init.kaiming_uniform_(self.aux_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.aux_proj.bias)
        # Per-channel learnable gate, zero-initialised → model = bandsel at step 0.
        self.gate = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.in_chans_full = in_chans_full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x may be (B, C, H, W) (Prithvi num_frames=1 path) or (B, C, T, H, W).
        # Bandsel preserves the layout; aux always uses a 4D Conv2d input.
        x_sel = x.index_select(1, self.band_indices)
        z_main = self.original(x_sel)

        x_aux_input = x[:, :, 0] if x.dim() == 5 else x
        z_aux = self.aux_proj(x_aux_input)            # (B, D, H/p, W/p)
        z_aux = z_aux.flatten(2).transpose(1, 2)      # (B, N, D)

        if z_main.dim() == 5:
            z_main = z_main.squeeze(2).flatten(2).transpose(1, 2)
        elif z_main.dim() == 4:
            z_main = z_main.flatten(2).transpose(1, 2)
        # else assume (B, N, D) already

        return z_main + self.gate.view(1, 1, -1) * z_aux

    @property
    def gate_l2(self) -> float:
        return float(self.gate.detach().norm().item())

    @property
    def gate_mean_abs(self) -> float:
        return float(self.gate.detach().abs().mean().item())

    # Prithvi's forward_features reads attributes like `input_size`, `grid_size`,
    # `proj` directly from `self.patch_embed`. Proxy those to the wrapped original
    # so the surrounding code believes nothing changed.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)
