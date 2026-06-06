"""Parameter-free band-selection adapter.

Picks a fixed subset of input channels (one per pre-training band, by closest
central wavelength) so the pretrained patch_embed can process the result
without any learnable front-end. The selection indices are stored in a buffer
and the module has zero trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BandSelector(nn.Module):
    def __init__(self, indices: list[int]) -> None:
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.in_chans = max(indices) + 1   # informational
        self.out_chans = len(indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W) → (B, len(indices), H, W)
        return x.index_select(1, self.indices)
