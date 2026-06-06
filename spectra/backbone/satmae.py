"""SatMAE multispectral backbone wrapper for dense prediction.

This wrapper targets the official multispectral SatMAE ViT-L checkpoint trained
on fMoW-Sentinel with grouped Sentinel-2 channels. It exposes the same local
interface used by the rest of the fine-tuning code:

  - model.encoder: ViT encoder with .blocks for NestedLoRA
  - model.encoder.patch_embed: a single module that accepts the native 10-band
    SatMAE input, so BandSelector / virtual residual adapters can wrap it
  - model.forward(x): dense segmentation logits

The grouped SatMAE token sequence contains one token stream per spectral group.
For UPerNet, grouped tokens from each spatial location are averaged into one
spatial feature map per selected transformer depth.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SATMAE_SOURCE_DIR = Path(os.environ.get("SATMAE_ROOT", ""))
SATMAE_CHECKPOINT = Path(os.environ.get("SATMAE_CHECKPOINT", ""))

SATMAE_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 6),  # B2, B3, B4, B8
    (3, 4, 5, 7),  # B5, B6, B7, B8A
    (8, 9),        # B11, B12
)
SATMAE_FEATURE_INDICES = [5, 11, 17, 23]
SATMAE_DECODER_CHANNELS = 256
SATMAE_NATIVE_SIZE = 96
SATMAE_PATCH_SIZE = 8
SATMAE_IN_CHANS = 10


class SatMAEGroupedPatchEmbed(nn.Module):
    """Single-module view of SatMAE's grouped patch embedding list."""

    def __init__(self, patch_embeds: nn.ModuleList, channel_groups: tuple[tuple[int, ...], ...]) -> None:
        super().__init__()
        self.group_patch_embeds = patch_embeds
        self.channel_groups = tuple(tuple(int(i) for i in group) for group in channel_groups)
        self.num_groups = len(self.channel_groups)
        self.in_chans_full = sum(len(group) for group in self.channel_groups)
        self.in_chans = self.in_chans_full
        self.embed_dim = int(self.group_patch_embeds[0].proj.out_channels)
        self.patch_size = self.group_patch_embeds[0].patch_size
        if not isinstance(self.patch_size, tuple):
            self.patch_size = (int(self.patch_size), int(self.patch_size))

        for patch_embed in self.group_patch_embeds:
            if hasattr(patch_embed, "strict_img_size"):
                patch_embed.strict_img_size = False
            if hasattr(patch_embed, "dynamic_img_pad"):
                patch_embed.dynamic_img_pad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            if x.shape[2] != 1:
                raise ValueError(f"SatMAE multispectral path expects one frame, got shape={tuple(x.shape)}")
            x = x[:, :, 0]
        if x.dim() != 4:
            raise ValueError(f"SatMAE patch embed expects a 4D BCHW tensor, got rank {x.dim()}")
        if x.shape[1] != self.in_chans_full:
            raise ValueError(
                f"SatMAE patch embed expects {self.in_chans_full} channels, got {x.shape[1]}"
            )

        group_tokens = []
        for patch_embed, group in zip(self.group_patch_embeds, self.channel_groups):
            x_group = x[:, list(group), :, :]
            group_tokens.append(patch_embed(x_group))
        return torch.stack(group_tokens, dim=1)  # (B, G, L, D)


class SatMAESegModel(nn.Module):
    """SatMAE ViT-L encoder plus UPerNet segmentation head."""

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        feature_indices: list[int] | None = None,
        decoder_channels: int = SATMAE_DECODER_CHANNELS,
    ) -> None:
        super().__init__()
        from terratorch.models.decoders.upernet_decoder import UperNetDecoder

        self.encoder = encoder
        self.feature_indices = feature_indices or list(SATMAE_FEATURE_INDICES)
        self.decoder = UperNetDecoder(
            embed_dim=[encoder.embed_dim] * len(self.feature_indices),
            channels=decoder_channels,
            scale_modules=True,
        )
        self.seg_head = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)

    def _num_groups(self) -> int:
        patch_embed = self.encoder.patch_embed
        while hasattr(patch_embed, "original"):
            patch_embed = patch_embed.original
        return int(patch_embed.num_groups)

    def _tokens_to_maps(self, token_outputs: list[torch.Tensor], img_hw: tuple[int, int]) -> list[torch.Tensor]:
        num_groups = self._num_groups()

        features: list[torch.Tensor] = []
        for idx in self.feature_indices:
            tokens = token_outputs[idx]
            patch_tokens = tokens[:, 1:, :]
            bsz, n_tokens, dim = patch_tokens.shape
            if n_tokens % num_groups != 0:
                raise ValueError(
                    f"SatMAE token count {n_tokens} is not divisible by num_groups={num_groups}"
                )
            n_spatial = n_tokens // num_groups
            side = int(n_spatial**0.5)
            if side * side != n_spatial:
                raise ValueError(f"SatMAE spatial token count {n_spatial} is not square")
            h_patch = w_patch = side

            grouped = patch_tokens.reshape(bsz, num_groups, n_spatial, dim)
            spatial_tokens = grouped.mean(dim=1)
            fmap = spatial_tokens.transpose(1, 2).reshape(bsz, dim, h_patch, w_patch)
            features.append(fmap.contiguous())
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_hw = (x.shape[-2], x.shape[-1])
        token_outputs = self.encoder.forward_features(x)
        features = self._tokens_to_maps(token_outputs, img_hw)
        decoded = self.decoder(features)
        logits = self.seg_head(decoded)
        return F.interpolate(logits, size=img_hw, mode="bilinear", align_corners=False)


def _resize_spatial_pos_embed(pos_embed: torch.Tensor, h_patch: int, w_patch: int) -> torch.Tensor:
    """Resize SatMAE's spatial positional embedding for non-96px crops."""

    bsz, n_pos, dim = pos_embed.shape
    side = int(n_pos**0.5)
    if side * side != n_pos:
        raise ValueError(f"SatMAE spatial pos_embed length {n_pos} is not square")
    if side == h_patch and side == w_patch:
        return pos_embed
    grid = pos_embed.reshape(bsz, side, side, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(h_patch, w_patch), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(bsz, h_patch * w_patch, dim)


def _patch_satmae_forward_features(encoder: nn.Module) -> None:
    """Return one CLS+grouped-patch-token tensor after every transformer block."""

    def forward_features_all(self, x: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        x = self.patch_embed(x)  # (B, G, L, D)
        bsz, num_groups, n_spatial, dim = x.shape

        ph, pw = self.patch_embed.patch_size
        h_patch = int(x.shape[2] ** 0.5)
        w_patch = h_patch
        if h_patch * w_patch != n_spatial:
            # Prefer image-derived grid for rectangular inputs.
            # The grouped patch embed preserves H-major patch order.
            raise ValueError(f"SatMAE expects a square patch grid, got {n_spatial} tokens")

        channel_embed = self.channel_embed.unsqueeze(2)
        spatial_pos = _resize_spatial_pos_embed(self.pos_embed[:, 1:, :], h_patch, w_patch)
        spatial_pos = spatial_pos.unsqueeze(1)
        channel_embed = channel_embed.expand(-1, -1, spatial_pos.shape[2], -1)
        spatial_pos = spatial_pos.expand(-1, channel_embed.shape[1], -1, -1)
        pos_channel = torch.cat((spatial_pos, channel_embed), dim=-1)

        x = x + pos_channel
        x = x.view(bsz, num_groups * n_spatial, dim)

        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)
        cls_tokens = cls_pos_channel + self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        outputs: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x.clone())
        if outputs:
            outputs[-1] = self.norm(outputs[-1])
        return outputs

    encoder.forward_features = MethodType(forward_features_all, encoder)
    encoder.forward = encoder.forward_features


def _load_satmae_source_model() -> nn.Module:
    if not SATMAE_SOURCE_DIR.exists():
        raise FileNotFoundError(f"SatMAE source directory not found: {SATMAE_SOURCE_DIR}")
    if not SATMAE_CHECKPOINT.exists():
        raise FileNotFoundError(f"SatMAE checkpoint not found: {SATMAE_CHECKPOINT}")

    # The official SatMAE repo still references np.float. Patch locally at
    # import time instead of editing the external checkout.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if str(SATMAE_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(SATMAE_SOURCE_DIR))

    from models_vit_group_channels import vit_large_patch16

    encoder = vit_large_patch16(
        img_size=SATMAE_NATIVE_SIZE,
        patch_size=SATMAE_PATCH_SIZE,
        in_chans=SATMAE_IN_CHANS,
        num_classes=0,
        channel_groups=SATMAE_GROUPS,
    )

    checkpoint = torch.load(SATMAE_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("decoder") and key != "mask_token"
    }
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info("SatMAE checkpoint load missing=%d sample=%s", len(missing), missing[:10])
    if unexpected:
        logger.info("SatMAE checkpoint load unexpected=%d sample=%s", len(unexpected), unexpected[:10])

    encoder.patch_embed = SatMAEGroupedPatchEmbed(encoder.patch_embed, SATMAE_GROUPS)
    _patch_satmae_forward_features(encoder)
    return encoder


def load_satmae(backbone_name: str, n_classes: int, device: torch.device) -> SatMAESegModel:
    """Load multispectral SatMAE ViT-L fMoW-Sentinel weights."""

    if backbone_name != "satmae_sentinel_vitl":
        raise ValueError(f"Unknown SatMAE backbone: {backbone_name}")

    logger.info("Loading SatMAE multispectral ViT-L from %s", SATMAE_CHECKPOINT)
    encoder = _load_satmae_source_model()
    model = SatMAESegModel(encoder, n_classes).to(device)
    logger.info(
        "SatMAE loaded: blocks=%d embed_dim=%d patch_size=%s feature_indices=%s groups=%s",
        len(encoder.blocks),
        encoder.embed_dim,
        encoder.patch_embed.patch_size,
        SATMAE_FEATURE_INDICES,
        SATMAE_GROUPS,
    )
    return model
