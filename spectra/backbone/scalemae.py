"""Scale-MAE backbone wrapper for segmentation fine-tuning.

The TorchGeo Scale-MAE implementation exposes a ViT encoder only. This module
wraps that encoder with a UPerNet decoder so it matches the local training
pipeline interface:

  - model.encoder: ViT encoder with .blocks for NestedLoRA
  - model.decoder: UPerNetDecoder
  - model.seg_head: 1x1 segmentation classifier
  - model.forward(x): dense segmentation logits
"""

from __future__ import annotations

import logging
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models.scale_mae import get_2d_sincos_pos_embed_with_resolution

logger = logging.getLogger(__name__)

SCALEMAE_FEATURE_INDICES = [7, 11, 15, 23]
SCALEMAE_DECODER_CHANNELS = 256


class ScaleMAENormalizedPatchEmbed(nn.Module):
    """RGB normalization wrapper around the pretrained Scale-MAE patch embed."""

    def __init__(self, patch_embed: nn.Module) -> None:
        super().__init__()
        self.inner = patch_embed
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", imagenet_mean, persistent=False)
        self.register_buffer("rgb_std", imagenet_std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"Scale-MAE normalized patch embed expects 3 RGB channels, got {x.shape[1]}")
        mean = self.rgb_mean.to(dtype=x.dtype, device=x.device)
        std = self.rgb_std.to(dtype=x.dtype, device=x.device)
        return self.inner((x - mean) / std)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                inner = super().__getattr__("inner")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(inner, name)


class ScaleMAESegModel(nn.Module):
    """Scale-MAE ViT-L encoder plus UPerNet segmentation head."""

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        feature_indices: list[int] | None = None,
        decoder_channels: int = SCALEMAE_DECODER_CHANNELS,
    ) -> None:
        super().__init__()
        from terratorch.models.decoders.upernet_decoder import UperNetDecoder

        self.encoder = encoder
        self.feature_indices = feature_indices or list(SCALEMAE_FEATURE_INDICES)
        self.decoder = UperNetDecoder(
            embed_dim=[encoder.embed_dim] * len(self.feature_indices),
            channels=decoder_channels,
            scale_modules=True,
        )
        self.seg_head = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)

    def _tokens_to_maps(self, token_outputs: list[torch.Tensor], img_hw: tuple[int, int]) -> list[torch.Tensor]:
        h_img, w_img = img_hw
        patch_size = self.encoder.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            ph, pw = patch_size
        else:
            ph = pw = int(patch_size)
        h_patch, w_patch = h_img // ph, w_img // pw

        features: list[torch.Tensor] = []
        for idx in self.feature_indices:
            tokens = token_outputs[idx]
            patch_tokens = tokens[:, 1:, :]
            bsz, n_tokens, dim = patch_tokens.shape
            expected = h_patch * w_patch
            if n_tokens != expected:
                side = int(n_tokens**0.5)
                if side * side != n_tokens:
                    raise ValueError(
                        f"Scale-MAE token count {n_tokens} does not match image grid "
                        f"{h_patch}x{w_patch} and is not square."
                    )
                h_patch = w_patch = side
            fmap = patch_tokens.transpose(1, 2).reshape(bsz, dim, h_patch, w_patch)
            features.append(fmap.contiguous())
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_hw = (x.shape[-2], x.shape[-1])
        token_outputs = self.encoder.forward_features(x)
        features = self._tokens_to_maps(token_outputs, img_hw)
        decoded = self.decoder(features)
        logits = self.seg_head(decoded)
        return F.interpolate(logits, size=img_hw, mode="bilinear", align_corners=False)


def _patch_scalemae_forward_features(encoder: nn.Module) -> None:
    """Return one CLS+patch-token tensor after every transformer block.

    NestedLoRA's profiler expects ``encoder.forward_features(x)`` to return a
    list of block outputs, matching the Prithvi path. TorchGeo's Scale-MAE
    normally returns only the final feature tensor, so we patch it here.
    """

    def forward_features_all(self, x: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        expected_chans = getattr(self.patch_embed, "in_chans_full", None)
        if expected_chans is None:
            proj = getattr(self.patch_embed, "proj", None)
            expected_chans = getattr(proj, "in_channels", None)
        if expected_chans is not None and x.shape[1] != int(expected_chans):
            raise ValueError(
                f"Scale-MAE patch path expects {int(expected_chans)} input channels, got {x.shape[1]}"
            )
        x = self.patch_embed(x)
        if x.dim() == 4:
            if x.shape[-1] == self.embed_dim:
                grid_h, grid_w = int(x.shape[1]), int(x.shape[2])
                x = x.reshape(x.shape[0], grid_h * grid_w, self.embed_dim)
            elif x.shape[1] == self.embed_dim:
                grid_h, grid_w = int(x.shape[2]), int(x.shape[3])
                x = x.flatten(2).transpose(1, 2)
            else:
                raise ValueError(f"Unexpected Scale-MAE patch embedding shape: {tuple(x.shape)}")
        elif x.dim() == 3:
            n_tokens = int(x.shape[1])
            grid_h = grid_w = int(n_tokens**0.5)
            if grid_h * grid_w != n_tokens:
                raise ValueError(f"Scale-MAE expects a square patch grid, got {n_tokens} tokens")
        else:
            raise ValueError(f"Unexpected Scale-MAE patch embedding rank: {x.dim()}")
        if grid_h != grid_w:
            raise ValueError(f"Scale-MAE expects square inputs after padding, got patch grid {grid_h}x{grid_w}")
        res = torch.tensor(self.res, dtype=x.dtype, device=x.device).repeat(x.shape[0])
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            self.embed_dim,
            grid_h,
            res,
            cls_token=True,
        ).to(dtype=x.dtype, device=x.device)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        outputs: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x.clone())
        if outputs and hasattr(self, "norm"):
            outputs[-1] = self.norm(outputs[-1])
        return outputs

    encoder.forward_features = MethodType(forward_features_all, encoder)
    encoder.forward = encoder.forward_features


def load_scalemae(backbone_name: str, n_classes: int, device: torch.device) -> ScaleMAESegModel:
    """Load TorchGeo Scale-MAE ViT-L/16 fMoW-RGB weights."""

    if backbone_name != "scalemae_fmow_rgb":
        raise ValueError(f"Unknown Scale-MAE backbone: {backbone_name}")

    from torchgeo.models import ScaleMAELarge16_Weights, scalemae_large_patch16

    logger.info("Loading Scale-MAE ViT-L/16 fMoW-RGB from TorchGeo")
    encoder = scalemae_large_patch16(
        weights=ScaleMAELarge16_Weights.FMOW_RGB,
        img_size=224,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )
    encoder.patch_embed = ScaleMAENormalizedPatchEmbed(encoder.patch_embed)
    _patch_scalemae_forward_features(encoder)
    model = ScaleMAESegModel(encoder, n_classes).to(device)
    logger.info(
        "Scale-MAE loaded: blocks=%d embed_dim=%d patch_size=%s feature_indices=%s",
        len(encoder.blocks),
        encoder.embed_dim,
        encoder.patch_embed.patch_size,
        SCALEMAE_FEATURE_INDICES,
    )
    return model
