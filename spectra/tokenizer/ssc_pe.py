"""SSC-PE: Spectral-Spatial Conditioned Patch Embedding.

Architecture (per the SPECTRA proposal):
    e_{n,c}  = Conv_{p×p, D}(X[:, c:c+1, :, :])       # shared band-wise patch embed
    s_c      = SRFEncoder(SRF_c)                        # DeepSet or triple fallback
    U(s_c)   = B @ diag(MLP_mix(s_c)) @ A              # rank-K spectral mixer
    a_{n,c}  = σ(MLP_gate([pool(e_{n,c}), s_c]))       # spatial-spectral gate
    z_0[n]   = Σ_c  a_{n,c} · U(s_c) · e_{n,c}        # output token

Fallback chain:
    1. DeepSetSRFEncoder (preferred, uses full SRF curve)
    2. TripleSRFEncoder  (λ, FWHM, area) — used when no SRF curve available
    3. Gate disabled (a_{n,c} = 1) — warm-up fallback if stopping rule fails
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# SRF Encoders
# ---------------------------------------------------------------------------

class TripleSRFEncoder(nn.Module):
    """Encodes a band from (λ_center, FWHM, area) triple → s_c of dim srf_dim."""

    def __init__(self, srf_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, srf_dim),
        )

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            triples: (C, 3) normalized (λ, FWHM, area) triples
        Returns:
            s: (C, srf_dim) band embeddings
        """
        return self.mlp(triples)


class DeepSetSRFEncoder(nn.Module):
    """Encodes a full SRF curve (variable-length) via DeepSet → s_c of dim srf_dim.

    SRF_c is a sequence of (wavelength, response) pairs. The DeepSet encodes
    each pair independently then pools, making it permutation-invariant.
    """

    def __init__(self, srf_dim: int = 64, inner_dim: int = 128) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(inner_dim, srf_dim),
            nn.GELU(),
            nn.Linear(srf_dim, srf_dim),
        )

    def forward(self, srf_curves: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            srf_curves: (C, L, 2) wavelength-response pairs (padded)
            mask:       (C, L) boolean mask (True = valid sample)
        Returns:
            s: (C, srf_dim)
        """
        C, L, _ = srf_curves.shape
        phi_out = self.phi(srf_curves.reshape(C * L, 2)).reshape(C, L, -1)
        if mask is not None:
            phi_out = phi_out * mask.unsqueeze(-1).float()
            counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled = phi_out.sum(dim=1) / counts
        else:
            pooled = phi_out.mean(dim=1)
        return self.rho(pooled)


# ---------------------------------------------------------------------------
# Spectral Mixer
# ---------------------------------------------------------------------------

class SpectralMixer(nn.Module):
    """Rank-K spectral mixer: U(s_c) = B @ diag(MLP_mix(s_c)) @ A.

    A: (K, D), B: (D, K) — shared parameters.
    MLP_mix produces K per-band scaling factors conditioned on s_c.
    """

    def __init__(self, embed_dim: int, srf_dim: int, rank: int = 16) -> None:
        super().__init__()
        self.rank = rank
        self.embed_dim = embed_dim
        self.A = nn.Parameter(torch.empty(rank, embed_dim))
        self.B = nn.Parameter(torch.empty(embed_dim, rank))
        self.mlp_mix = nn.Sequential(
            nn.Linear(srf_dim, rank * 2),
            nn.GELU(),
            nn.Linear(rank * 2, rank),
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (C, srf_dim) band embeddings
            e: (B, C, N, D) patch embeddings per band
        Returns:
            out: (B, C, N, D) spectrally mixed embeddings
        """
        # diag_vals offset by +1 so default scale factor is 1.0 (residual)
        diag_vals = self.mlp_mix(s) + 1.0      # (C, K)
        e_proj = torch.einsum("bcnd,kd->bcnk", e, self.A)  # (B,C,N,K)
        e_scaled = e_proj * diag_vals.unsqueeze(0).unsqueeze(2)  # (B,C,N,K)
        out = torch.einsum("bcnk,dk->bcnd", e_scaled, self.B)     # (B,C,N,D)
        return out


# ---------------------------------------------------------------------------
# Spatial-Spectral Gate
# ---------------------------------------------------------------------------

class SpatialSpectralGate(nn.Module):
    """a_{n,c} = σ(MLP_gate([pool(e_{n,c}), s_c]))."""

    def __init__(self, embed_dim: int, srf_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + srf_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, e: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: (B, C, N, D) patch embeddings per band
            s: (C, srf_dim)
        Returns:
            gate: (B, C, N, 1)
        """
        B, C, N, D = e.shape
        pooled = e.mean(dim=2)                          # (B, C, D) spatial pool
        s_expand = s.unsqueeze(0).expand(B, -1, -1)    # (B, C, srf_dim)
        inp = torch.cat([pooled, s_expand], dim=-1)     # (B, C, D+srf_dim)
        gate = torch.sigmoid(self.mlp(inp))             # (B, C, 1)
        return gate.unsqueeze(2)                         # (B, C, 1, 1) broadcast over N


# ---------------------------------------------------------------------------
# SSC-PE Main Module
# ---------------------------------------------------------------------------

class SSCPE(nn.Module):
    """Spectral-Spatial Conditioned Patch Embedding.

    Replaces the standard backbone patch embedding for arbitrary-band inputs.

    Args:
        in_chans:          Number of input bands C_t
        embed_dim:         Output token dimension D (must match backbone)
        patch_size:        Spatial patch size p (e.g. 16)
        srf_dim:           Dimension of SRF embedding s_c
        mixer_rank:        Rank K of the spectral mixer U(s_c)
        use_gate:          Enable spatial-spectral gate (set False for ablation A-GATE)
        use_deepset:       Use DeepSet SRF encoder (True) or triple fallback (False)
        gate_disabled_fallback: If True, sets a_{n,c}=1/C (uniform weights, no gate)
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: int = 16,
        srf_dim: int = 64,
        mixer_rank: int = 16,
        use_gate: bool = True,
        use_deepset: bool = False,
        gate_disabled_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_gate = use_gate and not gate_disabled_fallback
        self.gate_disabled_fallback = gate_disabled_fallback

        # Shared band-wise patch conv (same weights applied to each band's single-channel patch)
        self.band_patch_conv = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # SRF encoder
        if use_deepset and not gate_disabled_fallback:
            self.srf_encoder: nn.Module = DeepSetSRFEncoder(srf_dim=srf_dim)
        else:
            self.srf_encoder = TripleSRFEncoder(srf_dim=srf_dim)

        # Spectral mixer
        self.spectral_mixer = SpectralMixer(embed_dim=embed_dim, srf_dim=srf_dim, rank=mixer_rank)

        # Gate (optional)
        if self.use_gate:
            self.gate = SpatialSpectralGate(embed_dim=embed_dim, srf_dim=srf_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # fan_in so each band's patch tokens start at reasonable scale
        nn.init.kaiming_normal_(self.band_patch_conv.weight, mode="fan_in", nonlinearity="relu")
        if self.band_patch_conv.bias is not None:
            nn.init.zeros_(self.band_patch_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        srf_input: torch.Tensor,
        srf_curves: Optional[torch.Tensor] = None,
        srf_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, C_t, H, W) multi-band image
            srf_input:  (C_t, 3) normalized (λ, FWHM, area) triples — always required
            srf_curves: (C_t, L, 2) full SRF curves for DeepSet encoder (optional)
            srf_mask:   (C_t, L) valid-sample mask for DeepSet padding
        Returns:
            tokens: (B, N, D) patch token sequence matching ViT input format
        """
        B, C, H, W = x.shape
        assert C == self.in_chans, f"Expected {self.in_chans} bands, got {C}"

        # 1. Band-wise patch embedding: apply shared conv to each band independently
        # Process all bands in one batched conv call for efficiency
        x_bands = x.reshape(B * C, 1, H, W)
        e_flat = self.band_patch_conv(x_bands)         # (B*C, D, H', W')
        H_p, W_p = e_flat.shape[2], e_flat.shape[3]
        N = H_p * W_p
        e_flat = e_flat.flatten(2).transpose(1, 2)     # (B*C, N, D)
        e = e_flat.reshape(B, C, N, self.embed_dim)    # (B, C, N, D)

        # 2. SRF encoding
        device = x.device
        srf_input = srf_input.to(device)
        if isinstance(self.srf_encoder, DeepSetSRFEncoder) and srf_curves is not None:
            s = self.srf_encoder(srf_curves.to(device), srf_mask)  # (C, srf_dim)
        else:
            s = self.srf_encoder(srf_input)                         # (C, srf_dim)

        # 3. Spectral mixing
        e_mixed = self.spectral_mixer(s, e)            # (B, C, N, D)

        # 4. Spatial-spectral gate
        if self.use_gate:
            alpha = self.gate(e, s)                    # (B, C, 1, 1)
        elif self.gate_disabled_fallback:
            alpha = torch.full((B, C, 1, 1), 1.0 / C, device=device)
        else:
            alpha = torch.ones(B, C, 1, 1, device=device) / C

        # 5. Weighted sum over bands → output tokens
        tokens = (alpha * e_mixed).sum(dim=1)          # (B, N, D)
        return tokens

    def extra_repr(self) -> str:
        return (f"in_chans={self.in_chans}, embed_dim={self.embed_dim}, "
                f"patch_size={self.patch_size}, use_gate={self.use_gate}")
