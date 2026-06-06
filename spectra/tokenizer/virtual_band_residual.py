"""Virtual-band residual adapter.

Architecture:
    x_sel    = x[:, selected_idx]           shape (B, K, H, W)
    x_R      = mask-out version of x        shape (B, C_in, H, W)
    delta    = R(x_R)                       shape (B, K, H, W)
    x_virt   = x_sel + delta                shape (B, K, H, W)
    z        = pretrained_patch_embed(x_virt)

Where:
    R = Conv1x1(C_in, hidden) → GELU → Conv1x1(hidden, hidden) → GELU → Conv1x1(hidden, 6)
    The final Conv1x1 layer is zero-initialized so delta_6 = 0 at step 0,
    making the module exactly equivalent to bandsel-only at the first forward.

Mask modes:
    "extra"          — zero out selected channels of x before passing to R; R
                       only sees the non-selected ("extra") bands. Tests "do
                       the extra bands carry useful signal beyond the selected
                       six?" (resB)
    "selected"       — zero out extra channels of x before passing to R; R
                       only sees the selected six bands. Parameter-matched
                       control: tests "does extra trainable capacity alone
                       help, even without any new spectral information?" (resC)
    "all"            — R sees all 13 bands as-is. Tests "does modeling the
                       selected/extra spectral relationship help?" (resD)
    "shuffle_extra"  — like "all", but at TRAIN time the extra channels of
                       R's input are shuffled along the batch dim (preserves
                       marginal band statistics, destroys image-label
                       alignment for extras); deterministic permutation at
                       eval. The selected-band main path always receives the
                       real x_sel. Tests "is D's gain from real extra-band
                       information or just from extra parameters / statistics?"
                       (resE — negative control for D)

Eval-time corruption (optional, set externally after training):
    `eval_corruption_mode` ∈ {None, "zero_extra", "shuffle_extra"}
    Used to interrogate a *trained* resD model — replay the validation set with
    extras zeroed or shuffled to measure how much it actually relies on the
    extra-band signal.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class VirtualBandResidual(nn.Module):
    def __init__(
        self,
        original_patch_embed: nn.Module,
        selected_idx: list[int],
        extra_idx: list[int],
        mask_mode: str,                  # one of {"extra","selected","all","shuffle_extra"}
        in_chans_full: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        valid_modes = ("extra", "selected", "all", "shuffle_extra")
        assert mask_mode in valid_modes, f"mask_mode={mask_mode!r} not in {valid_modes}"
        self.mask_mode = mask_mode
        self.in_chans_full = in_chans_full

        # Optional eval-time corruption applied to *the input x* before any
        # subsequent processing. Used to interrogate a trained model.
        # None | "zero_extra" | "shuffle_extra"
        self.eval_corruption_mode: str | None = None
        # Subset of channel indices to corrupt. When None, all extras
        # (self.extra_idx) are used. Allowing a subset enables per-band
        # corruption diagnostics (Phase 2).
        self.eval_corruption_bands: torch.Tensor | None = None
        # Diagnostic controls. force_delta_zero makes the wrapper an exact
        # band-selection no-op even if R has nonzero weights. residual_scale is
        # useful for eval-time ablations without rebuilding the model.
        self.force_delta_zero: bool = False
        self.residual_scale: float = 1.0

        # Explicit shuffle seeds for mask_mode='shuffle_extra' (resE) and for
        # the eval-time corruption helper. `set_train_shuffle_seed(int)` builds
        # a persistent torch.Generator so the per-step shuffle sequence during
        # training is reproducible. `eval_shuffle_seed` drives the deterministic
        # eval-time permutation used by both the resE eval forward path and
        # `_apply_extra_corruption(..., 'shuffle_extra')`.
        self.train_shuffle_seed: int | None = None
        self.eval_shuffle_seed:  int        = 42
        self._train_shuffle_gen: torch.Generator | None = None

        self.original = original_patch_embed
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.register_buffer("selected_idx", torch.tensor(selected_idx, dtype=torch.long))
        self.register_buffer("extra_idx",    torch.tensor(extra_idx,    dtype=torch.long))

        out_chans = len(selected_idx)
        self.R = nn.Sequential(
            nn.Conv2d(in_chans_full, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_chans, kernel_size=1),
        )
        # Default kaiming init for the first two Conv1x1 layers.
        # Zero-init ONLY the final Conv1x1 so delta = 0 at step 0,
        # i.e. the module is bandsel-only at the first forward pass.
        nn.init.zeros_(self.R[-1].weight)
        nn.init.zeros_(self.R[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W) or (B, C_in, 1, H, W)
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        # ---- (1) optional eval-time corruption of extras in x ----
        # Applied to a copy of x *before* the bandsel/R split, so x_sel and x_R
        # both see the corruption. selected_idx channels are never touched.
        if (not self.training) and self.eval_corruption_mode is not None:
            x_4d = self._apply_extra_corruption(x_4d, self.eval_corruption_mode)

        # ---- (2) bandsel main path: always REAL selected channels of (possibly corrupted) x ----
        x_sel = x_4d.index_select(1, self.selected_idx)            # (B, 6, H, W)

        # ---- (3) build R's input depending on mask_mode ----
        x_R = x_4d.clone()
        if self.mask_mode == "extra":
            x_R[:, self.selected_idx] = 0          # R sees only extra bands  (resB)
        elif self.mask_mode == "selected":
            x_R[:, self.extra_idx] = 0             # R sees only selected bands (resC)
        elif self.mask_mode == "all":
            pass                                   # R sees all bands as-is (resD)
        elif self.mask_mode == "shuffle_extra":
            # resE: shuffle the extras in batch dim so R cannot use the real
            # image-label alignment of the extras. Selected channels of R's
            # input are unchanged; the bandsel main path is unaffected.
            #
            # If train_shuffle_seed was set via set_train_shuffle_seed(), the
            # per-step training permutation is drawn from a persistent CPU
            # Generator (so the entire training trajectory is reproducible);
            # otherwise we fall back to non-deterministic torch.randperm.
            B = x_R.shape[0]
            if self.training:
                if self._train_shuffle_gen is not None:
                    perm = torch.randperm(B, generator=self._train_shuffle_gen).to(x_R.device)
                else:
                    perm = torch.randperm(B, device=x_R.device)
            else:
                perm = self._deterministic_perm(B, x_R.device)
            x_R[:, self.extra_idx] = x_4d[perm][:, self.extra_idx]

        # ---- (4) residual correction + add into selected-band space ----
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            delta_6 = self.R(x_R) * self.residual_scale             # (B, K, H, W)
        x_virtual = x_sel + delta_6                                 # (B, K, H, W)

        # Restore the temporal dim if the original expects 5D input.
        if was_5d:
            x_virtual = x_virtual.unsqueeze(2)
        return self.original(x_virtual)

    # ------------------------------------------------------------------
    # Helpers for corruption
    # ------------------------------------------------------------------
    def _apply_extra_corruption(self, x_4d: torch.Tensor, mode: str) -> torch.Tensor:
        """Corrupt extra channels of x; leave selected channels untouched.

        When `self.eval_corruption_bands` is set (long tensor of channel ids),
        only those channels are corrupted; otherwise all of `self.extra_idx`.
        """
        out = x_4d.clone()
        bands = self.eval_corruption_bands if self.eval_corruption_bands is not None else self.extra_idx
        if mode == "zero_extra":
            out[:, bands] = 0
        elif mode == "shuffle_extra":
            perm = self._deterministic_perm(out.shape[0], out.device)
            out[:, bands] = x_4d[perm][:, bands]
        else:
            raise ValueError(f"unknown eval_corruption_mode={mode!r}")
        return out

    @torch.no_grad()
    def measure_residual(self, x: torch.Tensor) -> dict:
        """Diagnostic forward pass: report how big the residual correction is,
        in both input-band space and token space, plus per-output-band L2.

        Uses mask_mode='all' semantics (R sees real x) so the measurement
        reflects deployment, not training-time shuffling for resE.

        Returns a dict of Python floats / lists; safe to merge into result JSON.
        """
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            delta_6 = self.R(x_4d) * self.residual_scale     # always use real x for measurement

        x_sel_norm = x_sel.norm().item() + 1e-12
        delta_norm = delta_6.norm().item()
        # per-output-band Frobenius norms
        per_band_delta = [delta_6[:, j].norm().item() for j in range(delta_6.shape[1])]

        # Token-space shift through the (frozen) pretrained patch_embed.
        # Prithvi's patch_embed expects 5D (B,C,T,H,W); the temporal dim is
        # normally added by forward_features. We bypass that here, so always
        # promote to 5D when the wrapped module looks Prithvi-like (has input_size).
        needs_5d = hasattr(self.original, "input_size")
        x_main = x_sel.unsqueeze(2)              if needs_5d else x_sel
        x_virt = (x_sel + delta_6).unsqueeze(2)  if needs_5d else (x_sel + delta_6)
        z_main = self.original(x_main)
        z_virt = self.original(x_virt)
        if z_main.dim() == 4:
            z_main = z_main.flatten(2).transpose(1, 2)
            z_virt = z_virt.flatten(2).transpose(1, 2)
        elif z_main.dim() == 5:
            z_main = z_main.squeeze(2).flatten(2).transpose(1, 2)
            z_virt = z_virt.squeeze(2).flatten(2).transpose(1, 2)
        token_shift = (z_virt - z_main).norm().item() / (z_main.norm().item() + 1e-12)

        # R[0] (first Conv1x1, shape (hidden, C_in, 1, 1)): per-input-band L2 of weights.
        w0 = self.R[0].weight.detach()
        per_input_band_w0_l2 = [w0[:, c].norm().item() for c in range(w0.shape[1])]

        return {
            "delta_l2": delta_norm,
            "x_sel_l2": x_sel.norm().item(),
            "delta_to_xsel_ratio": delta_norm / x_sel_norm,
            "token_shift_ratio": token_shift,
            "per_out_band_delta_l2": per_band_delta,                 # length 6
            "per_in_band_R0_w_l2":   per_input_band_w0_l2,            # length 13
            "R_final_layer_l2":      self.delta_l2,                   # already exposed property
        }

    def _deterministic_perm(self, B: int, device: torch.device) -> torch.Tensor:
        """Reproducible permutation of [0..B-1] that is non-identity for B>1.
        Uses a CPU-side Generator seeded with `self.eval_shuffle_seed` (default 42).
        Both the resE eval forward path and `_apply_extra_corruption(...,
        'shuffle_extra')` go through this method."""
        g = torch.Generator(device="cpu").manual_seed(int(self.eval_shuffle_seed))
        return torch.randperm(B, generator=g).to(device)

    def set_train_shuffle_seed(self, seed: int | None) -> None:
        """Initialize the persistent generator used for the resE train-time
        per-step permutation. None disables (falls back to non-deterministic
        torch.randperm). Call this AFTER construction, before training."""
        self.train_shuffle_seed = seed
        if seed is None:
            self._train_shuffle_gen = None
        else:
            self._train_shuffle_gen = torch.Generator(device="cpu").manual_seed(int(seed))

    @property
    def delta_l2(self) -> float:
        """L2 norm of the final-Conv weight (proxy for 'how much is R contributing')."""
        w = self.R[-1].weight.detach()
        return float(w.norm().item())

    def __getattr__(self, name):
        # Proxy unknown attributes to the wrapped original patch_embed so the
        # surrounding Prithvi code (which reads .input_size, .grid_size, .proj)
        # works unchanged.
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)
