"""Band-contribution router residual adapter.

This adapter keeps the pretrained patch embedding frozen and learns an
interpretable source-band contribution table before a small residual correction.

For each pretrained target band j:
    routed_j = sum_i softmax(router_logits[j])[i] * x_i
    delta_j  = R(routed)_j
    x_virtual_j = x_selected_j + delta_j

The final layer of R is zero-initialized, so the step-0 forward is exactly
equivalent to selected-band patch embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BandContributionRouterResidual(nn.Module):
    """Static all-band router feeding a zero-init residual correction."""

    def __init__(
        self,
        original_patch_embed: nn.Module,
        selected_idx: list[int],
        candidate_idx: list[int],
        in_chans_full: int,
        hidden_dim: int = 32,
        router_init: str = "uniform",
        anchor_logit: float = 4.0,
    ) -> None:
        super().__init__()
        self.mask_mode = "router_static_all"
        self.in_chans_full = in_chans_full
        self.force_delta_zero: bool = False
        self.residual_scale: float = 1.0
        self.train_shuffle_seed: int | None = None
        self.eval_shuffle_seed: int = 42

        self.original = original_patch_embed
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.register_buffer("selected_idx", torch.tensor(selected_idx, dtype=torch.long))
        self.register_buffer("candidate_idx", torch.tensor(candidate_idx, dtype=torch.long))
        extra_idx = [i for i in range(in_chans_full) if i not in selected_idx]
        self.register_buffer("extra_idx", torch.tensor(extra_idx, dtype=torch.long))

        out_chans = len(selected_idx)
        in_chans_router = len(candidate_idx)
        self.router_logits = nn.Parameter(torch.zeros(out_chans, in_chans_router))
        if router_init == "anchor":
            candidate_pos = {int(b): i for i, b in enumerate(candidate_idx)}
            with torch.no_grad():
                for row, band in enumerate(selected_idx):
                    pos = candidate_pos.get(int(band))
                    if pos is not None:
                        self.router_logits[row, pos] = float(anchor_logit)
        elif router_init != "uniform":
            raise ValueError(f"unknown router_init={router_init!r}")
        self.R = nn.Sequential(
            nn.Conv2d(out_chans, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_chans, kernel_size=1),
        )
        nn.init.zeros_(self.R[-1].weight)
        nn.init.zeros_(self.R[-1].bias)

    def adapter_parameters(self):
        yield self.router_logits
        yield from self.R.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            routed = self._route_candidates(x_4d)
            delta_6 = self.R(routed) * self.residual_scale
        x_virtual = x_sel + delta_6

        if was_5d:
            x_virtual = x_virtual.unsqueeze(2)
        return self.original(x_virtual)

    def _route_candidates(self, x_4d: torch.Tensor) -> torch.Tensor:
        x_candidates = x_4d.index_select(1, self.candidate_idx)
        weights = torch.softmax(self.router_logits, dim=-1)
        return torch.einsum("kc,bchw->bkhw", weights, x_candidates)

    def set_train_shuffle_seed(self, seed: int | None) -> None:
        self.train_shuffle_seed = seed

    @torch.no_grad()
    def contribution_matrix(self) -> list[list[float]]:
        weights = torch.softmax(self.router_logits.detach().float(), dim=-1)
        return [[float(v) for v in row.cpu().tolist()] for row in weights]

    @torch.no_grad()
    def contribution_matrix_full(self) -> list[list[float]]:
        weights = torch.softmax(self.router_logits.detach().float(), dim=-1).cpu()
        full = torch.zeros(weights.shape[0], self.in_chans_full, dtype=weights.dtype)
        full[:, self.candidate_idx.cpu()] = weights
        return [[float(v) for v in row.tolist()] for row in full]

    @torch.no_grad()
    def top_contributions(self, k: int = 3) -> list[list[dict[str, float | int]]]:
        weights = torch.softmax(self.router_logits.detach().float(), dim=-1).cpu()
        candidate_idx = self.candidate_idx.cpu()
        topk = min(k, weights.shape[1])
        rows = []
        for row in weights:
            vals, idxs = torch.topk(row, topk)
            rows.append([
                {"band": int(candidate_idx[int(i)].item()), "weight": float(v.item())}
                for v, i in zip(vals, idxs)
            ])
        return rows

    @torch.no_grad()
    def measure_residual(self, x: torch.Tensor) -> dict:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
            routed = torch.zeros_like(x_sel)
        else:
            routed = self._route_candidates(x_4d)
            delta_6 = self.R(routed) * self.residual_scale

        x_sel_norm = x_sel.norm().item() + 1e-12
        delta_norm = delta_6.norm().item()
        per_band_delta = [delta_6[:, j].norm().item() for j in range(delta_6.shape[1])]

        needs_5d = hasattr(self.original, "input_size")
        x_main = x_sel.unsqueeze(2) if needs_5d else x_sel
        x_virt = (x_sel + delta_6).unsqueeze(2) if needs_5d else (x_sel + delta_6)
        z_main = self.original(x_main)
        z_virt = self.original(x_virt)
        if z_main.dim() == 4:
            z_main = z_main.flatten(2).transpose(1, 2)
            z_virt = z_virt.flatten(2).transpose(1, 2)
        elif z_main.dim() == 5:
            z_main = z_main.squeeze(2).flatten(2).transpose(1, 2)
            z_virt = z_virt.squeeze(2).flatten(2).transpose(1, 2)
        token_shift = (z_virt - z_main).norm().item() / (z_main.norm().item() + 1e-12)

        w0 = self.R[0].weight.detach()
        per_routed_band_w0_l2 = [w0[:, c].norm().item() for c in range(w0.shape[1])]
        weights = torch.softmax(self.router_logits.detach().float(), dim=-1)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)

        return {
            "delta_l2": delta_norm,
            "x_sel_l2": x_sel.norm().item(),
            "delta_to_xsel_ratio": delta_norm / x_sel_norm,
            "token_shift_ratio": token_shift,
            "per_out_band_delta_l2": per_band_delta,
            "per_in_band_R0_w_l2": per_routed_band_w0_l2,
            "R_final_layer_l2": self.delta_l2,
            "router_entropy_mean": float(entropy.mean().item()),
            "router_entropy_per_target": [float(v) for v in entropy.cpu().tolist()],
            "router_top3": self.top_contributions(k=3),
            "routed_l2": float(routed.norm().item()),
        }

    @property
    def delta_l2(self) -> float:
        w = self.R[-1].weight.detach()
        return float(w.norm().item())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)


class BandGatedResidual(nn.Module):
    """D-style all-band residual with trainable, interpretable band gates.

    Unlike the static router, this does not compress the input bands before the
    residual branch. The residual adapter still sees all source channels, but
    each channel is modulated by a learned gate initialized to 1.0.
    """

    def __init__(
        self,
        original_patch_embed: nn.Module,
        selected_idx: list[int],
        in_chans_full: int,
        hidden_dim: int = 32,
        gate_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mask_mode = "bre_gatedD"
        self.in_chans_full = in_chans_full
        self.force_delta_zero: bool = False
        self.residual_scale: float = 1.0
        self.train_shuffle_seed: int | None = None
        self.eval_shuffle_seed: int = 42
        self.gate_max = float(gate_max)

        self.original = original_patch_embed
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.register_buffer("selected_idx", torch.tensor(selected_idx, dtype=torch.long))
        extra_idx = [i for i in range(in_chans_full) if i not in selected_idx]
        self.register_buffer("extra_idx", torch.tensor(extra_idx, dtype=torch.long))
        self.register_buffer("candidate_idx", torch.arange(in_chans_full, dtype=torch.long))

        out_chans = len(selected_idx)
        self.router_logits = nn.Parameter(torch.zeros(in_chans_full))
        self.R = nn.Sequential(
            nn.Conv2d(in_chans_full, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_chans, kernel_size=1),
        )
        nn.init.zeros_(self.R[-1].weight)
        nn.init.zeros_(self.R[-1].bias)

    def adapter_parameters(self):
        yield self.router_logits
        yield from self.R.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            x_R = x_4d * self._gate_values().view(1, -1, 1, 1)
            delta_6 = self.R(x_R) * self.residual_scale
        x_virtual = x_sel + delta_6

        if was_5d:
            x_virtual = x_virtual.unsqueeze(2)
        return self.original(x_virtual)

    def _gate_values(self) -> torch.Tensor:
        return self.gate_max * torch.sigmoid(self.router_logits)

    def set_train_shuffle_seed(self, seed: int | None) -> None:
        self.train_shuffle_seed = seed

    @torch.no_grad()
    def gate_values(self) -> list[float]:
        return [float(v) for v in self._gate_values().detach().float().cpu().tolist()]

    @torch.no_grad()
    def contribution_matrix_full(self) -> list[list[float]]:
        gates = self._gate_values().detach().float().cpu()
        return [[float(v) for v in gates.tolist()] for _ in range(len(self.selected_idx))]

    @torch.no_grad()
    def top_contributions(self, k: int = 3) -> list[list[dict[str, float | int]]]:
        gates = self._gate_values().detach().float().cpu()
        topk = min(k, gates.numel())
        vals, idxs = torch.topk(gates, topk)
        row = [
            {"band": int(i.item()), "weight": float(v.item())}
            for v, i in zip(vals, idxs)
        ]
        return [list(row) for _ in range(len(self.selected_idx))]

    @torch.no_grad()
    def measure_residual(self, x: torch.Tensor) -> dict:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
            x_R = torch.zeros_like(x_4d)
        else:
            x_R = x_4d * self._gate_values().view(1, -1, 1, 1)
            delta_6 = self.R(x_R) * self.residual_scale

        x_sel_norm = x_sel.norm().item() + 1e-12
        delta_norm = delta_6.norm().item()
        per_band_delta = [delta_6[:, j].norm().item() for j in range(delta_6.shape[1])]

        needs_5d = hasattr(self.original, "input_size")
        x_main = x_sel.unsqueeze(2) if needs_5d else x_sel
        x_virt = (x_sel + delta_6).unsqueeze(2) if needs_5d else (x_sel + delta_6)
        z_main = self.original(x_main)
        z_virt = self.original(x_virt)
        if z_main.dim() == 4:
            z_main = z_main.flatten(2).transpose(1, 2)
            z_virt = z_virt.flatten(2).transpose(1, 2)
        elif z_main.dim() == 5:
            z_main = z_main.squeeze(2).flatten(2).transpose(1, 2)
            z_virt = z_virt.squeeze(2).flatten(2).transpose(1, 2)
        token_shift = (z_virt - z_main).norm().item() / (z_main.norm().item() + 1e-12)

        w0 = self.R[0].weight.detach()
        per_input_band_w0_l2 = [w0[:, c].norm().item() for c in range(w0.shape[1])]
        gates = self._gate_values().detach().float()

        return {
            "delta_l2": delta_norm,
            "x_sel_l2": x_sel.norm().item(),
            "delta_to_xsel_ratio": delta_norm / x_sel_norm,
            "token_shift_ratio": token_shift,
            "per_out_band_delta_l2": per_band_delta,
            "per_in_band_R0_w_l2": per_input_band_w0_l2,
            "R_final_layer_l2": self.delta_l2,
            "router_gate_mean": float(gates.mean().item()),
            "router_gate_min": float(gates.min().item()),
            "router_gate_max": float(gates.max().item()),
            "router_top3": self.top_contributions(k=3),
            "gated_input_l2": float(x_R.norm().item()),
        }

    @property
    def delta_l2(self) -> float:
        w = self.R[-1].weight.detach()
        return float(w.norm().item())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)


class LightweightResidualGate(nn.Module):
    """D-style all-band residual with learned per-output residual gates.

    This keeps the original resD adapter input and hidden structure unchanged:
    all source bands are passed to R, and R predicts a 6-channel correction in
    pretrained-band space. The only extra parameters are one learned gate per
    virtual output band, initialized to 1.0 through 2 * sigmoid(0).
    """

    def __init__(
        self,
        original_patch_embed: nn.Module,
        selected_idx: list[int],
        in_chans_full: int,
        hidden_dim: int = 32,
        gate_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mask_mode = "bre_light_gate"
        self.in_chans_full = in_chans_full
        self.force_delta_zero: bool = False
        self.residual_scale: float = 1.0
        self.train_shuffle_seed: int | None = None
        self.eval_shuffle_seed: int = 42
        self.gate_max = float(gate_max)

        self.original = original_patch_embed
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.register_buffer("selected_idx", torch.tensor(selected_idx, dtype=torch.long))
        extra_idx = [i for i in range(in_chans_full) if i not in selected_idx]
        self.register_buffer("extra_idx", torch.tensor(extra_idx, dtype=torch.long))

        out_chans = len(selected_idx)
        self.gate_logits = nn.Parameter(torch.zeros(out_chans))
        self.R = nn.Sequential(
            nn.Conv2d(in_chans_full, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_chans, kernel_size=1),
        )
        nn.init.zeros_(self.R[-1].weight)
        nn.init.zeros_(self.R[-1].bias)

    def adapter_parameters(self):
        yield self.gate_logits
        yield from self.R.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            gates = self._gate_values().view(1, -1, 1, 1)
            delta_6 = self.R(x_4d) * gates * self.residual_scale
        x_virtual = x_sel + delta_6

        if was_5d:
            x_virtual = x_virtual.unsqueeze(2)
        return self.original(x_virtual)

    def _gate_values(self) -> torch.Tensor:
        return self.gate_max * torch.sigmoid(self.gate_logits)

    def set_train_shuffle_seed(self, seed: int | None) -> None:
        self.train_shuffle_seed = seed

    @torch.no_grad()
    def gate_values(self) -> list[float]:
        return [float(v) for v in self._gate_values().detach().float().cpu().tolist()]

    @torch.no_grad()
    def measure_residual(self, x: torch.Tensor) -> dict:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            gates = self._gate_values().view(1, -1, 1, 1)
            delta_6 = self.R(x_4d) * gates * self.residual_scale

        x_sel_norm = x_sel.norm().item() + 1e-12
        delta_norm = delta_6.norm().item()
        per_band_delta = [delta_6[:, j].norm().item() for j in range(delta_6.shape[1])]

        needs_5d = hasattr(self.original, "input_size")
        x_main = x_sel.unsqueeze(2) if needs_5d else x_sel
        x_virt = (x_sel + delta_6).unsqueeze(2) if needs_5d else (x_sel + delta_6)
        z_main = self.original(x_main)
        z_virt = self.original(x_virt)
        if z_main.dim() == 4:
            z_main = z_main.flatten(2).transpose(1, 2)
            z_virt = z_virt.flatten(2).transpose(1, 2)
        elif z_main.dim() == 5:
            z_main = z_main.squeeze(2).flatten(2).transpose(1, 2)
            z_virt = z_virt.squeeze(2).flatten(2).transpose(1, 2)
        token_shift = (z_virt - z_main).norm().item() / (z_main.norm().item() + 1e-12)

        w0 = self.R[0].weight.detach()
        per_input_band_w0_l2 = [w0[:, c].norm().item() for c in range(w0.shape[1])]
        gates = self._gate_values().detach().float()

        return {
            "delta_l2": delta_norm,
            "x_sel_l2": x_sel.norm().item(),
            "delta_to_xsel_ratio": delta_norm / x_sel_norm,
            "token_shift_ratio": token_shift,
            "per_out_band_delta_l2": per_band_delta,
            "per_in_band_R0_w_l2": per_input_band_w0_l2,
            "R_final_layer_l2": self.delta_l2,
            "router_gate_mean": float(gates.mean().item()),
            "router_gate_min": float(gates.min().item()),
            "router_gate_max": float(gates.max().item()),
            "residual_output_gates": [float(v) for v in gates.cpu().tolist()],
        }

    @property
    def delta_l2(self) -> float:
        w = self.R[-1].weight.detach()
        return float(w.norm().item())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)


class BandPerOutputGatedResidual(nn.Module):
    """D-style all-band residual with per-output source-band gates.

    `BandGatedResidual` learns one gate per source band and shares it across all
    six virtual output bands. This variant learns a K x C gate table, where K is
    the number of selected/pretrained target bands and C is the number of
    observed source bands. That lets each virtual output band use a different
    source-band contribution pattern while keeping the residual branch
    zero-initialized as an exact band-selection no-op at step 0.
    """

    def __init__(
        self,
        original_patch_embed: nn.Module,
        selected_idx: list[int],
        in_chans_full: int,
        hidden_dim: int = 32,
        gate_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mask_mode = "bre_perout_gate"
        self.in_chans_full = in_chans_full
        self.force_delta_zero: bool = False
        self.residual_scale: float = 1.0
        self.train_shuffle_seed: int | None = None
        self.eval_shuffle_seed: int = 42
        self.gate_max = float(gate_max)

        self.original = original_patch_embed
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.register_buffer("selected_idx", torch.tensor(selected_idx, dtype=torch.long))
        extra_idx = [i for i in range(in_chans_full) if i not in selected_idx]
        self.register_buffer("extra_idx", torch.tensor(extra_idx, dtype=torch.long))
        self.register_buffer("candidate_idx", torch.arange(in_chans_full, dtype=torch.long))

        out_chans = len(selected_idx)
        self.out_chans = out_chans
        self.gate_logits = nn.Parameter(torch.zeros(out_chans, in_chans_full))
        self.R = nn.Sequential(
            nn.Conv2d(out_chans * in_chans_full, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_chans, kernel_size=1),
        )
        nn.init.zeros_(self.R[-1].weight)
        nn.init.zeros_(self.R[-1].bias)

    def adapter_parameters(self):
        yield self.gate_logits
        yield from self.R.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            x_R = self._gated_pair_input(x_4d)
            delta_6 = self.R(x_R) * self.residual_scale
        x_virtual = x_sel + delta_6

        if was_5d:
            x_virtual = x_virtual.unsqueeze(2)
        return self.original(x_virtual)

    def _gate_values(self) -> torch.Tensor:
        return self.gate_max * torch.sigmoid(self.gate_logits)

    def _gated_pair_input(self, x_4d: torch.Tensor) -> torch.Tensor:
        gates = self._gate_values()
        x_pair = x_4d.unsqueeze(1) * gates.view(1, self.out_chans, self.in_chans_full, 1, 1)
        return x_pair.flatten(1, 2)

    def set_train_shuffle_seed(self, seed: int | None) -> None:
        self.train_shuffle_seed = seed

    @torch.no_grad()
    def gate_values(self) -> list[list[float]]:
        gates = self._gate_values().detach().float().cpu()
        return [[float(v) for v in row.tolist()] for row in gates]

    @torch.no_grad()
    def contribution_matrix_full(self) -> list[list[float]]:
        return self.gate_values()

    @torch.no_grad()
    def top_contributions(self, k: int = 3) -> list[list[dict[str, float | int]]]:
        gates = self._gate_values().detach().float().cpu()
        candidate_idx = self.candidate_idx.cpu()
        topk = min(k, gates.shape[1])
        rows = []
        for row in gates:
            vals, idxs = torch.topk(row, topk)
            rows.append([
                {"band": int(candidate_idx[int(i)].item()), "weight": float(v.item())}
                for v, i in zip(vals, idxs)
            ])
        return rows

    @torch.no_grad()
    def measure_residual(self, x: torch.Tensor) -> dict:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
            x_R = torch.zeros(
                x_4d.shape[0],
                self.out_chans * self.in_chans_full,
                x_4d.shape[2],
                x_4d.shape[3],
                device=x_4d.device,
                dtype=x_4d.dtype,
            )
        else:
            x_R = self._gated_pair_input(x_4d)
            delta_6 = self.R(x_R) * self.residual_scale

        x_sel_norm = x_sel.norm().item() + 1e-12
        delta_norm = delta_6.norm().item()
        per_band_delta = [delta_6[:, j].norm().item() for j in range(delta_6.shape[1])]

        needs_5d = hasattr(self.original, "input_size")
        x_main = x_sel.unsqueeze(2) if needs_5d else x_sel
        x_virt = (x_sel + delta_6).unsqueeze(2) if needs_5d else (x_sel + delta_6)
        z_main = self.original(x_main)
        z_virt = self.original(x_virt)
        if z_main.dim() == 4:
            z_main = z_main.flatten(2).transpose(1, 2)
            z_virt = z_virt.flatten(2).transpose(1, 2)
        elif z_main.dim() == 5:
            z_main = z_main.squeeze(2).flatten(2).transpose(1, 2)
            z_virt = z_virt.squeeze(2).flatten(2).transpose(1, 2)
        token_shift = (z_virt - z_main).norm().item() / (z_main.norm().item() + 1e-12)

        w0 = self.R[0].weight.detach()
        per_pair_w0_l2 = [w0[:, c].norm().item() for c in range(w0.shape[1])]
        gates = self._gate_values().detach().float()

        return {
            "delta_l2": delta_norm,
            "x_sel_l2": x_sel.norm().item(),
            "delta_to_xsel_ratio": delta_norm / x_sel_norm,
            "token_shift_ratio": token_shift,
            "per_out_band_delta_l2": per_band_delta,
            "per_gate_pair_R0_w_l2": per_pair_w0_l2,
            "R_final_layer_l2": self.delta_l2,
            "router_gate_mean": float(gates.mean().item()),
            "router_gate_min": float(gates.min().item()),
            "router_gate_max": float(gates.max().item()),
            "router_gate_per_target_mean": [float(v) for v in gates.mean(dim=1).cpu().tolist()],
            "router_gate_per_target_min": [float(v) for v in gates.min(dim=1).values.cpu().tolist()],
            "router_gate_per_target_max": [float(v) for v in gates.max(dim=1).values.cpu().tolist()],
            "router_top3": self.top_contributions(k=3),
            "gated_pair_input_l2": float(x_R.norm().item()),
        }

    @property
    def delta_l2(self) -> float:
        w = self.R[-1].weight.detach()
        return float(w.norm().item())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)


class BandSourceToTargetGatedResidual(nn.Module):
    """Source-to-target gated residual with target-wise grouped adapters.

    This is the grouped version of `BandPerOutputGatedResidual`. Both variants
    learn a K x C source-to-target gate table, where K is the number of
    pretrained-compatible virtual target bands and C is the number of observed
    source bands. The difference is the residual adapter:

    * BandPerOutputGatedResidual: one shared 6C -> 32 -> 32 -> 6 adapter.
    * This class: K grouped adapters, each C -> hidden -> hidden -> 1.

    The final grouped 1x1 convolution is zero-initialized, so the step-0
    forward is exactly the selected-band baseline.
    """

    def __init__(
        self,
        original_patch_embed: nn.Module,
        selected_idx: list[int],
        in_chans_full: int,
        hidden_dim: int = 32,
        gate_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mask_mode = "bre_s2t"
        self.in_chans_full = in_chans_full
        self.force_delta_zero: bool = False
        self.residual_scale: float = 1.0
        self.train_shuffle_seed: int | None = None
        self.eval_shuffle_seed: int = 42
        self.gate_max = float(gate_max)

        self.original = original_patch_embed
        for p in self.original.parameters():
            p.requires_grad_(False)

        self.register_buffer("selected_idx", torch.tensor(selected_idx, dtype=torch.long))
        extra_idx = [i for i in range(in_chans_full) if i not in selected_idx]
        self.register_buffer("extra_idx", torch.tensor(extra_idx, dtype=torch.long))
        self.register_buffer("candidate_idx", torch.arange(in_chans_full, dtype=torch.long))

        out_chans = len(selected_idx)
        self.out_chans = out_chans
        self.hidden_dim = hidden_dim
        self.gate_logits = nn.Parameter(torch.zeros(out_chans, in_chans_full))
        self.R = nn.Sequential(
            nn.Conv2d(out_chans * in_chans_full, out_chans * hidden_dim, kernel_size=1, groups=out_chans),
            nn.GELU(),
            nn.Conv2d(out_chans * hidden_dim, out_chans * hidden_dim, kernel_size=1, groups=out_chans),
            nn.GELU(),
            nn.Conv2d(out_chans * hidden_dim, out_chans, kernel_size=1, groups=out_chans),
        )
        nn.init.zeros_(self.R[-1].weight)
        nn.init.zeros_(self.R[-1].bias)

    def adapter_parameters(self):
        yield self.gate_logits
        yield from self.R.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
        else:
            x_R = self._gated_pair_input(x_4d)
            delta_6 = self.R(x_R) * self.residual_scale
        x_virtual = x_sel + delta_6

        if was_5d:
            x_virtual = x_virtual.unsqueeze(2)
        return self.original(x_virtual)

    def _gate_values(self) -> torch.Tensor:
        return self.gate_max * torch.sigmoid(self.gate_logits)

    def _gated_pair_input(self, x_4d: torch.Tensor) -> torch.Tensor:
        gates = self._gate_values()
        x_pair = x_4d.unsqueeze(1) * gates.view(1, self.out_chans, self.in_chans_full, 1, 1)
        return x_pair.flatten(1, 2)

    def set_train_shuffle_seed(self, seed: int | None) -> None:
        self.train_shuffle_seed = seed

    @torch.no_grad()
    def gate_values(self) -> list[list[float]]:
        gates = self._gate_values().detach().float().cpu()
        return [[float(v) for v in row.tolist()] for row in gates]

    @torch.no_grad()
    def contribution_matrix_full(self) -> list[list[float]]:
        return self.gate_values()

    @torch.no_grad()
    def top_contributions(self, k: int = 3) -> list[list[dict[str, float | int]]]:
        gates = self._gate_values().detach().float().cpu()
        candidate_idx = self.candidate_idx.cpu()
        topk = min(k, gates.shape[1])
        rows = []
        for row in gates:
            vals, idxs = torch.topk(row, topk)
            rows.append([
                {"band": int(candidate_idx[int(i)].item()), "weight": float(v.item())}
                for v, i in zip(vals, idxs)
            ])
        return rows

    @torch.no_grad()
    def measure_residual(self, x: torch.Tensor) -> dict:
        was_5d = x.dim() == 5
        x_4d = x[:, :, 0] if was_5d else x

        x_sel = x_4d.index_select(1, self.selected_idx)
        if self.force_delta_zero:
            delta_6 = torch.zeros_like(x_sel)
            x_R = torch.zeros(
                x_4d.shape[0],
                self.out_chans * self.in_chans_full,
                x_4d.shape[2],
                x_4d.shape[3],
                device=x_4d.device,
                dtype=x_4d.dtype,
            )
        else:
            x_R = self._gated_pair_input(x_4d)
            delta_6 = self.R(x_R) * self.residual_scale

        x_sel_norm = x_sel.norm().item() + 1e-12
        delta_norm = delta_6.norm().item()
        per_band_delta = [delta_6[:, j].norm().item() for j in range(delta_6.shape[1])]

        needs_5d = hasattr(self.original, "input_size")
        x_main = x_sel.unsqueeze(2) if needs_5d else x_sel
        x_virt = (x_sel + delta_6).unsqueeze(2) if needs_5d else (x_sel + delta_6)
        z_main = self.original(x_main)
        z_virt = self.original(x_virt)
        if z_main.dim() == 4:
            z_main = z_main.flatten(2).transpose(1, 2)
            z_virt = z_virt.flatten(2).transpose(1, 2)
        elif z_main.dim() == 5:
            z_main = z_main.squeeze(2).flatten(2).transpose(1, 2)
            z_virt = z_virt.squeeze(2).flatten(2).transpose(1, 2)
        token_shift = (z_virt - z_main).norm().item() / (z_main.norm().item() + 1e-12)

        w0 = self.R[0].weight.detach()
        per_pair_w0_l2 = [w0[:, c].norm().item() for c in range(w0.shape[1])]
        gates = self._gate_values().detach().float()

        return {
            "delta_l2": delta_norm,
            "x_sel_l2": x_sel.norm().item(),
            "delta_to_xsel_ratio": delta_norm / x_sel_norm,
            "token_shift_ratio": token_shift,
            "per_out_band_delta_l2": per_band_delta,
            "per_gate_pair_R0_w_l2": per_pair_w0_l2,
            "R_final_layer_l2": self.delta_l2,
            "router_gate_mean": float(gates.mean().item()),
            "router_gate_min": float(gates.min().item()),
            "router_gate_max": float(gates.max().item()),
            "router_gate_per_target_mean": [float(v) for v in gates.mean(dim=1).cpu().tolist()],
            "router_gate_per_target_min": [float(v) for v in gates.min(dim=1).values.cpu().tolist()],
            "router_gate_per_target_max": [float(v) for v in gates.max(dim=1).values.cpu().tolist()],
            "router_top3": self.top_contributions(k=3),
            "gated_pair_input_l2": float(x_R.norm().item()),
        }

    @property
    def delta_l2(self) -> float:
        w = self.R[-1].weight.detach()
        return float(w.norm().item())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)
