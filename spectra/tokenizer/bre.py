"""BRE: Band-Routed Embedding.

BRE keeps the pretrained patch embedding frozen, selects the target bands that
best match the pretraining sensor, and learns a zero-initialized correction from
all observed source bands. Each virtual target band has its own source-band
gate, which makes the learned routing inspectable after training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BRE(nn.Module):
    """Band-routed embedding adapter around a frozen pretrained patch embed."""

    def __init__(
        self,
        original_patch_embed: nn.Module,
        selected_idx: list[int],
        in_chans_full: int,
        hidden_dim: int = 32,
        gate_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mask_mode = "bre"
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
            delta = torch.zeros_like(x_sel)
        else:
            delta = self.R(self._gated_pair_input(x_4d)) * self.residual_scale
        x_virtual = x_sel + delta

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
            delta = torch.zeros_like(x_sel)
            routed = torch.zeros(
                x_4d.shape[0],
                self.out_chans * self.in_chans_full,
                x_4d.shape[2],
                x_4d.shape[3],
                device=x_4d.device,
                dtype=x_4d.dtype,
            )
        else:
            routed = self._gated_pair_input(x_4d)
            delta = self.R(routed) * self.residual_scale

        x_sel_norm = x_sel.norm().item() + 1e-12
        delta_norm = delta.norm().item()
        per_band_delta = [delta[:, j].norm().item() for j in range(delta.shape[1])]

        needs_5d = hasattr(self.original, "input_size")
        x_main = x_sel.unsqueeze(2) if needs_5d else x_sel
        x_virt = (x_sel + delta).unsqueeze(2) if needs_5d else (x_sel + delta)
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
        pair_w0_l2 = [w0[:, c].norm().item() for c in range(w0.shape[1])]
        gates = self._gate_values().detach().float()

        return {
            "delta_l2": delta_norm,
            "x_sel_l2": x_sel.norm().item(),
            "delta_to_xsel_ratio": delta_norm / x_sel_norm,
            "token_shift_ratio": token_shift,
            "per_out_band_delta_l2": per_band_delta,
            "per_gate_pair_R0_w_l2": pair_w0_l2,
            "R_final_layer_l2": self.delta_l2,
            "router_gate_mean": float(gates.mean().item()),
            "router_gate_min": float(gates.min().item()),
            "router_gate_max": float(gates.max().item()),
            "router_gate_per_target_mean": [float(v) for v in gates.mean(dim=1).cpu().tolist()],
            "router_gate_per_target_min": [float(v) for v in gates.min(dim=1).values.cpu().tolist()],
            "router_gate_per_target_max": [float(v) for v in gates.max(dim=1).values.cpu().tolist()],
            "router_top3": self.top_contributions(k=3),
            "gated_pair_input_l2": float(routed.norm().item()),
        }

    @property
    def delta_l2(self) -> float:
        return float(self.R[-1].weight.detach().norm().item())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                original = super().__getattr__("original")
            except AttributeError as e:
                raise AttributeError(name) from e
            return getattr(original, name)
