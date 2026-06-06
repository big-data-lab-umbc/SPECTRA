"""ST-LoRA budgeted stage-wise LoRA rank planner.

ST-LoRA keeps the backbone frozen and chooses only per-stage LoRA ranks. It
uses a stagewise LogME profile in two possible directions:

- transfer: allocate rank to high-LogME stages.
- repair: allocate rank to high-gap stages, where gap = q_max - q_s.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .logme_profiler import StagewiseProfile
from .mgas import compute_param_cost, compute_time_cost

logger = logging.getLogger(__name__)


@dataclass
class STPlannerConfig:
    strategy: str = "transfer"  # transfer | repair
    reference_rank: int = 32
    tau: float = 0.05
    stage_prior: tuple[float, ...] = (0.8, 1.0, 1.1, 1.2)
    rank_grid: tuple[int, ...] = (4, 8, 16, 32, 64)
    candidate_budgets: tuple[int, ...] = (32, 48, 60, 72, 80, 92, 96, 104, 112)
    min_rank: int = 4
    budget_f_min: float = 0.40
    budget_f_max: float = 0.85
    budget_midpoint: float = 0.50
    budget_slope: float = 3.0
    q_bank_csv: Optional[Path] = None
    q_min_bank: Optional[float] = None
    q_max_bank: Optional[float] = None
    budget_override: Optional[int] = None
    n_stages: int = 4


@dataclass
class STPlannerPlan:
    ranks: list[int]
    unfrozen: list[bool]
    strategy: str
    budget: int
    budget_raw: float
    budget_fraction: float
    q_norm: float
    q_min_bank: float
    q_max_bank: float
    q_overall: float
    profile_scores: list[float]
    delta_q: list[float]
    stage_prior: list[float]
    logits: list[float]
    weights: list[float]
    continuous_ranks: list[float]
    param_fraction: Optional[float] = None
    gpu_fraction: Optional[float] = None

    def schedule_str(self) -> str:
        return " ".join(f"(r={r},u=0)" for r in self.ranks)


class STPlanner:
    def __init__(self, config: Optional[STPlannerConfig] = None) -> None:
        self.config = config or STPlannerConfig()
        if self.config.strategy not in {"transfer", "repair"}:
            raise ValueError(f"Unsupported ST-LoRA strategy: {self.config.strategy}")
        if self.config.tau <= 0:
            raise ValueError("ST-LoRA tau must be > 0")
        if len(self.config.stage_prior) != self.config.n_stages:
            raise ValueError("ST-LoRA stage_prior length must match n_stages")
        if any(p <= 0 for p in self.config.stage_prior):
            raise ValueError("ST-LoRA stage_prior values must be > 0")
        if self.config.min_rank not in self.config.rank_grid:
            raise ValueError("ST-LoRA min_rank must be present in rank_grid")

    def plan(
        self,
        profile: StagewiseProfile,
        embed_dim: int,
        stage_dims: list[int],
        stage_n_params: list[int],
        t_lora: list[float],
        t_full: list[float],
        gamma: float = 1.0,
    ) -> STPlannerPlan:
        cfg = self.config
        scores = [float(q) for q in profile.scores]
        delta_q = [float(dq) for dq in profile.delta_q()]
        q_overall = max(scores)
        q_min_bank, q_max_bank = self._resolve_q_bank(q_overall)
        q_norm = _clamp01((q_overall - q_min_bank) / max(q_max_bank - q_min_bank, 1e-12))
        budget_fraction = _auto_budget_fraction(
            q_norm,
            f_min=cfg.budget_f_min,
            f_max=cfg.budget_f_max,
            midpoint=cfg.budget_midpoint,
            slope=cfg.budget_slope,
        )
        max_budget = cfg.reference_rank * cfg.n_stages
        budget_raw = max_budget * budget_fraction
        budget = cfg.budget_override if cfg.budget_override is not None else _snap_down(
            budget_raw,
            cfg.candidate_budgets,
            min_budget=cfg.min_rank * cfg.n_stages,
            max_budget=max_budget,
        )

        signal = scores if cfg.strategy == "transfer" else delta_q
        logits, weights = _softmax_with_prior(signal, cfg.stage_prior, cfg.tau)
        continuous, ranks = _allocate_ranks(
            weights=weights,
            budget=budget,
            rank_grid=cfg.rank_grid,
            min_rank=cfg.min_rank,
        )
        unfrozen = [False] * cfg.n_stages
        param_fraction = compute_param_cost(ranks, unfrozen, embed_dim, stage_dims, stage_n_params)
        gpu_fraction = compute_time_cost(ranks, unfrozen, t_lora, t_full, gamma)

        plan = STPlannerPlan(
            ranks=ranks,
            unfrozen=unfrozen,
            strategy=cfg.strategy,
            budget=budget,
            budget_raw=budget_raw,
            budget_fraction=budget_fraction,
            q_norm=q_norm,
            q_min_bank=q_min_bank,
            q_max_bank=q_max_bank,
            q_overall=q_overall,
            profile_scores=scores,
            delta_q=delta_q,
            stage_prior=list(cfg.stage_prior),
            logits=logits,
            weights=weights,
            continuous_ranks=continuous,
            param_fraction=param_fraction,
            gpu_fraction=gpu_fraction,
        )
        logger.info(
            "ST-LoRA-%s plan: %s budget=%d raw=%.2f q_norm=%.3f weights=%s "
            "(param=%.3f%% gpu=%.1f%%)",
            cfg.strategy,
            plan.schedule_str(),
            budget,
            budget_raw,
            q_norm,
            [round(w, 4) for w in weights],
            100 * param_fraction,
            100 * gpu_fraction,
        )
        return plan

    def _resolve_q_bank(self, q_overall: float) -> tuple[float, float]:
        cfg = self.config
        if cfg.q_min_bank is not None and cfg.q_max_bank is not None:
            return float(cfg.q_min_bank), float(cfg.q_max_bank)
        values: list[float] = []
        if cfg.q_bank_csv is not None and cfg.q_bank_csv.exists():
            with cfg.q_bank_csv.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw = row.get("q_max")
                    if raw is None or raw == "":
                        continue
                    try:
                        values.append(float(raw))
                    except ValueError:
                        continue
        if values:
            return min(values), max(values)
        logger.warning(
            "ST-LoRA q-bank unavailable; using q_overall as both min/max, q_norm will be 0.5"
        )
        eps = 1.0
        return q_overall - eps, q_overall + eps


def _auto_budget_fraction(
    q_norm: float,
    f_min: float,
    f_max: float,
    midpoint: float,
    slope: float,
) -> float:
    return f_min + (f_max - f_min) * 0.5 * (1.0 - math.tanh(slope * (q_norm - midpoint)))


def _softmax_with_prior(
    signal: list[float],
    stage_prior: tuple[float, ...],
    tau: float,
) -> tuple[list[float], list[float]]:
    logits = [float(x) / tau + math.log(float(p)) for x, p in zip(signal, stage_prior)]
    m = max(logits)
    exp_values = [math.exp(x - m) for x in logits]
    denom = sum(exp_values)
    return logits, [x / denom for x in exp_values]


def _allocate_ranks(
    weights: list[float],
    budget: int,
    rank_grid: tuple[int, ...],
    min_rank: int,
) -> tuple[list[float], list[int]]:
    n_stages = len(weights)
    min_budget = min_rank * n_stages
    if budget < min_budget:
        raise ValueError(f"ST-LoRA budget {budget} is lower than min budget {min_budget}")
    grid = tuple(sorted(set(int(r) for r in rank_grid)))
    if min_rank not in grid:
        raise ValueError("min_rank must be in rank_grid")

    remaining = budget - min_budget
    continuous = [min_rank + remaining * float(w) for w in weights]
    ranks = [_floor_to_grid(x, grid, min_rank) for x in continuous]

    # Budget is an upper bound. Upgrade only if it reduces rounding error.
    while True:
        used = sum(ranks)
        candidates = []
        for i, rank in enumerate(ranks):
            next_rank = _next_grid_value(rank, grid)
            if next_rank is None:
                continue
            delta = next_rank - rank
            if used + delta > budget:
                continue
            benefit = abs(continuous[i] - rank) - abs(continuous[i] - next_rank)
            if benefit <= 1e-12:
                continue
            candidates.append((benefit / delta, benefit, weights[i], i, next_rank))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, _, _, i, next_rank = candidates[0]
        ranks[i] = next_rank

    return continuous, ranks


def _floor_to_grid(value: float, grid: tuple[int, ...], min_rank: int) -> int:
    candidates = [rank for rank in grid if rank <= value + 1e-12 and rank >= min_rank]
    return max(candidates) if candidates else min_rank


def _next_grid_value(rank: int, grid: tuple[int, ...]) -> Optional[int]:
    for value in grid:
        if value > rank:
            return value
    return None


def _snap_down(value: float, candidates: tuple[int, ...], min_budget: int, max_budget: int) -> int:
    valid = [int(b) for b in candidates if min_budget <= int(b) <= max_budget and int(b) <= value + 1e-12]
    if valid:
        return max(valid)
    valid = [int(b) for b in candidates if min_budget <= int(b) <= max_budget]
    if valid:
        return min(valid)
    return min(max(int(round(value)), min_budget), max_budget)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
