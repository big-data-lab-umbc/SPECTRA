"""MGAS: Monotonicity-Guarded Adaptation Scheduler.

Closed-form, zero-parameter rule mapping a one-shot stagewise LogME profile
to per-stage (r_s, u_s) under a joint parameter-and-time budget.

Reference (from proposal):
    k_s = staircase(Δq[s]):  < 0.02 → 0;  < 0.10 → 4;  < 0.25 → 8;  else → 16
    s*  = min { s : Δq[s] ≤ ε }, ε = 0.02
    s ≤ s* → (r_s, u_s) = (k_s, 0)   # frozen + LoRA
    s > s* → (r_s, u_s) = (k_s, 1)   # unfrozen
    Budget reduction: while over budget, reduce arg_max_s r_s by one step
                      in descending-s order (16→8→4→0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .logme_profiler import StagewiseProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MGASConfig:
    epsilon: float = 0.02            # suffix-cut threshold (Δq[s] ≤ ε → s ≤ s*)
    delta: float = 0.01              # monotonicity tolerance (legacy, unused with max-ref Δq)
    b_param: float = 0.50            # max fraction of full-FT params
    b_gpu: float = 0.50             # max fraction of full-FT GPU-h
    n_stages: int = 4
    rank_steps: tuple[int, ...] = (0, 4, 8, 16)

    # Staircase thresholds (Δq → rank)
    staircase_thresholds: tuple[float, ...] = (0.02, 0.10, 0.25)
    staircase_ranks: tuple[int, ...]       = (0, 4, 8, 16)

    # Minimum Δq required to unfreeze a stage (on top of s* suffix-cut condition).
    # Prevents s*=0 from unfreezing all stages when Δq values are uniformly small.
    # Stages with Δq < unfreeze_min_delta_q use LoRA only (backbone stays frozen).
    # Default = top staircase threshold: only unfreeze when transferability gap ≥ 0.25.
    unfreeze_min_delta_q: float = 0.25

    # Minimum LoRA rank assigned to any stage (floor applied after staircase).
    # Default=0 preserves original behaviour (staircase can assign rank=0).
    # Set to 4 for cross-sensor cells: even "transferable" early stages need some
    # adaptation because SSC-PE's cross-sensor mapping is imperfect. Without this,
    # MGAS assigns rank=0 to stages 0/1 on sen1floods11, leaving only 12/24 blocks
    # with LoRA — insufficient vs LoRA-8 (rank=8 at all 24 blocks, mIoU=0.726).
    min_rank: int = 0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class MGASPlan:
    ranks: list[int]              # per-stage LoRA rank [r_0, r_1, r_2, r_3]
    unfrozen: list[bool]          # per-stage unfreeze flag [u_0, u_1, u_2, u_3]
    suffix_cut: int               # s* (0-indexed)
    precondition_ok: bool
    budget_reduced: bool          # True if budget constraint reduced any rank
    delta_q: list[float]
    param_fraction: Optional[float] = None
    gpu_fraction: Optional[float]   = None

    def schedule_str(self) -> str:
        pairs = [f"(r={r},u={int(u)})" for r, u in zip(self.ranks, self.unfrozen)]
        return " ".join(pairs)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def staircase_g(delta_q_s: float, config: MGASConfig) -> int:
    """Map Δq[s] → LoRA rank k_s via the staircase function, floored at config.min_rank."""
    thresholds = config.staircase_thresholds
    ranks      = config.staircase_ranks
    for thresh, rank in zip(thresholds, ranks):
        if delta_q_s < thresh:
            return max(rank, config.min_rank)
    return ranks[-1]   # else → 16


def find_suffix_cut(delta_q: list[float], epsilon: float) -> int:
    """s* = min {s : Δq[s] ≤ ε}. Returns n_stages-1 if no stage qualifies."""
    for s, dq in enumerate(delta_q):
        if dq <= epsilon:
            return s
    return len(delta_q) - 1


def compute_param_cost(
    ranks: list[int],
    unfrozen: list[bool],
    embed_dim: int,
    stage_dims: list[int],
    stage_n_params: list[int],
) -> float:
    """Compute Σ_s c_s^param as fraction of full-FT parameter count."""
    total_lora  = sum(r * (embed_dim + d) for r, d in zip(ranks, stage_dims))
    total_unfrz = sum(u * n for u, n in zip(unfrozen, stage_n_params))
    full_ft     = sum(stage_n_params)
    return (total_lora + total_unfrz) / max(full_ft, 1)


def compute_time_cost(
    ranks: list[int],
    unfrozen: list[bool],
    t_lora: list[float],
    t_full: list[float],
    gamma: float = 1.0,
) -> float:
    """Compute Σ_s c_s^time as fraction of full-FT GPU-h.

    LoRA cost is treated as BINARY (on/off, independent of rank) because
    timing measurements on Prithvi ViT-L show rank-8 costs only ~2.5% more
    than rank-1 (A/B overhead is <2% of attention compute). Using r * t_lora
    would drastically overestimate cost for high ranks.
    """
    total_time = gamma * sum(int(r > 0) * tl + int(u) * tf
                             for r, u, tl, tf in zip(ranks, unfrozen, t_lora, t_full))
    full_time  = gamma * sum(t_full)
    return total_time / max(full_time, 1e-9)


def reduce_budget(
    ranks: list[int],
    unfrozen: list[bool],
    embed_dim: int,
    stage_dims: list[int],
    stage_n_params: list[int],
    t_lora: list[float],
    t_full: list[float],
    b_param: float,
    b_gpu: float,
    gamma: float,
    rank_steps: tuple[int, ...],
    delta_q: Optional[list[float]] = None,
) -> tuple[list[int], list[bool], bool]:
    """Iteratively reduce the highest-s, highest-r stage until both budgets are met.

    Priority 1: reduce ranks (highest rank at highest stage first).
    Priority 2 (safety net): if all ranks are at minimum and budget still violated,
    freeze unfrozen stages starting from the one with smallest Δq (least adaptation
    needed), so the most-needed unfrozen stages are preserved.
    """
    ranks    = list(ranks)
    unfrozen = list(unfrozen)
    reduced  = False

    for _ in range(200):   # guard against infinite loop
        p_frac = compute_param_cost(ranks, unfrozen, embed_dim, stage_dims, stage_n_params)
        t_frac = compute_time_cost(ranks, unfrozen, t_lora, t_full, gamma)
        if p_frac <= b_param and t_frac <= b_gpu:
            break
        # Priority 1: reduce highest-rank stage
        best_s = _highest_rank_stage(ranks, descending_s=True, rank_steps=rank_steps)
        if best_s is not None:
            current = ranks[best_s]
            idx = rank_steps.index(current)
            ranks[best_s] = rank_steps[idx - 1] if idx > 0 else 0
            reduced = True
            continue
        # Priority 2: freeze the unfrozen stage with smallest Δq (least adaptation need)
        unfrozen_stages = [(s, (delta_q[s] if delta_q is not None else float("inf")))
                           for s in range(len(unfrozen)) if unfrozen[s]]
        if unfrozen_stages:
            s_to_freeze = min(unfrozen_stages, key=lambda x: x[1])[0]
            unfrozen[s_to_freeze] = False
            reduced = True
        else:
            break   # nothing left to reduce

    return ranks, unfrozen, reduced


def _highest_rank_stage(
    ranks: list[int], descending_s: bool, rank_steps: tuple[int, ...]
) -> Optional[int]:
    """Return index of stage with highest reducible rank (prefer high-s stages)."""
    # reducible = rank > minimum (0 or 4)
    min_rank = rank_steps[1] if len(rank_steps) > 1 else 0
    candidates = [(s, r) for s, r in enumerate(ranks) if r > min_rank]
    if not candidates:
        # Try to reduce from any non-zero rank
        candidates = [(s, r) for s, r in enumerate(ranks) if r > 0]
    if not candidates:
        return None
    # sort by rank descending, then by stage descending
    candidates.sort(key=lambda x: (x[1], x[0] if descending_s else -x[0]), reverse=True)
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

class MGASPlanner:
    """MGAS: Monotonicity-Guarded Adaptation Scheduler.

    Typical usage::
        planner = MGASPlanner(config=MGASConfig())
        plan = planner.plan(profile, embed_dim=1024, stage_dims=[...], ...)
        lora_backbone.apply_schedule(plan.ranks, plan.unfrozen)
    """

    def __init__(self, config: Optional[MGASConfig] = None) -> None:
        self.config = config or MGASConfig()

    def plan(
        self,
        profile: StagewiseProfile,
        embed_dim: int,
        stage_dims: list[int],
        stage_n_params: list[int],
        t_lora: list[float],
        t_full: list[float],
        gamma: float = 1.0,
    ) -> MGASPlan:
        """Compute per-stage (r_s, u_s) from the LogME profile.

        Args:
            profile:         StagewiseProfile from StagewiseLogMEProfiler
            embed_dim:       Backbone hidden dim D
            stage_dims:      Per-stage attention/MLP dim d_s (list of n_stages)
            stage_n_params:  Number of trainable params in each stage (for budget)
            t_lora:          Per-stage time for rank-1 LoRA training (seconds)
            t_full:          Per-stage time for full-FT one epoch (seconds)
            gamma:           Cost-model calibration constant (from D.2 fit)
        """
        cfg = self.config
        delta_q = profile.delta_q()    # Δq[s] = max(q) - q[s]

        if not profile.precondition_ok:
            logger.warning(
                "MGAS: precondition FAILED (all-identical scores or -inf features). "
                "Returning linear-probe schedule (all frozen, rank-0)."
            )
            return MGASPlan(
                ranks=[0] * cfg.n_stages,
                unfrozen=[False] * cfg.n_stages,
                suffix_cut=cfg.n_stages - 1,
                precondition_ok=False,
                budget_reduced=False,
                delta_q=delta_q,
            )

        # If no stage has meaningful gap from the best stage, the profile carries no
        # useful signal for rank/unfreeze decisions → fall back to uniform LoRA-8.
        # This prevents the degenerate plan (rank=0, unfreeze stages 1-3) that arises
        # when all Δq ≤ ε and find_suffix_cut returns s*=0.
        if max(delta_q) <= cfg.epsilon:
            fallback_rank = 8
            logger.warning(
                "MGAS: no useful signal (max Δq=%.4f ≤ ε=%.4f) — "
                "returning LoRA-8 fallback (rank=%d, all frozen).",
                max(delta_q), cfg.epsilon, fallback_rank,
            )
            return MGASPlan(
                ranks=[fallback_rank] * cfg.n_stages,
                unfrozen=[False] * cfg.n_stages,
                suffix_cut=0,
                precondition_ok=True,
                budget_reduced=False,
                delta_q=delta_q,
            )

        # Per-stage rank from staircase
        ranks   = [staircase_g(dq, cfg) for dq in delta_q]

        # Suffix cut s*
        s_star  = find_suffix_cut(delta_q, cfg.epsilon)

        # Unfreeze decision: stage s is unfrozen only when it's after s* AND its
        # transferability gap exceeds the unfreeze threshold.
        # Without the threshold, s*=0 (stage-0 best) unfreezes ALL remaining stages,
        # causing budget violations and unstable training on calibration cells where
        # Δq is uniformly small (0.01–0.14 for HLS/S2 cross-sensor pairs).
        unfrozen = [
            (s > s_star) and (delta_q[s] >= cfg.unfreeze_min_delta_q)
            for s in range(cfg.n_stages)
        ]

        # Budget reduction
        ranks, unfrozen, budget_reduced = reduce_budget(
            ranks, unfrozen, embed_dim, stage_dims, stage_n_params,
            t_lora, t_full, cfg.b_param, cfg.b_gpu, gamma, cfg.rank_steps,
            delta_q=delta_q,
        )

        p_frac = compute_param_cost(ranks, unfrozen, embed_dim, stage_dims, stage_n_params)
        t_frac = compute_time_cost(ranks, unfrozen, t_lora, t_full, gamma)

        plan = MGASPlan(
            ranks=ranks,
            unfrozen=unfrozen,
            suffix_cut=s_star,
            precondition_ok=True,
            budget_reduced=budget_reduced,
            delta_q=delta_q,
            param_fraction=p_frac,
            gpu_fraction=t_frac,
        )
        logger.info("MGAS plan: %s  (param=%.1f%%  gpu=%.1f%%)",
                    plan.schedule_str(), 100 * p_frac, 100 * t_frac)
        return plan

    def no_profile_plan(
        self,
        embed_dim: int,
        stage_dims: list[int],
        stage_n_params: list[int],
        t_lora: list[float],
        t_full: list[float],
        gamma: float = 1.0,
    ) -> MGASPlan:
        """MGAS-NO-PROFILE baseline: uniform Δq=0 → all stages get staircase(0)=0.
        Only the suffix-cut and budget rules apply (profile is ignored).
        """
        cfg = self.config
        delta_q = [0.0] * cfg.n_stages
        ranks   = [staircase_g(0.0, cfg)] * cfg.n_stages   # → 0 for all
        s_star  = find_suffix_cut(delta_q, cfg.epsilon)      # → 0 (all ≤ ε)
        # All stages "at or before" s* → all frozen + LoRA rank 0 → linear probe
        unfrozen = [False] * cfg.n_stages

        ranks, unfrozen, budget_reduced = reduce_budget(
            ranks, unfrozen, embed_dim, stage_dims, stage_n_params,
            t_lora, t_full, cfg.b_param, cfg.b_gpu, gamma, cfg.rank_steps,
            delta_q=delta_q,
        )
        return MGASPlan(
            ranks=ranks, unfrozen=unfrozen, suffix_cut=s_star,
            precondition_ok=True, budget_reduced=budget_reduced,
            delta_q=delta_q,
        )
