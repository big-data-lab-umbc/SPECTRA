"""Fixed adaptation schedule baselines for comparison with MGAS.

Each function returns (ranks: list[int], unfrozen: list[bool]) for 4 stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BaselineSchedule:
    name: str
    ranks: list[int]
    unfrozen: list[bool]

    def schedule_str(self) -> str:
        return " ".join(f"(r={r},u={int(u)})" for r, u in zip(self.ranks, self.unfrozen))


def linear_probe_schedule(n_stages: int = 4) -> BaselineSchedule:
    """LP: all stages frozen, no LoRA."""
    return BaselineSchedule("LP", [0] * n_stages, [False] * n_stages)


def full_ft_schedule(n_stages: int = 4) -> BaselineSchedule:
    """Full FT: all stages unfrozen, no LoRA (backbone fully updated)."""
    return BaselineSchedule("FullFT", [0] * n_stages, [True] * n_stages)


def make_lora_schedule(rank: int, n_stages: int = 4) -> BaselineSchedule:
    """Uniform LoRA of given rank across all stages, backbone frozen."""
    return BaselineSchedule(f"LoRA-{rank}", [rank] * n_stages, [False] * n_stages)


def lora1_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(1, n_stages)


def lora2_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(2, n_stages)


def lora4_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(4, n_stages)


def lora8_schedule(n_stages: int = 4) -> BaselineSchedule:
    """All-stages-LoRA-8: uniform rank-8 LoRA, all stages frozen."""
    return make_lora_schedule(8, n_stages)


def lora16_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(16, n_stages)


def lora32_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(32, n_stages)


def lora64_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(64, n_stages)


def last_stage_full_ft_schedule(n_stages: int = 4) -> BaselineSchedule:
    """Last-stage-full-FT: last stage unfrozen, remaining frozen with no LoRA."""
    ranks    = [0] * n_stages
    unfrozen = [False] * n_stages
    unfrozen[-1] = True
    return BaselineSchedule("LastStageFT", ranks, unfrozen)


def deflect_last2_schedule(n_stages: int = 4) -> BaselineSchedule:
    """DEFLECT + last-2 FT: last 2 stages unfrozen, earlier stages frozen."""
    ranks    = [0] * n_stages
    unfrozen = [False] * n_stages
    for s in range(max(0, n_stages - 2), n_stages):
        unfrozen[s] = True
    return BaselineSchedule("DEFLECT+last2", ranks, unfrozen)


def surgical_ft_schedule(
    logme_scores: Optional[list[float]] = None,
    n_stages: int = 4,
    top_k: int = 1,
) -> BaselineSchedule:
    """Surgical FT: unfreeze top-k stages by LogME score (highest score = most relevant).

    When logme_scores is None, defaults to unfreezing the last stage (same as LastStageFT).
    """
    unfrozen = [False] * n_stages
    if logme_scores is not None:
        top_stages = sorted(range(n_stages), key=lambda s: logme_scores[s], reverse=True)[:top_k]
        for s in top_stages:
            unfrozen[s] = True
    else:
        unfrozen[-1] = True
    return BaselineSchedule("SurgicalFT", [0] * n_stages, unfrozen)


def mgas_no_profile_schedule(
    embed_dim: int,
    stage_dims: list[int],
    stage_n_params: list[int],
    t_lora: list[float],
    t_full: list[float],
    gamma: float = 1.0,
    b_param: float = 0.50,
    b_gpu: float = 0.50,
    n_stages: int = 4,
) -> BaselineSchedule:
    """MGAS-NO-PROFILE: apply budget rules without a transferability profile.

    Uses the same joint budget constraint as MGAS but with no LogME profiling.
    Default: last stage unfrozen (uniform fallback), then reduce ranks if over budget.
    Pre-registered C3 counterfactual.
    """
    from spectra.planner.mgas import MGASPlanner, MGASConfig
    from spectra.planner.logme_profiler import StagewiseProfile
    import logging

    planner = MGASPlanner(MGASConfig(b_param=b_param, b_gpu=b_gpu, n_stages=n_stages))
    plan = planner.no_profile_plan(
        embed_dim=embed_dim,
        stage_dims=stage_dims,
        stage_n_params=stage_n_params,
        t_lora=t_lora,
        t_full=t_full,
        gamma=gamma,
    )
    return BaselineSchedule("MGAS-NO-PROFILE", plan.ranks, plan.unfrozen)


ALL_SCHEDULES = {
    "lp":          linear_probe_schedule,
    "full_ft":     full_ft_schedule,
    "lora1":       lora1_schedule,
    "lora2":       lora2_schedule,
    "lora4":       lora4_schedule,
    "lora8":       lora8_schedule,
    "lora16":      lora16_schedule,
    "lora32":      lora32_schedule,
    "lora64":      lora64_schedule,
    "last_stage":  last_stage_full_ft_schedule,
    "deflect_last2": deflect_last2_schedule,
}
