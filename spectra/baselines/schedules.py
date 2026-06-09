"""Fixed adaptation schedule baselines for SPECTRA comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    """Full FT: all stages unfrozen, no LoRA."""
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
    return make_lora_schedule(8, n_stages)


def lora16_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(16, n_stages)


def lora32_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(32, n_stages)


def lora64_schedule(n_stages: int = 4) -> BaselineSchedule:
    return make_lora_schedule(64, n_stages)


def last_stage_full_ft_schedule(n_stages: int = 4) -> BaselineSchedule:
    """Last-stage full fine-tuning with earlier stages frozen."""
    ranks = [0] * n_stages
    unfrozen = [False] * n_stages
    unfrozen[-1] = True
    return BaselineSchedule("LastStageFT", ranks, unfrozen)


def surgical_ft_schedule(
    logme_scores: Optional[list[float]] = None,
    n_stages: int = 4,
    top_k: int = 1,
) -> BaselineSchedule:
    """Unfreeze the highest-scoring stages, defaulting to the last stage."""
    unfrozen = [False] * n_stages
    if logme_scores is not None:
        top_stages = sorted(range(n_stages), key=lambda s: logme_scores[s], reverse=True)[:top_k]
        for s in top_stages:
            unfrozen[s] = True
    else:
        unfrozen[-1] = True
    return BaselineSchedule("SurgicalFT", [0] * n_stages, unfrozen)


ALL_SCHEDULES = {
    "lp": linear_probe_schedule,
    "full_ft": full_ft_schedule,
    "lora1": lora1_schedule,
    "lora2": lora2_schedule,
    "lora4": lora4_schedule,
    "lora8": lora8_schedule,
    "lora16": lora16_schedule,
    "lora32": lora32_schedule,
    "lora64": lora64_schedule,
    "last_stage": last_stage_full_ft_schedule,
    "surgical": surgical_ft_schedule,
}
