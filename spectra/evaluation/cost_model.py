"""D.2 Joint cost-model γ fit.

Fits γ such that predicted GPU-h ≈ measured GPU-h across 8 calibration cells.
Gate G1: R² ≥ 0.90 required; otherwise declare as limitation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


@dataclass
class CostModelFit:
    gamma: float
    r2_time: float
    r2_param: float
    gate_g1_passed: bool
    n_cells: int
    residuals: list[float]


@dataclass
class TimingRecord:
    """One timing measurement for a (backbone, stage) pair."""
    backbone: str
    stage: int
    t_lora_per_step: float    # seconds per gradient step with rank-1 LoRA
    t_full_per_step: float    # seconds per gradient step with full FT
    rank_used: int = 1        # LoRA rank used during timing


class CostModel:
    """Calibrates γ from timing passes and validates with R² metric.

    Usage::
        cm = CostModel()
        cm.add_timing(TimingRecord(...))
        fit = cm.fit(gate_threshold=0.90)
    """

    def __init__(self) -> None:
        self._timings: list[TimingRecord] = []
        self._measured_gpu_h: list[float] = []
        self._predicted_base: list[float] = []

    def add_timing(self, record: TimingRecord) -> None:
        self._timings.append(record)

    def add_cell_observation(
        self,
        backbone: str,
        ranks: list[int],
        unfrozen: list[bool],
        n_steps: int,
        measured_gpu_h: float,
    ) -> None:
        """Add one (backbone, cell) measurement for γ fitting.

        Args:
            backbone:       Backbone name (must match TimingRecord.backbone)
            ranks:          Per-stage LoRA rank used
            unfrozen:       Per-stage unfreeze flag
            n_steps:        Total training steps
            measured_gpu_h: Actual wall-clock GPU-h for this run
        """
        timings = {(t.backbone, t.stage): t for t in self._timings}
        predicted_base = 0.0
        for s, (r, u) in enumerate(zip(ranks, unfrozen)):
            key = (backbone, s)
            if key not in timings:
                logger.warning("No timing for %s stage %d — skipping cell", backbone, s)
                return
            t = timings[key]
            predicted_base += r * t.t_lora_per_step + int(u) * t.t_full_per_step
        predicted_base *= n_steps / 3600   # steps → hours

        self._measured_gpu_h.append(measured_gpu_h)
        self._predicted_base.append(predicted_base)

    def fit(self, gate_threshold: float = 0.90) -> CostModelFit:
        """Fit γ by minimizing MSE between γ·predicted_base and measured GPU-h."""
        if len(self._measured_gpu_h) < 2:
            raise RuntimeError("Need ≥ 2 cell observations to fit γ.")

        y_true = np.array(self._measured_gpu_h)
        x      = np.array(self._predicted_base)

        # Closed-form: γ* = (x·y_true) / (x·x)
        gamma = float(np.dot(x, y_true) / np.dot(x, x))
        y_pred = gamma * x
        r2 = float(r2_score(y_true, y_pred))
        residuals = (y_true - y_pred).tolist()

        fit = CostModelFit(
            gamma=gamma,
            r2_time=r2,
            r2_param=float("nan"),   # parameter count is deterministic; no fit needed
            gate_g1_passed=r2 >= gate_threshold,
            n_cells=len(y_true),
            residuals=residuals,
        )
        if fit.gate_g1_passed:
            logger.info("Gate G1 PASSED: R²=%.4f ≥ %.2f  γ=%.4f", r2, gate_threshold, gamma)
        else:
            logger.warning(
                "Gate G1 FAILED: R²=%.4f < %.2f  γ=%.4f. "
                "Declaring cost-model accuracy as a limitation.",
                r2, gate_threshold, gamma,
            )
        return fit

    def save(self, path: Path, fit: CostModelFit) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "gamma": fit.gamma,
            "r2_time": fit.r2_time,
            "gate_g1_passed": fit.gate_g1_passed,
            "n_cells": fit.n_cells,
            "residuals": fit.residuals,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Cost-model fit saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> tuple["CostModel", CostModelFit]:
        data = json.loads(path.read_text())
        cm = cls()
        fit = CostModelFit(
            gamma=data["gamma"],
            r2_time=data["r2_time"],
            r2_param=float("nan"),
            gate_g1_passed=data["gate_g1_passed"],
            n_cells=data["n_cells"],
            residuals=data["residuals"],
        )
        return cm, fit
