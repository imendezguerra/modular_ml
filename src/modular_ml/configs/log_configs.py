"""Logging / sweep-objective config dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MetricSpec:
    """One metric contributing to the sweep objective.

    ``weight`` is signed (use negative for lower-is-better terms).
    ``epoch_reduction`` picks how to reduce the metric's history before
    combining: one of ``last``, ``max``, ``min``, ``mean``.
    """

    _target_: str = "modular_ml.configs.log_configs.MetricSpec"
    name: str = ""
    weight: float = 1.0
    epoch_reduction: str = "last"
    missing_value: float = 0.0


@dataclass
class ObjectiveConfig:
    """How to combine metrics into a single scalar sweep objective."""

    _target_: str = "modular_ml.configs.log_configs.ObjectiveConfig"
    # 'sum', 'mean', or 'weighted_mean'.
    combination: str = "weighted_mean"
    metrics: List[MetricSpec] = field(default_factory=list)


@dataclass
class LoggerConfig:
    """How runs are scored: the sweep objective, built from logged metrics."""

    _target_: str = "modular_ml.configs.log_configs.LoggerConfig"
    objective: Optional[ObjectiveConfig] = None
