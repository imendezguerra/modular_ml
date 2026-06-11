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
    """What to log to wandb.

    ``wandb_groups`` selects metric prefixes to forward (e.g. ``train``,
    ``test``, ``pca``); an empty list forwards everything. ``activation_keys``
    are data-dict keys to plot as diagnostic time series.
    """

    _target_: str = "modular_ml.configs.log_configs.LoggerConfig"
    wandb_groups: List[str] = field(default_factory=list)
    activation_keys: List[str] = field(default_factory=list)
    objective: Optional[ObjectiveConfig] = None
