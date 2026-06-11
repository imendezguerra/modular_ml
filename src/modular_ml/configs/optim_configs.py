"""Optimizer and modular-loss config dataclasses.

The total training objective is a weighted sum of :class:`LossSpec` terms. Each
term names a concrete ``LossModule`` via ``_loss_target_``, a ``criterion``,
the ``in_keys`` it reads from the data dict, and an optional ``label_key`` for
supervised terms. ``schedule`` ramps the term's weight over epochs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modular_ml.configs import enums


@dataclass
class LossSpec:
    """Hydra-instantiable description of a single loss term."""

    _target_: str = "modular_ml.configs.optim_configs.LossSpec"
    _loss_target_: str = "modular_ml.losses.loss.LossModule"

    id: str = "main_loss"
    criterion: enums.LossCriterion = enums.LossCriterion.mse
    in_keys: List[str] = field(default_factory=list)
    label_key: Optional[str] = None
    requires_model: bool = False
    reg_weight: float = 1.0
    # On/off flag kept separate from reg_weight so sweeps can toggle a term
    # without needing 0.0 as a sweepable value.
    use_reg: float = 1.0

    params: Dict[str, Any] = field(default_factory=dict)
    # Optional epoch schedule for this term's weight (train only by default).
    schedule: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Optimizer settings plus the dict of loss terms."""

    _target_: str = "modular_ml.configs.optim_configs.OptimizerConfig"
    optim: enums.OptimizerType = enums.OptimizerType.Adam
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    clip_grad: float = 0.0  # 0 disables gradient clipping

    losses: Dict[str, LossSpec] = field(default_factory=dict)
