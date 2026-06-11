"""Model / architecture config dataclasses.

A model is a list of blocks. Each :class:`BlockSpec` is a Hydra-instantiable
description of one block: ``_block_target_`` names the concrete ``Block``
subclass, ``params`` are its constructor kwargs, ``init`` holds per-parameter
init specs, and ``freeze`` lists parameters to freeze. The network factory
(:func:`modular_ml.models.network.build_modular_model`) turns these specs into a
``ModularNet``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modular_ml.configs import enums


@dataclass
class BlockSpec:
    """Hydra-instantiable description of a single network block."""

    _target_: str = "modular_ml.configs.model_configs.BlockSpec"
    _block_target_: str = "modular_ml.models.blocks.Block"

    id: str = "block_0"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)
    in_dim: int = 0
    out_dim: int = 0

    # Extra constructor kwargs passed straight to the block.
    params: Dict[str, Any] = field(default_factory=dict)
    # Per-parameter initialisation overrides ({param_name: InitSpec}).
    init: Dict[str, Any] = field(default_factory=dict)
    # Parameters/submodules to freeze after construction.
    freeze: List[str] = field(default_factory=list)


@dataclass
class RuntimeScheduleConfig:
    """Epoch-driven architecture transitions (freeze/unfreeze/input modes).

    ``milestones`` and ``states`` are parallel lists: when ``epoch`` reaches
    ``milestones[i]`` the matching ``states[i]`` mapping is applied to the model.
    """

    _target_: str = "modular_ml.configs.model_configs.RuntimeScheduleConfig"
    milestones: List[int] = field(default_factory=list)
    states: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ArchitectureConfig:
    """A composable architecture: an ordered dict of block specs.

    ``modules`` is a dict (not a list) so wandb sweeps can address individual
    block parameters by name. ``global_params`` is a scratch space for shared
    YAML values referenced via interpolation.
    """

    _target_: str = "modular_ml.configs.model_configs.ArchitectureConfig"
    modules: Dict[str, BlockSpec] = field(default_factory=dict)
    global_params: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[RuntimeScheduleConfig] = None


@dataclass
class ModelConfig:
    """Top-level model configuration."""

    _target_: str = "modular_ml.configs.model_configs.ModelConfig"

    name: str = "my_model"
    n_inputs: int = 4
    n_outputs: int = 3
    in_key: str = "stim"
    out_key: str = "prediction"
    dtype: enums.DType = enums.DType.float32

    # Leaky-integrator time constant for recurrent blocks. alpha = dt/tau.
    tau_s: float = 0.05
    dt_s: float = 1.0e-2
    alpha: float = 0.0

    batch_size: int = 64
    epochs: int = 200
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    def __post_init__(self):
        if self.alpha == 0.0:
            self.alpha = self.dt_s / self.tau_s
