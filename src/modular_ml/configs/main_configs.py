"""Root config dataclass and the Hydra loader.

``load_config`` is the single entry point that turns a YAML run file into a
fully-typed :class:`Config` object: it composes config groups (via the
``defaults:`` list), resolves interpolations/resolvers, then instantiates every
``_target_`` into its dataclass. Pass ``instantiate=False`` to keep the
resolved ``DictConfig``: the pipeline saves it as the run's ``sim_config.yaml``
and passes it to :func:`instantiate_config`. Sweeps apply their parameters with
:func:`merge_overrides` first.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf

import modular_ml.tools.hydra_utils  # noqa: F401  (registers resolvers)
from modular_ml.configs.dataset_configs import DatasetConfig
from modular_ml.configs.log_configs import LoggerConfig
from modular_ml.configs.model_configs import ModelConfig
from modular_ml.configs.optim_configs import OptimizerConfig
from modular_ml.configs.processing_configs import ProcessingConfig


@dataclass
class Config:
    """Top-level configuration tying every subsystem together."""

    _target_: str = "modular_ml.configs.main_configs.Config"
    seed: int = 0
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


def load_config(
    path: Union[str, Path],
    instantiate: bool = True,
    resolve: bool = True,
    overrides: Optional[List[str]] = None,
) -> Union[Config, DictConfig]:
    """Compose, resolve and instantiate a run config from a YAML file path.

    Args:
        path: Run config YAML.
        instantiate: Return a typed ``Config`` instead of a ``DictConfig``.
        resolve: Resolve interpolations.
        overrides: Hydra overrides, e.g. ``["model=model_gru", "seed=1"]``.
    """
    path = Path(path).resolve()

    # Absolute config dir: a path relative to this file breaks on Windows when the
    # config lives on another drive (os.path.relpath cannot cross drives).
    with hydra.initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = hydra.compose(config_name=path.stem, overrides=overrides or [])

    if resolve:
        OmegaConf.resolve(cfg)
    return instantiate_config(cfg) if instantiate else cfg


def instantiate_config(cfg: DictConfig) -> Any:
    """Instantiate every ``_target_`` of a composed config into its dataclass."""
    return hydra.utils.instantiate(cfg, _convert_="all")


def merge_overrides(base_cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    """Apply flat dotted-key overrides to an unresolved config, then resolve it.

    Used by sweeps: the wandb agent supplies parameters as ``{"optim.lr": 1e-3}``.
    They are applied before resolution so interpolations pick them up. Unknown
    keys raise, so a typo in a sweep file fails loudly.

    Args:
        base_cfg: Composed, unresolved config (``load_config(..., resolve=False)``).
        overrides: ``{dotted.key: value}`` mapping.
    """
    cfg = deepcopy(base_cfg)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    OmegaConf.resolve(cfg)
    return cfg
