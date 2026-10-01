"""Root config dataclass and the Hydra loader.

``load_config`` is the single entry point that turns a YAML run file into a
fully-typed :class:`Config` object: it composes config groups (via the
``defaults:`` list), resolves interpolations/resolvers, then instantiates every
``_target_`` into its dataclass. Pass ``instantiate=False`` to keep a raw
``DictConfig`` (used when merging sweep overrides before instantiation).
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

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
) -> Union[Config, DictConfig]:
    """Compose, resolve and instantiate a run config from a YAML file path."""
    path = Path(path).resolve()

    # Absolute config dir: a path relative to this file breaks on Windows when the
    # config lives on another drive (os.path.relpath cannot cross drives).
    with hydra.initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = hydra.compose(config_name=path.stem)

    if resolve:
        OmegaConf.resolve(cfg)
    if instantiate:
        return hydra.utils.instantiate(cfg, _convert_="all")
    return cfg


def merge_overrides(base_cfg: DictConfig, overrides: Union[List[str], Dict]) -> Config:
    """Merge dotlist/dict overrides into a composed config, then instantiate.

    Used by sweeps: the wandb agent supplies overrides as a flat mapping of
    dotted keys, which are merged before resolution + instantiation.
    """
    cfg = deepcopy(base_cfg)
    if isinstance(overrides, list):
        override_cfg = OmegaConf.from_dotlist(overrides)
    else:
        override_cfg = OmegaConf.create(overrides)
    merged = OmegaConf.merge(cfg, override_cfg)
    OmegaConf.resolve(merged)
    return hydra.utils.instantiate(merged, _convert_="all")
