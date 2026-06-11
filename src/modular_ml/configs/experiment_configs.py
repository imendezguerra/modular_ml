"""Multi-stage experiment config dataclasses.

An experiment is an ordered dict of :class:`RunSpec` stages. Each stage points
at a run config and may initialise its weights from a previous stage's wandb
checkpoint, enabling pretrain -> finetune pipelines with full provenance.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from modular_ml.configs import enums


@dataclass
class RunSpec:
    """One stage of an experiment."""

    _target_: str = "modular_ml.configs.experiment_configs.RunSpec"
    id: str = "stage_0"
    config_path: Union[str, Path] = ""
    # Initialise weights from a previous stage (by run-spec id) within this run.
    load_params_from: Optional[str] = None
    # Or initialise from an existing wandb run's checkpoint artifact.
    init_from_wandb_run: Optional[str] = None
    init_from_wandb_artifact_filename: Optional[str] = None
    init_from_wandb_version: Optional[str] = None
    init_load_mode: enums.InitLoadMode = enums.InitLoadMode.weights_only
    run_mode: enums.RunMode = enums.RunMode.train_test
    notes: Optional[str] = None


@dataclass
class ExperimentConfig:
    """An ordered collection of run stages."""

    _target_: str = "modular_ml.configs.experiment_configs.ExperimentConfig"
    runs: Dict[str, RunSpec] = field(default_factory=dict)
