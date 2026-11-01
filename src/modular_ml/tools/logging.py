"""Thin wrappers around the Weights & Biases API.

The pipeline always logs to wandb (the template requires a wandb account). These
helpers centralise run init, metric/figure/artifact logging, and artifact
download so the rest of the code stays terse.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import wandb
from loguru import logger

from modular_ml.constants import PROJECT, WANDB_USER

PathLike = Union[str, Path]


def init_wandb_run(config: Dict[str, Any], project: str = PROJECT, **kwargs) -> None:
    """Initialise a wandb run, or add the full config to the active one.

    A sweep agent starts the run itself with only the swept parameters, so the
    full resolved config is added to it here.
    """
    if wandb.run is None:
        wandb.init(project=project, config=config, **kwargs)
    else:
        wandb.config.update(config, allow_val_change=True)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a flat dict of scalar metrics to the active run."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_figure(name: str, figure, step: Optional[int] = None) -> None:
    """Log a matplotlib figure under ``name``."""
    if wandb.run is not None:
        wandb.log({name: wandb.Image(figure)}, step=step)


def log_artifact(
    path_to_file: PathLike,
    artifact_type: str,
    artifact_suffix: str,
    metadata: Optional[dict] = None,
) -> None:
    """Upload a file as a wandb artifact named ``<run id>-<suffix>``."""
    if wandb.run is None:
        logger.error("Artifact not logged: no active wandb run.")
        return
    artifact = wandb.Artifact(
        name=f"{wandb.run.id}-{artifact_suffix}", type=artifact_type, metadata=metadata
    )
    artifact.add_file(str(path_to_file))
    wandb.run.log_artifact(artifact)
    logger.debug(f"Artifact {artifact.name} logged.")


def _artifact_ref(run_id: str, suffix: str, version: str = "latest") -> str:
    project_path = f"{PROJECT}" if WANDB_USER is None else f"{WANDB_USER}/{PROJECT}"
    return f"{project_path}/{run_id}-{suffix}:{version}"


def download_artifact(
    run_id: str,
    artifact_suffix: str,
    artifact_type: str,
    version: str = "latest",
) -> Path:
    """Download an artifact (by run id + suffix) and return its local dir."""
    ref = _artifact_ref(run_id, artifact_suffix, version)
    if wandb.run is None:
        artifact = wandb.Api().artifact(ref, type=artifact_type)
    else:
        artifact = wandb.run.use_artifact(ref, type=artifact_type)
    return Path(artifact.download())


def finish() -> None:
    """Finish the active wandb run, if any."""
    if wandb.run is not None:
        wandb.finish()
