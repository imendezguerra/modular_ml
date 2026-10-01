"""Load artifacts (config, outputs, model) back from a finished wandb run.

Used by notebooks/analysis after training. Looks in the local OUTPUT_DIR first,
then falls back to downloading the run's wandb artifacts.
"""

from pathlib import Path
from typing import Dict

import torch

from modular_ml.constants import OUTPUT_DIR
from modular_ml.tools.loaders import h5_load, yaml_load
from modular_ml.tools.logging import download_artifact


def _local_or_download(run_name: str, filename: str, suffix: str, artifact_type: str) -> Path:
    local = OUTPUT_DIR / run_name / filename
    if local.exists():
        return local
    art_dir = download_artifact(run_name, suffix, artifact_type)
    return Path(art_dir) / filename


def load_config_from_run(run_name: str) -> Dict:
    """Load the resolved ``sim_config.yaml`` saved for a run."""
    return yaml_load(_local_or_download(run_name, "sim_config.yaml", "config", "config"))


def load_outputs_from_run(run_name: str) -> Dict:
    """Load the saved ``test_outputs.h5`` for a run."""
    return h5_load(_local_or_download(run_name, "test_outputs.h5", "outputs", "outputs"))


def load_model_params_from_run(
    run_name: str, filename: str = "model_trained.pt", device: str = "cpu"
) -> Dict:
    """Load a saved checkpoint dict for a run."""
    path = _local_or_download(run_name, filename, "model", "model")
    return torch.load(path, map_location=device, weights_only=False)
