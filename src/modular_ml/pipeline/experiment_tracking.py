"""Provenance for multi-stage experiments.

Builds a resolved execution manifest (which stages ran, their config hashes,
wandb run ids and checkpoint sources) and writes it to disk so an experiment is
fully reproducible. The pipeline logs it to wandb as an ``experiment`` artifact
after every stage.
"""

import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List

from modular_ml.configs.experiment_configs import RunSpec
from modular_ml.tools.loaders import yaml_save


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def new_instance_id() -> str:
    """Return a short unique id for one experiment invocation."""
    return uuid.uuid4().hex[:12]


def build_stage_entry(
    run_spec: RunSpec,
    run_index: int,
    run_count: int,
    wandb_run_id: str,
    wandb_run_name: str,
    config_sha256: str,
) -> Dict[str, Any]:
    """Describe one executed stage for the manifest.

    Args:
        run_spec: The stage's ``RunSpec``.
        run_index: Zero-based stage index.
        run_count: Number of stages.
        wandb_run_id: Unique id of the stage's wandb run.
        wandb_run_name: Display name of the stage's wandb run.
        config_sha256: Hash of the stage's run config file.
    """
    return {
        "run_spec_id": run_spec.id,
        "run_spec_index": run_index + 1,
        "run_spec_count": run_count,
        "config_path": str(run_spec.config_path),
        "config_sha256": config_sha256,
        "run_mode": getattr(run_spec.run_mode, "value", run_spec.run_mode),
        "load_params_from": run_spec.load_params_from,
        "init_from_wandb_run": run_spec.init_from_wandb_run,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
        "notes": run_spec.notes,
    }


def build_manifest(
    experiment_path: Path,
    experiment_instance_id: str,
    stage_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the full experiment manifest."""
    return {
        "experiment": {
            "instance_id": experiment_instance_id,
            "config_name": Path(experiment_path).name,
            "config_sha256": sha256_file(experiment_path),
        },
        "stages": stage_entries,
    }


def write_manifest(manifest: Dict[str, Any], output_dir: Path) -> Path:
    """Write ``experiment_manifest.yaml`` to ``output_dir`` and return its path."""
    path = Path(output_dir) / "experiment_manifest.yaml"
    yaml_save(manifest, path)
    return path
