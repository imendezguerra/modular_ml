"""Pipeline entry points: single run, hyperparameter sweep, staged experiment.

These are the user-facing commands (also exposed as a Typer CLI). They wire the
config system to the :class:`~modular_ml.pipeline.runner.Runner` and to wandb
sweeps/experiments. wandb is required; sweeps use ``wandb.sweep`` + ``wandb.agent``.
"""

from pathlib import Path
from typing import Dict, List, Optional

import typer
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from modular_ml.configs.main_configs import instantiate_config, load_config, merge_overrides
from modular_ml.constants import CONFIG_DIR, PROJECT
from modular_ml.pipeline.experiment_tracking import (
    build_manifest,
    build_stage_entry,
    new_instance_id,
    sha256_file,
    write_manifest,
)
from modular_ml.pipeline.runner import Runner
from modular_ml.tools import logging as wlog
from modular_ml.tools.loaders import yaml_load

app = typer.Typer(add_completion=False, help="modular_ml pipeline commands.")


# --- helpers -----------------------------------------------------------------
def _save_sim_config(raw_cfg: DictConfig, runner: Runner) -> None:
    """Save the resolved config the run was built from and log it to wandb."""
    out_path = runner.output_dir / "sim_config.yaml"
    OmegaConf.save(config=raw_cfg, f=out_path)
    if runner.use_wandb:
        wlog.log_artifact(out_path, "config", "config")


# --- single run --------------------------------------------------------------
def run_pipeline(
    config_path: Path,
    use_wandb: bool = True,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Runner:
    """Train + test a single run config and log everything to wandb."""
    raw = load_config(config_path, instantiate=False)
    runner = Runner(instantiate_config(raw), use_wandb=use_wandb, tags=tags, notes=notes)
    _save_sim_config(raw, runner)
    runner.run_train()
    runner.run_test()
    runner.compute_objective()
    wlog.finish()
    return runner


# --- sweep -------------------------------------------------------------------
def run_sweep(
    config_path: Path,
    sweep_path: Path,
    count: int = 10,
    sweep_name: Optional[str] = None,
    project: str = PROJECT,
) -> str:
    """Launch a wandb sweep over a run config and return the sweep id."""
    sweep_cfg = yaml_load(sweep_path)
    if sweep_name:
        sweep_cfg["name"] = sweep_name
    sweep_id = wandb.sweep(sweep_cfg, project=project)

    base_cfg = load_config(config_path, instantiate=False, resolve=False)

    def _sweep_run() -> None:
        wandb.init(project=project)
        raw = merge_overrides(base_cfg, dict(wandb.config))
        runner = Runner(instantiate_config(raw), use_wandb=True)
        _save_sim_config(raw, runner)
        runner.run_train()
        runner.run_test()
        runner.compute_objective()
        wlog.finish()

    wandb.agent(sweep_id, function=_sweep_run, count=count)
    logger.info(f"Sweep complete: {sweep_id}")
    return sweep_id


# --- staged experiment -------------------------------------------------------
def _resolve_stage_config_path(config_path) -> Path:
    p = Path(config_path)
    return p if p.is_absolute() else CONFIG_DIR / "runs" / p


def run_experiment(experiment_path: Path) -> Dict:
    """Run a multi-stage experiment and log its provenance manifest to wandb.

    After each stage, the manifest so far is written to the stage's output dir
    and logged as an ``experiment`` artifact of that stage's run, so the last
    stage's run holds the complete manifest.
    """
    exp_cfg = load_config(experiment_path)
    stages = list(exp_cfg.runs.values())
    ids = [spec.id for spec in stages]
    for i, spec in enumerate(stages):
        if spec.load_params_from and spec.load_params_from not in ids[:i]:
            raise ValueError(
                f"Stage '{spec.id}': load_params_from '{spec.load_params_from}' "
                "is not an earlier stage."
            )
    instance_id = new_instance_id()
    stage_entries: List[Dict] = []
    prev_outputs: Dict[str, Path] = {}

    for i, spec in enumerate(stages):
        cfg_path = _resolve_stage_config_path(spec.config_path)
        raw = load_config(cfg_path, instantiate=False)
        runner = Runner(
            instantiate_config(raw),
            use_wandb=True,
            notes=spec.notes,
            tags=["experiment", instance_id, spec.id],
        )
        _save_sim_config(raw, runner)

        # Initialise weights from a previous stage or an external wandb run.
        if spec.load_params_from:
            ckpt = prev_outputs[spec.load_params_from] / "model_trained.pt"
            runner.load_model(ckpt, weights_only_state=True)
            logger.info(f"Stage '{spec.id}' initialised from '{spec.load_params_from}'.")
        elif spec.init_from_wandb_run:
            filename = spec.init_from_wandb_artifact_filename or "model_trained.pt"
            art_dir = wlog.download_artifact(
                spec.init_from_wandb_run,
                filename.replace(".pt", ""),
                "model",
                spec.init_from_wandb_version or "latest",
            )
            runner.load_model(Path(art_dir) / filename, weights_only_state=True)

        mode = getattr(spec.run_mode, "value", spec.run_mode)
        if mode in ("train", "train_test"):
            runner.run_train()
        if mode in ("test", "train_test"):
            runner.run_test()
        runner.compute_objective()

        stage_entries.append(
            build_stage_entry(
                spec, i, len(stages), runner.run_id, runner.run_name, sha256_file(cfg_path)
            )
        )
        prev_outputs[spec.id] = runner.output_dir
        manifest = build_manifest(Path(experiment_path), instance_id, stage_entries)
        manifest_path = write_manifest(manifest, runner.output_dir)
        wlog.log_artifact(manifest_path, "experiment", "manifest", metadata=manifest)
        wlog.finish()

    logger.info(f"Experiment manifest written to {manifest_path}")
    return manifest


# --- Typer CLI ---------------------------------------------------------------
@app.command()
def train(config: Path, notes: str = typer.Option("", help="Run notes")):
    """Train + test a single run config."""
    run_pipeline(config, notes=notes or None)


@app.command()
def sweep(
    config: Path,
    sweep: Path,
    count: int = 10,
    name: str = typer.Option("", help="Sweep name"),
):
    """Launch a wandb sweep over a run config."""
    run_sweep(config, sweep, count=count, sweep_name=name or None)


@app.command()
def experiment(config: Path):
    """Run a multi-stage experiment."""
    run_experiment(config)


if __name__ == "__main__":
    app()
