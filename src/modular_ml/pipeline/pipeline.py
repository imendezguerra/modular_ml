"""Pipeline entry points: single run, hyperparameter sweep, staged experiment.

These are the user-facing commands (also exposed as a Typer CLI). They wire the
config system to the :class:`~modular_ml.pipeline.runner.Runner` and to wandb
sweeps/experiments. wandb is required; sweeps use ``wandb.sweep`` + ``wandb.agent``.
"""

from pathlib import Path
from typing import Dict, List, Optional

import hydra
import typer
import wandb
from loguru import logger
from omegaconf import OmegaConf

from modular_ml.configs.main_configs import Config, load_config
from modular_ml.constants import CONFIG_DIR, OUTPUT_DIR, PROJECT
from modular_ml.pipeline.experiment_tracking import (
    build_manifest,
    build_stage_entry,
    new_instance_id,
    write_manifest_files,
)
from modular_ml.pipeline.runner import Runner
from modular_ml.tools import logging as wlog
from modular_ml.tools.loaders import yaml_load

app = typer.Typer(add_completion=False, help="modular_ml pipeline commands.")


# --- helpers -----------------------------------------------------------------
def _save_sim_config(config_path: Path, runner: Runner) -> None:
    """Save the resolved run config next to the outputs and log it to wandb."""
    raw = load_config(config_path, instantiate=False, resolve=True)
    out_path = runner.output_dir / "sim_config.yaml"
    OmegaConf.save(config=raw, f=out_path)
    if runner.use_wandb:
        wlog.log_artifact(out_path, "config", "config")


def _dotlist(flat: Dict) -> List[str]:
    """Convert a flat {dotted_key: value} mapping to a dotlist for OmegaConf."""
    return [f"{k}={v}" for k, v in flat.items()]


# --- single run --------------------------------------------------------------
def run_pipeline(
    config_path: Path,
    use_wandb: bool = True,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Runner:
    """Train + test a single run config and log everything to wandb."""
    config: Config = load_config(config_path)
    runner = Runner(config, use_wandb=use_wandb, tags=tags, notes=notes)
    _save_sim_config(Path(config_path), runner)
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
        overrides = OmegaConf.from_dotlist(_dotlist(dict(wandb.config)))
        merged = OmegaConf.merge(base_cfg, overrides)
        OmegaConf.resolve(merged)
        config = hydra.utils.instantiate(merged, _convert_="all")
        runner = Runner(config, use_wandb=True, is_sweep=True)
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
    """Run a multi-stage experiment and write a provenance manifest."""
    exp_cfg = load_config(experiment_path)
    stages = list(exp_cfg.runs.values())
    instance_id = new_instance_id()
    stage_entries: List[Dict] = []
    prev_outputs: Dict[str, Path] = {}

    for i, spec in enumerate(stages):
        cfg_path = _resolve_stage_config_path(spec.config_path)
        config = load_config(cfg_path)
        runner = Runner(
            config, use_wandb=True, notes=spec.notes,
            tags=["experiment", instance_id, spec.id],
        )
        _save_sim_config(cfg_path, runner)

        # Initialise weights from a previous stage or an external wandb run.
        if spec.load_params_from and spec.load_params_from in prev_outputs:
            ckpt = prev_outputs[spec.load_params_from] / "model_trained.pt"
            runner.load_model(ckpt, weights_only_state=True)
            logger.info(f"Stage '{spec.id}' initialised from '{spec.load_params_from}'.")
        elif spec.init_from_wandb_run:
            filename = spec.init_from_wandb_artifact_filename or "model_trained.pt"
            art_dir = wlog.download_artifact(
                spec.init_from_wandb_run, filename.replace(".pt", ""), "model",
                spec.init_from_wandb_version or "latest",
            )
            runner.load_model(Path(art_dir) / filename, weights_only_state=True)

        mode = getattr(spec.run_mode, "value", spec.run_mode)
        if mode in ("train", "train_test", "retrain", "retrain_test"):
            runner.run_train()
        if mode in ("test", "train_test", "retrain_test"):
            runner.run_test()
        runner.compute_objective()

        stage_entries.append(build_stage_entry(spec, i, len(stages), runner.run_name))
        prev_outputs[spec.id] = runner.output_dir
        wlog.finish()

    manifest = build_manifest(Path(experiment_path), instance_id, stage_entries)
    out_dir = OUTPUT_DIR / f"experiment_{instance_id}"
    paths = write_manifest_files(manifest, out_dir)
    logger.info(f"Experiment manifest written to {paths['manifest']}")
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
