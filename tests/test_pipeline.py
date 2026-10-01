"""End-to-end pipeline tests with wandb disabled (no account needed)."""

from pathlib import Path

import pytest

from modular_ml.configs.dataset_configs import DatasetConfig
from modular_ml.configs.log_configs import LoggerConfig, MetricSpec, ObjectiveConfig
from modular_ml.configs.main_configs import Config
from modular_ml.configs.model_configs import ArchitectureConfig, BlockSpec, ModelConfig
from modular_ml.configs.optim_configs import LossSpec, OptimizerConfig
from modular_ml.configs.processing_configs import ProcessingConfig
from modular_ml.pipeline.pipeline import run_experiment
from modular_ml.pipeline.runner import Runner
from modular_ml.tools.loaders import yaml_save


def _sine_config(data_dir: Path, epochs: int = 4) -> Config:
    arch = ArchitectureConfig(
        modules={
            "rnn": BlockSpec(
                _block_target_="modular_ml.models.blocks.RNNBlock",
                id="rnn",
                in_keys=["stim"],
                out_keys=["activity"],
                in_dim=2,
                out_dim=16,
                params={"alpha": 0.2, "activation": "tanh"},
            ),
            "readout": BlockSpec(
                _block_target_="modular_ml.models.blocks.LinearBlock",
                id="readout",
                in_keys=["rnn.activity"],
                out_keys=["prediction"],
                in_dim=16,
                out_dim=1,
            ),
        }
    )
    return Config(
        seed=0,
        dataset=DatasetConfig(name="sine", data_dir=str(data_dir)),
        model=ModelConfig(
            n_inputs=2,
            n_outputs=1,
            in_key="stim",
            architecture=arch,
            alpha=0.2,
            epochs=epochs,
            batch_size=16,
        ),
        optim=OptimizerConfig(
            losses={
                "mse": LossSpec(
                    id="mse", criterion="mse", in_keys=["readout.prediction"], label_key="target"
                ),
                "smooth": LossSpec(
                    id="smooth",
                    criterion="temporal_smoothness",
                    in_keys=["rnn.activity"],
                    reg_weight=0.01,
                ),
            }
        ),
        processing=ProcessingConfig(run_pca=True, pca_keys=["rnn.activity"]),
        logger=LoggerConfig(
            objective=ObjectiveConfig(
                metrics=[MetricSpec(name="test/corrcoef", weight=1.0, epoch_reduction="last")]
            )
        ),
    )


def test_runner_sine_train_test(sine_dir):
    runner = Runner(_sine_config(sine_dir), use_wandb=True)
    runner.run_train()
    metrics = runner.run_test()
    assert "test/total_loss" in metrics
    assert "test/corrcoef" in metrics
    assert runner.compute_objective() is not None
    assert (runner.output_dir / "test_outputs.h5").exists()
    assert (runner.output_dir / "model_trained.pt").exists()


def test_runner_classification(blobs_dir):
    arch = ArchitectureConfig(
        modules={
            "encoder": BlockSpec(
                _block_target_="modular_ml.models.blocks.FixedEncoderBlock",
                id="encoder",
                in_keys=["features"],
                out_keys=["repr"],
                in_dim=2,
                out_dim=16,
                params={"hidden_dims": [32], "activation": "relu", "trainable": True},
            ),
            "readout": BlockSpec(
                _block_target_="modular_ml.models.blocks.LinearBlock",
                id="readout",
                in_keys=["encoder.repr"],
                out_keys=["logits"],
                in_dim=16,
                out_dim=3,
            ),
        }
    )
    cfg = Config(
        seed=0,
        dataset=DatasetConfig(name="blobs", data_dir=str(blobs_dir)),
        model=ModelConfig(
            n_inputs=2, n_outputs=3, in_key="features", architecture=arch, epochs=10, batch_size=32
        ),
        optim=OptimizerConfig(
            lr=1e-2,
            losses={
                "ce": LossSpec(
                    id="ce",
                    criterion="cross_entropy",
                    in_keys=["readout.logits"],
                    label_key="label",
                ),
            },
        ),
        processing=ProcessingConfig(run_pca=True, pca_keys=["encoder.repr"]),
    )
    runner = Runner(cfg, use_wandb=True)
    runner.run_train()
    metrics = runner.run_test()
    assert "test/accuracy" in metrics
    assert metrics["test/accuracy"] > 0.7  # the task is easily separable


def test_two_stage_experiment(sine_dir, tmp_path):
    # Write a self-contained run config + experiment referencing it by abs path.
    run_cfg = tmp_path / "mini_run.yaml"
    yaml_save(
        {
            "_target_": "modular_ml.configs.main_configs.Config",
            "seed": 0,
            "dataset": {
                "_target_": "modular_ml.configs.dataset_configs.DatasetConfig",
                "name": "sine",
                "data_dir": str(sine_dir),
                "task_type": "regression",
                "in_dim": 2,
                "out_dim": 1,
                "in_key": "stim",
                "out_key": "target",
                "label_key": "target",
            },
            "model": {
                "_target_": "modular_ml.configs.model_configs.ModelConfig",
                "n_inputs": 2,
                "n_outputs": 1,
                "in_key": "stim",
                "alpha": 0.2,
                "epochs": 3,
                "batch_size": 16,
                "architecture": {
                    "_target_": "modular_ml.configs.model_configs.ArchitectureConfig",
                    "modules": {
                        "rnn": {
                            "_target_": "modular_ml.configs.model_configs.BlockSpec",
                            "_block_target_": "modular_ml.models.blocks.RNNBlock",
                            "id": "rnn",
                            "in_keys": ["stim"],
                            "out_keys": ["activity"],
                            "in_dim": 2,
                            "out_dim": 12,
                            "params": {"alpha": 0.2, "activation": "tanh"},
                        },
                        "readout": {
                            "_target_": "modular_ml.configs.model_configs.BlockSpec",
                            "_block_target_": "modular_ml.models.blocks.LinearBlock",
                            "id": "readout",
                            "in_keys": ["rnn.activity"],
                            "out_keys": ["prediction"],
                            "in_dim": 12,
                            "out_dim": 1,
                        },
                    },
                },
            },
            "optim": {
                "_target_": "modular_ml.configs.optim_configs.OptimizerConfig",
                "losses": {
                    "mse": {
                        "_target_": "modular_ml.configs.optim_configs.LossSpec",
                        "id": "mse",
                        "criterion": "mse",
                        "in_keys": ["readout.prediction"],
                        "label_key": "target",
                    },
                },
            },
            "logger": {"_target_": "modular_ml.configs.log_configs.LoggerConfig"},
            "processing": {
                "_target_": "modular_ml.configs.processing_configs.ProcessingConfig",
                "run_pca": False,
            },
        },
        run_cfg,
    )

    exp_cfg = tmp_path / "mini_exp.yaml"
    yaml_save(
        {
            "_target_": "modular_ml.configs.experiment_configs.ExperimentConfig",
            "runs": {
                "pretrain": {
                    "_target_": "modular_ml.configs.experiment_configs.RunSpec",
                    "id": "pretrain",
                    "config_path": str(run_cfg),
                    "run_mode": "train_test",
                },
                "finetune": {
                    "_target_": "modular_ml.configs.experiment_configs.RunSpec",
                    "id": "finetune",
                    "config_path": str(run_cfg),
                    "load_params_from": "pretrain",
                    "run_mode": "train_test",
                },
            },
        },
        exp_cfg,
    )

    manifest = run_experiment(exp_cfg)
    assert len(manifest["stages"]) == 2
    assert manifest["stages"][1]["load_params_from"] == "pretrain"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
