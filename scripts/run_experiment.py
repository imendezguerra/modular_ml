"""Run a multi-stage experiment (pretrain -> finetune) with a provenance manifest."""

from modular_ml.constants import CONFIG_DIR
from modular_ml.pipeline.pipeline import run_experiment

EXPERIMENT = CONFIG_DIR / "experiments" / "experiment_two_stage.yaml"


if __name__ == "__main__":
    run_experiment(EXPERIMENT)
