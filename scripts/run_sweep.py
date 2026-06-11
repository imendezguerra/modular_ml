"""Launch a wandb hyperparameter sweep over a run config.

wandb invokes this script once per trial (see ``program:`` in the sweep YAML),
but ``run_sweep`` also creates the sweep and the agent, so running it directly
is all that is needed.
"""

from modular_ml.constants import CONFIG_DIR
from modular_ml.pipeline.pipeline import run_sweep

CONFIG = CONFIG_DIR / "runs" / "config_sequential.yaml"
SWEEP = CONFIG_DIR / "sweeps" / "sweep_rnn.yaml"


if __name__ == "__main__":
    run_sweep(CONFIG, SWEEP, count=6, sweep_name="rnn-grid")
