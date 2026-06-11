"""Train + test a single run config and log to wandb.

Edit ``CONFIG`` (or pass a path) to switch between the sequential and tabular
examples. Requires a wandb account (`wandb login`).
"""

import sys
from pathlib import Path

from modular_ml.constants import CONFIG_DIR
from modular_ml.pipeline.pipeline import run_pipeline

CONFIG = CONFIG_DIR / "runs" / "config_sequential.yaml"


if __name__ == "__main__":
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG
    run_pipeline(config, tags=["example"], notes="single run from run_pipeline.py")
