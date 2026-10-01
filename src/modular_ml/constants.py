"""Project-wide constants and directory paths."""

import os
from pathlib import Path

# Weights & Biases project name. Override with the MODULAR_ML_PROJECT env var.
PROJECT = os.environ.get("MODULAR_ML_PROJECT", "modular_ml")

# W&B entity (team/user). None lets wandb use the logged-in default entity.
WANDB_USER = os.environ.get("WANDB_ENTITY") or None

# Logging cadence (in epochs).
PRINT_EPOCH = 1  # how often metrics are pushed to wandb
PLOT_EPOCH = 25  # how often diagnostic figures are logged

# Repository layout. ROOT_DIR points at the repository root (two parents above
# this file: src/modular_ml/constants.py -> src -> ROOT).
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

for _d in (DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
