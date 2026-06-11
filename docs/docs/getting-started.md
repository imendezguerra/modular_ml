# Getting started

## Requirements

- Python 3.10+ (the conda env pins 3.11).
- A [Weights & Biases](https://wandb.ai) account. wandb is **required** — the
  pipeline always logs to it. (The test suite runs wandb in *disabled* mode, so
  you can run `pytest` without an account.)

## Install

With conda:

```bash
conda env create -f environment.yaml
conda activate modular_ml
```

Or with pip (in a virtualenv):

```bash
pip install -e ".[dev,docs]"
```

## Log in to wandb

```bash
wandb login
```

By default runs are logged to a project called `modular_ml`. Override it with
environment variables:

```bash
export MODULAR_ML_PROJECT=my-project   # wandb project name
export WANDB_ENTITY=my-team            # wandb entity (team/user)
```

## Generate the datasets

The example tasks use synthetic data written to `data/`:

```bash
python scripts/generate_data.py
```

This creates `data/sine/` and `data/blobs/`, each with `train.h5`, `test.h5`
and a `meta.yaml`.

## Run your first model

```bash
python scripts/run_pipeline.py configs/runs/config_sequential.yaml
```

Open the run in wandb to see `train/*`, `test/*` and `pca/*` metrics, the PCA
figure, and the logged config/model/outputs artifacts.

## Verify the install

```bash
pytest        # runs with wandb disabled, no account needed
ruff check src tests
```
