# modular_ml

[![CI](https://github.com/imendezguerra/modular_ml/actions/workflows/ci.yml/badge.svg)](https://github.com/imendezguerra/modular_ml/actions/workflows/ci.yml)

A small, opinionated **template** for building modular AI models that are fully
tractable in [Weights & Biases](https://wandb.ai). It shows how
[Hydra](https://hydra.cc) structured configs (backed by Python dataclasses) can
standardise configuration, and how a **modular network and loss** are
instantiated *entirely from those configs* and driven through a wandb-tracked
train / test / analyse pipeline — with hyperparameter sweeps and multi-stage
experiments.

It is deliberately domain-agnostic: two synthetic example tasks (a sequential
RNN task and a tabular MLP task) share one pipeline to demonstrate the
modularity.

## The core idea

A run is one `Config` dataclass. The model is a list of **blocks** that thread a
shared `DataDict` (each block reads `in_keys` and writes `id.out_key`). Both the
network and the loss are built from config via `hydra.utils.instantiate`:

```
YAML  ──compose──▶  DictConfig  ──resolve──▶  ──instantiate──▶  Config (dataclasses)
                                                                   │
                          build_modular_model / build_modular_loss │
                                                                   ▼
                                          ModularNet + ModularLoss ──▶ Runner ──▶ wandb
```

## Quickstart

```bash
# 1. Install (conda or pip)
conda env create -f environment.yaml && conda activate modular_ml
# or: pip install -e ".[dev,docs]"

# 2. Log in to wandb (required)
wandb login

# 3. Generate the synthetic datasets
python scripts/generate_data.py

# 4. Train + test a run (sequential or tabular)
python scripts/run_pipeline.py configs/runs/config_sequential.yaml
python scripts/run_pipeline.py configs/runs/config_tabular.yaml

# 5. Sweep / experiment
python scripts/run_sweep.py
python scripts/run_experiment.py
```

Run the tests (no wandb account needed — wandb runs in disabled mode):

```bash
pytest
```

For development, install the git hooks once (pre-commit ships with the `dev` extras).
They lint and format on every commit and run the tests on every push:

```bash
pre-commit install
pre-commit run --all-files   # optional: run every hook on the whole repo now
```

## Documentation

Full docs and an end-to-end tutorial live in [`docs/`](docs/) (built with
MkDocs):

```bash
mkdocs serve   # then open http://127.0.0.1:8000
```

Start with **Getting started**, then the **Tutorial**.
