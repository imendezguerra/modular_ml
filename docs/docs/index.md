# modular_ml

A small, opinionated **template** for building modular AI models that are fully
tractable in [Weights & Biases](https://wandb.ai).

It demonstrates a single, reusable pattern:

> **Everything is config. The model and the loss are *data* — lists of
> Hydra-instantiable specs — and the pipeline turns those specs into objects,
> trains them, and logs every metric, figure, checkpoint and analysis to wandb.**

The codebase is **config-driven**: most changes happen in YAML, not in Python
entry scripts. To add a layer, edit a model YAML. To add a regulariser, edit an
optimizer YAML. To launch a sweep, write a sweep YAML.

## What it showcases

- **Hydra + dataclasses** for standardised, typed, composable configuration.
- A **modular network**: blocks that pass a shared `DataDict` by key, built from
  config via `hydra.utils.instantiate`.
- A **modular loss**: a weighted, schedulable sum of loss terms, also built from
  config.
- A **wandb-tracked pipeline**: train/test metrics, analysis logging
  (PCA/dimensionality), checkpoints and artifacts.
- **Hyperparameter sweeps** (wandb sweeps) and **multi-stage experiments** with
  checkpoint reuse and provenance manifests.
- **Synthetic data generation** for two example tasks that share one pipeline:
  a sequential RNN task and a tabular MLP task.

## Two example tasks

| Task | Data | Model | Loss | Metric |
|------|------|-------|------|--------|
| Sequential | frequency-cued sine waves `[T,B,F]` | `RNNBlock` → `LinearBlock` | MSE + scheduled smoothness | correlation |
| Tabular | gaussian-blob classes `[B,F]` | `FixedEncoderBlock` → `LinearBlock` | cross-entropy + weight L2 | accuracy |

Both run through the **same** `Runner`. The only differences are the configs.

## Where to go next

- [Getting started](getting-started.md) — install and log in.
- [Tutorial](tutorial.md) — a full end-to-end walkthrough.
- [Configuration](configs.md) — how Hydra + dataclasses fit together.
- [Models and losses](models-and-losses.md) — the modular core.
