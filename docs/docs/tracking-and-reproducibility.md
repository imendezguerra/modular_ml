# Tracking & reproducibility

The template is built so that a run can be reconstructed from what is logged.

## What gets logged

| Kind | Where | Notes |
|------|-------|-------|
| Metrics | `wandb.log` | namespaced `train/*`, `test/*`, `pca/*`, `runtime_schedule/*` |
| Figures | `wandb.log` | `pca/projections` |
| Config | `config` artifact | the **resolved** `sim_config.yaml` |
| Checkpoints | `model` artifact | `model_trained.pt` + schedule-stage checkpoints |
| Outputs | `outputs` artifact | `test_outputs.h5` (all activations + labels) |
| Manifest | `experiment` artifact | for multi-stage experiments |

## Determinism

The `Runner` seeds `torch` and `numpy` from `cfg.seed`, and data shuffling uses
a seeded generator. Set `seed` in the run config (or override per sweep) for
reproducible runs.

## Reconstructing a run

The resolved config is saved as an artifact, so you can always recover the exact
settings — including swept overrides — that produced a run:

```python
from modular_ml.processing.io import (
    load_config_from_run,
    load_outputs_from_run,
    load_model_params_from_run,
)

cfg = load_config_from_run("lively-sun-12")  # sim_config.yaml
outputs = load_outputs_from_run("lively-sun-12")  # test_outputs.h5
ckpt = load_model_params_from_run("lively-sun-12")  # model_trained.pt
```

These helpers look in the local `outputs/<run>/` first, then fall back to
downloading the wandb artifact.

## Project / entity

The wandb project and entity come from environment variables so the template
works across accounts without code changes:

```bash
export MODULAR_ML_PROJECT=my-project
export WANDB_ENTITY=my-team
```
