# Tracking & reproducibility

The template is built so that a run can be reconstructed from what is logged.

## What gets logged

| Kind | Where | Notes |
|------|-------|-------|
| Metrics | `wandb.log` | namespaced `train/*`, `val/*`, `test/*`, `runtime_schedule/*`, plus analysis metrics (`pca/*`, `probe/*` in the examples) |
| Figures | `wandb.log` | analysis figures (`pca/projections` in the examples) |
| Config | `config` artifact + run config | the **resolved** `sim_config.yaml`, including sweep overrides; the full config is also in the wandb run config |
| Checkpoints | `model` artifact | `model_trained.pt` + schedule-stage checkpoints |
| Outputs | `outputs` artifact | `test_outputs.h5` (all activations + labels) |
| Manifest | `experiment` artifact | multi-stage experiments, logged after each stage; each stage's wandb run is also tagged `experiment` and `<id>` |
| Datasets | your own artifact | optional: upload with `modular_ml.tools.logging.log_artifact` (not done in the examples, to save wandb storage) |

## Determinism

The `Runner` seeds `torch` and `numpy` from `cfg.seed`. Data shuffling and the
train/validation split use seeded generators. Set `seed` in the run config (or override per sweep) for
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

cfg = load_config_from_run("3k9x2mqa")  # sim_config.yaml
outputs = load_outputs_from_run("3k9x2mqa")  # test_outputs.h5
ckpt = load_model_params_from_run("3k9x2mqa")  # model_trained.pt
```

Runs are addressed by their wandb **run id** (the last part of the run URL),
which is unique; display names like `lively-sun-12` can repeat. Local outputs
live in `outputs/<run_id>/` and artifacts are named `<run_id>-<kind>`. These
helpers look in the local folder first, then fall back to downloading the
wandb artifact.

## Project / entity

The wandb project and entity come from environment variables so the template
works across accounts without code changes:

```bash
export MODULAR_ML_PROJECT=my-project
export WANDB_ENTITY=my-team
```

## Next

[Make it yours](make-it-yours.md) shows which file to edit to swap in your own
data, blocks, losses and analyses.
