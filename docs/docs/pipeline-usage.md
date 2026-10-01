# Pipeline usage

A single run is `train + test`, driven by the `Runner`
(`pipeline/runner.py`).

## Run a config

```bash
python scripts/run_pipeline.py configs/runs/config_sequential.yaml
python scripts/run_pipeline.py configs/runs/config_tabular.yaml
```

Or from Python:

```python
from modular_ml.pipeline.pipeline import run_pipeline

run_pipeline("configs/runs/config_tabular.yaml", tags=["demo"])
```

Or via the Typer CLI:

```bash
modular-ml train configs/runs/config_sequential.yaml
```

## What the Runner does

1. **Build** the model, loss and optimizer from config.
2. **Train**: for each epoch, apply the runtime schedule, then for each batch
   forward the model, compute the weighted loss, backward, clip gradients,
   re-apply sparsity masks, and log `train/*` metrics.
3. **Test**: evaluate on the test split, log `test/*` metrics, save
   `test_outputs.h5`, run the PCA analysis (`pca/*`), and save the checkpoint.
4. **Objective**: combine configured metrics into a scalar `sweep/objective`.

## Metrics

Metrics are namespaced so dashboards and sweep objectives can select by group:

- `train/total_loss`, `train/<term>.<key>`, `train/corrcoef` or `train/accuracy`
- `test/total_loss`, `test/<term>.<key>`, `test/corrcoef` or `test/accuracy`
- `pca/n_pcs.<key>`, `pca/evr_top.<key>` and a `pca/projections` figure
- `runtime_schedule/stage_index` when a schedule fires

The supervised metric (correlation vs accuracy) is chosen automatically from the
dataset's `task_type`.

## Outputs and artifacts

Each run writes to `outputs/<run_name>/` and logs these wandb artifacts:

| Artifact | Contents |
|----------|----------|
| `config` | the resolved `sim_config.yaml` |
| `model`  | checkpoints (`model_trained.pt`, plus any schedule-stage checkpoints) |
| `outputs`| `test_outputs.h5` (all activations + labels) |

Load them back later with `processing/io.py`:

```python
from modular_ml.processing.io import load_outputs_from_run, load_config_from_run

out = load_outputs_from_run("glorious-sweep-7")
cfg = load_config_from_run("glorious-sweep-7")
```
