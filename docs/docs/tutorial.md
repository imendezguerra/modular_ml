# Tutorial: end to end

This walkthrough takes you from a fresh checkout to a trained model, a sweep,
and a two-stage experiment — and shows how to extend the template with a new
block and a new loss term. It uses only the configs that ship with the repo.

!!! tip "Prefer to run it interactively?"
    A runnable [**Google Colab notebook**](https://colab.research.google.com/github/imendezguerra/modular_ml/blob/main/notebooks/tutorial_colab.ipynb)
    (`notebooks/tutorial_colab.ipynb`) mirrors this page and executes end to end —
    including inline plots of the predictions, loss curve and PCA. It defaults to
    `WANDB_MODE=disabled` so it runs without a wandb account; flip one switch to
    log to a real project. Update the link/`REPO_URL` to your fork if you renamed the repo.

## 0. Setup

```bash
conda env create -f environment.yaml && conda activate modular_ml
wandb login
python scripts/generate_data.py
```

You now have `data/sine/` and `data/blobs/`.

## 1. Train the sequential model

```bash
python scripts/run_pipeline.py configs/runs/config_sequential.yaml
```

What happens:

- `config_sequential.yaml` composes the `seq_sine` dataset, the `model_rnn`
  architecture (`RNNBlock → LinearBlock`), the `optim_mse` objective, the
  `logger_reg` logger and `pca_rnn` analysis.
- The recurrent matrix starts **frozen** and is **unfrozen at epoch 100** by the
  runtime schedule — watch `runtime_schedule/stage_index` appear in wandb.
- The smoothness regulariser's weight **ramps 0 → 1** over the first 100 epochs
  (`schedules/smooth`).
- After testing you get `test/corrcoef`, a `pca/projections` figure, and logged
  `config` / `model` / `outputs` artifacts.

Open the run in wandb and confirm the training loss drops and the correlation
rises.

## 2. Train the tabular model — same pipeline

```bash
python scripts/run_pipeline.py configs/runs/config_tabular.yaml
```

Only the configs differ: `FixedEncoderBlock → LinearBlock`, a `cross_entropy`
term plus a `requires_model` weight penalty, and `test/accuracy` as the metric.
The `Runner`, the data layer and the loss machinery are identical. This is the
point of the template: **the engineering is shared; the configs vary.**

## 3. Inspect outputs in Python

```python
from modular_ml.processing.io import load_outputs_from_run
out = load_outputs_from_run("<your-run-name>")
print(out.keys())          # rnn.activity, readout.prediction, label.target, ...
print(out["rnn.activity"].shape)
```

## 4. Run a sweep

```bash
python scripts/run_sweep.py
```

This sweeps the learning rate and recurrent gain `g` over the sequential config
(`configs/sweeps/sweep_rnn.yaml`), maximising `sweep/objective`
(= `test/corrcoef`). In the wandb sweep view, compare runs and find the best
`g`/`lr`.

## 5. Run a two-stage experiment

```bash
python scripts/run_experiment.py
```

Stage 1 (`pretrain`) trains from scratch; stage 2 (`finetune`) initialises from
stage 1's `model_trained.pt` (`load_params_from: pretrain`). A manifest is
written to `outputs/experiment_<id>/experiment_manifest.yaml`.

## 6. Extend it — add a block

Say you want a second hidden RNN layer. Edit `configs/runs/model/model_rnn.yaml`
and insert a block between `rnn` and `readout`:

```yaml
    rnn2:
      _block_target_: modular_ml.models.blocks.RNNBlock
      id: rnn2
      in_keys: [rnn.activity]
      out_keys: [activity]
      in_dim: ${model.architecture.global_params.hidden}
      out_dim: ${model.architecture.global_params.hidden}
      params: { alpha: ${model.alpha}, activation: tanh }
```

…and point the readout at it: `in_keys: [rnn2.activity]`. No Python changes.

To add a genuinely new layer type, subclass `Block` in `models/blocks.py`,
implement `forward_block`, and reference it via `_block_target_`.

## 7. Extend it — add a loss term

Add a regulariser on the RNN activity in
`configs/runs/optim/optim_mse.yaml`:

```yaml
  rank:
    _target_: modular_ml.configs.optim_configs.LossSpec
    id: rank
    criterion: effective_rank
    in_keys: [rnn.activity]
    reg_weight: 1.0e-3
```

The term shows up as `train/rank.rnn.activity` and contributes to the total
loss. Available criteria are listed in
[Models and losses](models-and-losses.md).

## Recap

You trained two different models through one pipeline, swept hyperparameters,
ran a staged experiment, and extended both the architecture and the objective —
all by editing YAML. That is the workflow this template is designed for.
