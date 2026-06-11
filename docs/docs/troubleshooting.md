# Troubleshooting

### `wandb` asks me to log in / a run fails to start

The pipeline requires wandb. Run `wandb login` once. To run code paths without
an account (e.g. for tests), set `WANDB_MODE=disabled`.

### A YAML interpolation fails to parse

If an interpolation starts a list item, quote it: `in_keys: ["${model.in_key}"]`.
Unquoted `[${...}]` is invalid YAML because `{` opens a flow mapping.

### A swept hyperparameter doesn't take effect

Sweep keys are dotted paths into the **composed** config and are merged before
resolution. Make sure the key matches the config path exactly, e.g.
`model.architecture.global_params.g`, not just `g`. Confirm the value is
referenced (directly or via interpolation) somewhere in the config.

### Dimension mismatch when building the model

Block `in_dim`/`out_dim` must line up along the chain, and a block's `in_keys`
must match the `id.out_key` written by an upstream block. Dimensions usually
interpolate from the dataset (`${dataset.in_dim}`) and `global_params`, so check
those first. The dataset's `meta.yaml` is the source of truth for `in_dim` /
`out_dim`.

### `KeyError` for a data-dict key in the loss

A loss term's `in_keys` must name keys that exist in the data dict at loss time —
either a dataset key (e.g. `target`) or a block output (`rnn.activity`,
`readout.prediction`). For `requires_model: true` terms, `in_keys` are **module
ids**, not data keys.

### Cross-entropy shape errors

`cross_entropy` expects logits whose last dim is the number of classes and
integer labels. The tabular dataset stores `label` as int64; custom datasets
must do the same (set `task_type: classification` in `meta.yaml`).

### The runtime schedule didn't fire

Milestones are matched on **exact** epoch equality. Ensure a milestone is less
than `model.epochs` and that the schedule is attached under
`model.architecture.schedule`.

### Reproducing a previous run

Every run logs its resolved config as a `config` artifact. Fetch it with
`modular_ml.processing.io.load_config_from_run(run_name)` and re-run.
