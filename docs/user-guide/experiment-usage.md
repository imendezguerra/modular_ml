# Multi-stage experiments

An **experiment** chains several runs, optionally initialising each stage from a
previous stage's checkpoint, and writes a provenance **manifest**. This is how
you express pretrain → finetune pipelines reproducibly.

## An experiment spec

```yaml
# configs/experiments/experiment_two_stage.yaml
_target_: modular_ml.configs.experiment_configs.ExperimentConfig
runs:
  pretrain:
    id: pretrain
    config_path: config_sequential.yaml
    run_mode: train_test
    notes: stage 1 - pretrain from scratch
  finetune:
    id: finetune
    config_path: config_sequential.yaml
    load_params_from: pretrain        # reuse stage 1's weights
    run_mode: train_test
    notes: stage 2 - finetune from the pretrained checkpoint
```

`config_path` is resolved relative to `configs/runs/` (or absolute).
`run_mode` is one of `train`, `test`, `train_test`.

## Weight reuse

Two ways to initialise a stage:

- `load_params_from: <stage_id>` — load `model_trained.pt` from an earlier stage
  in **this** experiment (a later or unknown stage id raises before anything runs).
- `init_from_wandb_run: <run_id>` (+ `init_from_wandb_artifact_filename`,
  `init_from_wandb_version`) — load a checkpoint artifact from **any** prior
  wandb run.

## Run it

```bash
python scripts/run_experiment.py
# or: modular-ml experiment configs/experiments/experiment_two_stage.yaml
```

## The manifest

After each stage, the manifest so far is written to the stage's output folder
(`outputs/<run_id>/experiment_manifest.yaml`) and logged to that stage's wandb
run as an `experiment` artifact (`<run_id>-manifest`, with the manifest also as
artifact metadata). The last stage's run therefore holds the complete manifest.
It records, per stage: the run-spec id, the config path and its SHA-256, the
wandb run id and name, and the checkpoint source. This makes an experiment fully
traceable: you can reconstruct exactly which configs produced which runs.

```yaml
experiment:
  instance_id: 9f3c1a2b4d5e
  config_name: experiment_two_stage.yaml
  config_sha256: 5d41402abc...
stages:
  - run_spec_id: pretrain
    config_sha256: 2c26b46b68...
    load_params_from: null
    wandb_run_id: 3k9x2mqa
    wandb_run_name: lively-sun-12
  - run_spec_id: finetune
    config_sha256: 2c26b46b68...
    load_params_from: pretrain
    wandb_run_id: 7hd0c1ze
    wandb_run_name: brisk-frost-13
```
