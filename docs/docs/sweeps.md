# Hyperparameter sweeps

Sweeps use native [wandb sweeps](https://docs.wandb.ai/guides/sweeps). A sweep
YAML lists dotted config keys to vary and the objective to optimise.

## A sweep spec

```yaml
# configs/sweeps/sweep_rnn.yaml
program: scripts/run_sweep.py
method: grid
metric:
  name: sweep/objective
  goal: maximize
parameters:
  optim.lr:
    values: [0.0005, 0.001, 0.002]
  model.architecture.global_params.g:
    values: [1.0, 1.3]
```

The parameter keys are **dotted paths into the composed config**. They are
merged into the config *before* resolution, so interpolations downstream (e.g.
the recurrent gain `g` feeding an `InitNormalRec`) pick up the swept value.

## The objective

The metric `sweep/objective` is written by `Runner.compute_objective()` from the
logger config:

```yaml
# configs/runs/logger/logger_reg.yaml
objective:
  combination: weighted_mean
  metrics:
    - name: test/corrcoef
      weight: 1.0
      epoch_reduction: last
```

`epoch_reduction` reduces the metric's history (`last`/`max`/`min`/`mean`);
`combination` combines terms (`sum`/`mean`/`weighted_mean`). Use a negative
`weight` for lower-is-better terms.

## Launch a sweep

```bash
python scripts/run_sweep.py
```

This creates the sweep and runs the agent. Internally:

```python
sweep_id = wandb.sweep(sweep_cfg, project=PROJECT)

def _sweep_run():
    wandb.init(project=PROJECT)
    overrides = OmegaConf.from_dotlist([f"{k}={v}" for k, v in wandb.config.items()])
    merged = OmegaConf.merge(base_cfg, overrides)   # base loaded instantiate=False
    OmegaConf.resolve(merged)
    config = hydra.utils.instantiate(merged, _convert_="all")
    runner = Runner(config, use_wandb=True, is_sweep=True)
    runner.run_train(); runner.run_test(); runner.compute_objective()

wandb.agent(sweep_id, function=_sweep_run, count=count)
```

Point `run_sweep` at `config_tabular.yaml` + `sweep_mlp.yaml` to sweep the
tabular task instead.
