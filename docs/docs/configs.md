# Configuration

Configuration is the heart of this template. Every subsystem is described by a
Python **dataclass**, and every run is a YAML file that Hydra composes into one
of those dataclasses.

## The dataclass schema

The root config is `modular_ml.configs.main_configs.Config`:

```python
@dataclass
class Config:
    seed: int = 0
    dataset: DatasetConfig
    model: ModelConfig
    optim: OptimizerConfig
    logger: LoggerConfig
    processing: ProcessingConfig
```

Each field is its own dataclass (`DatasetConfig`, `ModelConfig`, …). Every
dataclass carries a `_target_` pointing at itself, which is what lets Hydra
turn the composed YAML back into typed Python objects.

## How a config is loaded

`load_config(path)` does three things:

1. **compose** — Hydra reads the run YAML and follows its `defaults:` list to
   pull in config groups (`dataset/…`, `model/…`, …).
2. **resolve** — interpolations (`${…}`) and custom resolvers are evaluated.
3. **instantiate** — every `_target_` becomes a dataclass instance.

```python
from modular_ml.configs.main_configs import load_config

cfg = load_config("configs/runs/config_sequential.yaml")
cfg.model.n_inputs  # -> 2  (typed, resolved)
```

Pass `instantiate=False` to keep a raw `DictConfig` (used when merging sweep
overrides before instantiation).

## Config groups and composition

A concrete run file is mostly a `defaults:` list:

```yaml
# configs/runs/config_sequential.yaml
_target_: modular_ml.configs.main_configs.Config
defaults:
  - dataset: seq_sine
  - model: model_rnn
  - optim: optim_mse
  - logger: logger_reg
  - processing: pca_rnn
  - _self_
seed: 0
```

Each entry selects a file from a group directory (e.g. `model/model_rnn.yaml`)
and mounts it under that key. Swap `model_rnn` for another file to change the
architecture without touching anything else. `_self_` last means values in this
file win over the groups.

## Interpolation and resolvers

Values can reference other values, and three custom resolvers are registered
(in `tools/hydra_utils.py`):

```yaml
n_inputs: ${dataset.in_dim}                       # cross-group reference
alpha:    ${divide:${model.dt_s},${model.tau_s}}  # dt / tau
out_dim:  ${multiply:${...base},${...ratio}}
layers:   ${list_repeat:64,3}                     # -> [64, 64, 64]
```

!!! tip "Quoting interpolations inside lists"
    YAML treats `{` specially, so an interpolation that starts a flow sequence
    must be quoted: `in_keys: ["${model.in_key}"]`. When a value is *exactly*
    one interpolation, OmegaConf preserves the referenced **type** (so
    `["${...hidden}"]` resolves to `[64]`, an int, not `["64"]`).

## Overriding from the command line

Because it's Hydra, any leaf can be overridden when you compose. In code:

```python
cfg = load_config("configs/runs/config_sequential.yaml")
cfg.model.epochs = 50  # plain attribute access after instantiation
```

For sweeps, overrides are merged *before* instantiation — see
[Sweeps](sweeps.md).
