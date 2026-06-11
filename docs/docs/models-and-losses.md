# Models and losses

This is the modular core. Both the network and the loss are **lists of specs**
that the factories turn into objects.

## Blocks and the DataDict

A `Block` (`models/blocks.py`) reads named inputs from a shared
`DataDict = Dict[str, Tensor]` and writes named outputs back, namespaced as
`id.out_key`:

```python
class Block(nn.Module):
    def forward(self, data_dict):
        inputs = self._get_inputs(data_dict)      # read self.in_keys
        outputs = self.forward_block(inputs)
        for key, out in zip(self.out_keys, outputs):
            data_dict[f"{self.id}.{key}"] = out    # write id.out_key
        return data_dict
```

Subclasses only implement `forward_block`. This uniform contract is what lets
`ModularNet` chain arbitrary blocks purely from config.

### Available general blocks

| Block | Purpose |
|-------|---------|
| `RNNBlock` | Leaky-integrator RNN over time `[T,B,N]`. |
| `LinearBlock` | Single linear layer + optional activation. |
| `FixedEncoderBlock` | MLP encoder, optionally frozen / L2-normalised. |
| `MultiInputLinearBlock` | Linear over concatenated inputs with per-input modes (active/freeze/disconnect). |
| `BroadcastBlock` | Tile features (no parameters). |
| `ConnectivityLinear` | Linear with weights quantised to {-1,0,+1} (straight-through). |

## Building a model from config

A model's `architecture.modules` is an ordered dict of `BlockSpec`s. Each spec
names the concrete class via `_block_target_`:

```yaml
modules:
  rnn:
    _block_target_: modular_ml.models.blocks.RNNBlock
    id: rnn
    in_keys: ["${model.in_key}"]
    out_keys: [activity]
    in_dim: ${model.n_inputs}
    out_dim: ${model.architecture.global_params.hidden}
    params: { alpha: ${model.alpha}, activation: tanh }
    init:
      linear_hh.weight:
        _target_: modular_ml.configs.init_configs.InitNormalRec
        g: 1.3
    freeze: [linear_hh.weight]
```

`build_modular_model(model_cfg)` Hydra-instantiates each block, applies the
per-parameter `init` specs and the `freeze` list, and returns a `ModularNet`
that runs the blocks in order.

### Per-parameter initialisation

The `init:` mapping assigns an `InitSpec` to individual parameters
(`InitNormal`, `InitUniform`, `InitNormalRec`, `InitFill`, `InitByInputs`).
`InitByInputs` initialises a weight matrix in column chunks — handy when one
linear layer fuses several inputs.

### Runtime schedule

An architecture can carry a `RuntimeScheduleConfig` that fires at epoch
milestones to freeze/unfreeze parameters or change `MultiInputLinearBlock`
input modes — a simple way to express curricula:

```yaml
schedule:
  milestones: [100]
  states:
    - modules:
        rnn: { unfreeze: [linear_hh.weight] }
```

The runner applies the stage and rebuilds the optimizer so newly-unfrozen
parameters start training.

## Building a loss from config

The objective is a weighted, schedulable sum of terms. Each `LossSpec` names a
`criterion`, the `in_keys` it reads, and an optional `label_key`:

```yaml
losses:
  mse:
    criterion: mse
    in_keys: [readout.prediction]
    label_key: ${dataset.label_key}
    reg_weight: 1.0
  smooth:
    criterion: temporal_smoothness
    in_keys: [rnn.activity]
    reg_weight: 0.01
    schedule: { type: linear_ramp, start_epoch: 0, end_epoch: 100,
                start_value: 0.0, end_value: 1.0 }
```

`build_modular_loss(optim_cfg)` produces a `ModularLoss`. At each step every
term is weighted by `reg_weight * use_reg * schedule(epoch)` and summed; the
per-term breakdown is returned for logging.

### Criteria

Supervised: `mse`, `l1`, `cross_entropy`. Regularisers: `norm2`, `norm_row`,
`pow2`, `var`, `monotonicity`, `trend`, `temporal_smoothness`, `effective_rank`.

A term with `requires_model: true` reads a **module's parameters** instead of
the data dict — useful for weight-norm penalties:

```yaml
weights:
  criterion: norm2
  in_keys: [readout]      # a module id, not a data key
  requires_model: true
  reg_weight: 1.0e-4
```
