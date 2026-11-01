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
        inputs = self._get_inputs(data_dict)  # read self.in_keys
        outputs = self.forward_block(inputs)
        for key, out in zip(self.out_keys, outputs):
            data_dict[f"{self.id}.{key}"] = out  # write id.out_key
        return data_dict
```

Subclasses only implement `forward_block`. This uniform contract is what lets
`ModularNet` chain arbitrary blocks purely from config.

### Available general blocks

| Block | Purpose |
|-------|---------|
| `RNNBlock` | Leaky-integrator RNN over time `[T,B,N]`. |
| `RecurrentBlock` | PyTorch GRU, LSTM or Elman RNN (`cell`); stacked, bidirectional. Outputs `[sequence, final hidden]`. |
| `Conv1dBlock` | Stack of temporal convolutions; keeps `T`, optionally `causal`. |
| `TransformerBlock` | Input projection + sinusoidal positions + transformer encoder; optionally `causal`. |
| `LinearBlock` | Single linear layer + optional activation. |
| `FixedEncoderBlock` | MLP encoder, optionally frozen / L2-normalised. |
| `MultiInputLinearBlock` | Linear over concatenated inputs with per-input modes (active/freeze/disconnect). |
| `BroadcastBlock` | Tile features (no parameters). |
| `ConnectivityLinear` | Linear with weights quantised to {-1,0,+1} (straight-through). |

Sequence blocks use `[T, B, F]`, the PyTorch default, so PyTorch modules plug in
without reshaping. Activations are shared across blocks: `tanh`, `relu`, `gelu`,
`none` (`TransformerBlock`: `relu`, `gelu`).

### Dropout

Blocks with layers take a `dropout` param (default `0.0`). It is active in
training and off in testing (the runner switches `train()` / `eval()`).

| Block | Where dropout applies |
|-------|-----------------------|
| `LinearBlock` | On the input (e.g. dropout on hidden activity before a readout). |
| `FixedEncoderBlock`, `Conv1dBlock` | After each hidden activation. |
| `RecurrentBlock` | Between stacked layers (needs `num_layers > 1`). |
| `TransformerBlock` | Inside each layer (attention, feed-forward, residual). |

### PyTorch parameter names

Blocks that wrap PyTorch modules keep PyTorch's parameter names, so `init` and
`freeze` use them directly, e.g. `rnn.weight_hh_l0` (`RecurrentBlock`),
`convs.0.weight` (`Conv1dBlock`), `encoder.layers.0` (`TransformerBlock`). Print
`[n for n, _ in block.named_parameters()]` to list them.


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
that runs the blocks in order. Each block receives its spec's `init` and
`freeze` as `init_specs` / `freeze_specs` and applies them once, at the end of
its constructor.

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

The runner applies the stage at the milestone epoch. The optimizer covers all
parameters from the start (frozen ones have no gradient and are skipped), so
newly-unfrozen parameters start training without resetting the optimizer state
of the others.

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

| Criterion | Kind | Notes / `params` |
|-----------|------|------------------|
| `mse`, `l1` | supervised | |
| `huber` | supervised | Robust regression; `delta`. |
| `cross_entropy` | supervised | Logits `[...,C]`, integer labels; `label_smoothing`, `weight` (per class), `ignore_index`. |
| `bce_with_logits` | supervised | Binary / multi-label; `pos_weight` (per output). |
| `cosine` | supervised | `1 - cos` similarity along `dim` (default `-1`). |
| `norm1`, `norm2` | regulariser | L1 / L2 norm (sparsity / weight decay). |
| `norm_row` | regulariser | Uniform row norms. |
| `pow2`, `var` | regulariser | Mean square / variance. |
| `monotonicity`, `trend` | regulariser | Along the last dim. |
| `temporal_smoothness` | regulariser | `dim` (default `0`, time), `p` (`1` or `2`). |
| `effective_rank` | regulariser | Negated participation ratio. |

A term's `params` are passed to its criterion:

```yaml
ce:
  criterion: cross_entropy
  in_keys: [readout.logits]
  label_key: ${dataset.label_key}
  params: { label_smoothing: 0.1, weight: [1.0, 2.0, 1.0, 1.0] }
```

A term with `requires_model: true` reads a **module's parameters** instead of
the data dict — useful for weight-norm penalties:

```yaml
weights:
  criterion: norm2
  in_keys: [readout]      # a module id, not a data key
  requires_model: true
  reg_weight: 1.0e-4
```
