# Make it yours

Most adaptations are config-only. You only write Python when you need a new
*kind* of thing: a new layer type, loss criterion, analysis or data source.

| I want to… | Edit (YAML) | Add (Python) |
|---|---|---|
| [Use my own data](#use-your-own-data) | `configs/runs/dataset/` | a loader or converter in `data/`, if needed |
| [Change the architecture](#change-the-architecture) | `configs/runs/model/` | – |
| [Add a new layer type](#add-a-new-block) | `configs/runs/model/` | a `Block` subclass in `models/blocks.py` |
| [Change the loss or add a regulariser](#change-the-loss) | `configs/runs/optim/` | – |
| [Add a new loss criterion](#add-a-new-loss-criterion) | `configs/runs/optim/` | a function in `losses/regularisers.py` |
| [Add a post-training analysis](#add-a-post-training-analysis) | `configs/runs/processing/` | a function in `processing/` + a call in the `Runner` |
| [Sweep hyperparameters](#sweep-hyperparameters) | `configs/sweeps/` | – |
| [Chain training stages](#chain-training-stages) | `configs/experiments/` | – |

A good workflow: copy the closest example config (e.g. `config_sequential.yaml`
and its group files), rename it, and change one group at a time.

## Use your own data

Add a loader or converter to `src/modular_ml/data/` (next to the example
generators in `data_gen.py`) that writes your data as `train.h5`, `test.h5` and
`meta.yaml` in the [data format](data.md), using `modular_ml.tools.loaders.h5_save`
and `yaml_save`. Then add a dataset config pointing at the output folder:

```yaml
# configs/runs/dataset/my_data.yaml
_target_: modular_ml.configs.dataset_configs.DatasetConfig
name: my_data
data_dir: my_data          # relative to data/, or an absolute path
task_type: classification  # or regression
in_key: features           # tensor fed to the first block
label_key: label           # tensor the supervised loss compares against
```

When `meta.yaml` is present, its dimensions (`in_dim`, `out_dim`, `dt`), task type
and tensor keys override the values in this file, so data and config stay in sync. To version the
dataset in wandb, upload it with `modular_ml.tools.logging.log_artifact`.

## Change the architecture

Edit the `modules` of a model config. Blocks run in the order listed; connect
them through `in_keys`, which name earlier outputs as `<block id>.<out key>`.
For example, to add a second RNN layer:

```yaml
    rnn2:
      _target_: modular_ml.configs.model_configs.BlockSpec
      _block_target_: modular_ml.models.blocks.RNNBlock
      id: rnn2
      in_keys: [rnn.activity]
      out_keys: [activity]
      in_dim: ${model.architecture.global_params.hidden}
      out_dim: ${model.architecture.global_params.hidden}
      params: { alpha: ${model.alpha}, activation: tanh }
```

…and point the readout at it: `in_keys: [rnn2.activity]`. No Python changes.

## Add a new block

Subclass `Block` in `models/blocks.py` and implement `forward_block`, which
returns a list with one tensor per `out_keys` entry. The constructor receives
`id`, `in_keys`, `out_keys`, `in_dim`, `out_dim`, the YAML `params`, and
`init_specs` / `freeze_specs` from the YAML `init` and `freeze` fields. Apply
them at the end of the constructor; this is the only place they are applied:

```python
class LayerNormBlock(Block):
    """Layer normalisation over the feature dim."""

    def __init__(self, id, in_keys, out_keys, in_dim, out_dim, init_specs=None, freeze_specs=None):
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def forward_block(self, inputs):
        return [self.norm(inputs)]
```

Then reference it from YAML with `_block_target_: modular_ml.models.blocks.LayerNormBlock`.
Wrapping a PyTorch module this way is how `RecurrentBlock`, `Conv1dBlock` and
`TransformerBlock` are built.

## Change the loss

Add, remove or reweight terms in an optim config. A term can target any tensor
in the data dict (outputs *or* hidden activity), or a module's parameters with
`requires_model: true`. Weights can follow a `schedule`. See
[Models and losses](models-and-losses.md#building-a-loss-from-config) for the
available criteria.

```yaml
  rank:
    _target_: modular_ml.configs.optim_configs.LossSpec
    id: rank
    criterion: effective_rank
    in_keys: [rnn.activity]
    reg_weight: 1.0e-3
```

The term is logged as `train/rank.rnn.activity`.

## Add a new loss criterion

1. Write the function in `losses/regularisers.py`: it takes one tensor (plus the
   label, for supervised terms) and returns a scalar. It is applied to each
   `in_keys` entry separately.
   Extra keyword arguments come from the term's YAML `params`.
2. Register it in `LossModule.CRITERION_FUNCS` in `losses/loss.py` (and in
   `SUPERVISED` if it uses a label).
3. Add its name to the `LossCriterion` enum in `configs/enums.py` (a test checks
   the two match).

## Add a post-training analysis

Analyses run on the trained model and its test outputs after testing (see
[Post-training analysis](analysis.md)). To add one, for example an
interpretability analysis:

1. Write the analysis in a new `processing/` module. It takes the output dict
   (and the model, if needed) and returns metrics and/or figures.
2. Add its options to `ProcessingConfig` (`configs/processing_configs.py`), e.g.
   `run_saliency: bool = False`, `saliency_keys: List[str]`.
3. Call it from `Runner._run_analysis` (`pipeline/runner.py`) and log the results
   with `self._log(...)` and `wlog.log_figure(...)`.
4. Switch it on per run in a `configs/runs/processing/*.yaml` file.

## Sweep hyperparameters

Write a sweep YAML whose `parameters` are dotted paths into the composed config
(e.g. `optim.lr`, `model.architecture.global_params.hidden`), and set the
objective in the logger config. See [Sweeps](sweeps.md).

## Chain training stages

List stages in an experiment YAML; each can start from a previous stage
(`load_params_from`) or any past wandb run (`init_from_wandb_run`). See
[Experiments](experiment-usage.md).

## Next

- [Repository structure](repo-structure.md): where each component lives.
- Concepts: each component in depth, starting with [Configuration](configs.md).
