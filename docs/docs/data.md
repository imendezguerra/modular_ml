# Data

The template ships two synthetic generators (`data/data_gen.py`) so the whole
pipeline runs out of the box, with no external datasets.

## On-disk format

Each dataset directory contains:

- `train.h5`, `test.h5` — flat HDF5 of named arrays (via `tools/loaders.h5_save`).
- `meta.yaml` — dimensions, dt, tensor keys and the data **layout**.

`DatasetConfig.__post_init__` reads `meta.yaml`, so dimensions stay in sync
between the data and the config.

## The two tasks

### Sequential (sine generation)

`generate_sine` produces a frequency/amplitude-cued sine task. A constant cue
per trial encodes the target wave; the model must emit the sine over time.

- Layout `TBF`: tensors are `[T, B, F]`, batched along the **B** axis.
- Keys: `stim` (input), `target` (label). `in_dim=2`, `out_dim=1`.

### Tabular (gaussian blobs)

`generate_blobs` produces a nonlinear multi-class classification task — classes
on a ring, perturbed with gaussian noise.

- Layout `BF`: tensors are `[B, F]`, batched along axis 0.
- Keys: `features` (input), `label` (int label). `out_dim = n_classes`.

## Batching

`data/datasets.py` wraps a split in a `TensorDataset` that is **layout-aware**:
it batches along the correct axis for either task and yields
`(data_dict, label_dict)` minibatches that drop straight into the model and
loss.

```python
from modular_ml.configs.dataset_configs import DatasetConfig
from modular_ml.data.datasets import load_split

ds = load_split(DatasetConfig(name="sine", data_dir="sine"), "train")
data, labels = next(ds.iter_batches(batch_size=32))
```

## Generating data

```bash
python scripts/generate_data.py             # both datasets
python scripts/generate_data.py gen_sine    # just one (by config stem)
```

The generation parameters live in `configs/data_gen/*.yaml`.

## Bringing your own data

Write a `train.h5` / `test.h5` / `meta.yaml` triple in the same format and point
a `dataset/*.yaml` config at the directory. As long as the tensor keys match
your blocks' `in_keys` and the loss `label_key`, nothing else changes.
