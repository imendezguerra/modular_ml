"""Dataset wrapper + batching used by the pipeline.

Handles both data layouts produced by :mod:`modular_ml.data.data_gen`:

* ``TBF`` (sequential): tensors are ``[T, B, F]``, batched along the B axis.
* ``BF``  (tabular):    tensors are ``[B, F]``, batched along the B axis (axis 0).

``load_split`` returns a :class:`TensorDataset`; iterate it with
:meth:`TensorDataset.iter_batches`, which yields ``(data_dict, label_dict)``.
The label tensor goes only to ``label_dict`` (the model never sees it), unless
it is also the input key (e.g. an autoencoder). :meth:`TensorDataset.split`
carves a validation set out of the train split.
"""

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch

from modular_ml.configs.dataset_configs import DatasetConfig
from modular_ml.tools.loaders import h5_load, yaml_load


class TensorDataset:
    """In-memory tensors for one split, with layout-aware batching."""

    def __init__(self, tensors: Dict[str, torch.Tensor], meta: Dict, in_key: str, label_key: str):
        self.tensors = tensors
        self.meta = meta
        self.layout = meta.get("layout", "BF")
        self.in_key = in_key
        self.label_key = label_key
        self.batch_axis = 1 if self.layout == "TBF" else 0
        self.n = tensors[in_key].shape[self.batch_axis]

    def _index(self, tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(self.batch_axis, idx.to(tensor.device))

    def _subset(self, idx: torch.Tensor) -> "TensorDataset":
        tensors = {k: self._index(v, idx) for k, v in self.tensors.items()}
        return TensorDataset(tensors, self.meta, self.in_key, self.label_key)

    def _split_dicts(
        self, tensors: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        labels = {self.label_key: tensors[self.label_key]}
        data = {k: v for k, v in tensors.items() if k != self.label_key or k == self.in_key}
        return data, labels

    def split(
        self, frac: float, seed: int = 0
    ) -> Tuple["TensorDataset", Optional["TensorDataset"]]:
        """Split off a random ``frac`` of samples (along the batch axis).

        Args:
            frac: Fraction for the second split (e.g. validation).
            seed: Seed for the permutation.

        Returns:
            ``(rest, held_out)``; ``held_out`` is None when it would be empty.
        """
        n_out = round(self.n * frac)
        if n_out == 0:
            return self, None
        order = torch.randperm(self.n, generator=torch.Generator().manual_seed(seed))
        return self._subset(order[n_out:]), self._subset(order[:n_out])

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, generator: torch.Generator = None
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Yield ``(data_dict, label_dict)`` minibatches over the batch axis."""
        order = torch.randperm(self.n, generator=generator) if shuffle else torch.arange(self.n)
        for start in range(0, self.n, batch_size):
            idx = order[start : start + batch_size]
            yield self._split_dicts({k: self._index(v, idx) for k, v in self.tensors.items()})

    def full(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return the whole split as a single ``(data_dict, label_dict)``."""
        return self._split_dicts(self.tensors)


def load_split(cfg: DatasetConfig, split: str, dtype: torch.dtype = torch.float32) -> TensorDataset:
    """Load ``train``/``test`` HDF5 for a dataset config into a TensorDataset."""
    data_dir = Path(cfg.data_dir)
    meta = yaml_load(data_dir / "meta.yaml")
    raw = h5_load(data_dir / f"{split}.h5")
    label_key = cfg.label_key
    tensors: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key == label_key and meta.get("task_type") == "classification":
            tensors[key] = torch.as_tensor(value, dtype=torch.long)
        else:
            tensors[key] = torch.as_tensor(value, dtype=dtype)
    return TensorDataset(tensors, meta, in_key=cfg.in_key, label_key=label_key)
