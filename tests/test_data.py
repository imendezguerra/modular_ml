"""Synthetic data generation, HDF5 round-trip and batching."""

import torch

from modular_ml.configs.dataset_configs import DatasetConfig
from modular_ml.data.datasets import load_split
from modular_ml.tools.loaders import h5_load


def test_sine_shapes_and_meta(sine_dir):
    train = h5_load(sine_dir / "train.h5")
    # Time-major [T, B, F].
    assert train["stim"].shape[0] == 20
    assert train["stim"].shape[2] == 2
    assert train["target"].shape[2] == 1


def test_blobs_shapes(blobs_dir):
    train = h5_load(blobs_dir / "train.h5")
    assert train["features"].ndim == 2
    assert train["label"].ndim == 1


def test_load_split_sequential_batching(sine_dir):
    cfg = DatasetConfig(name="sine", data_dir=str(sine_dir))
    ds = load_split(cfg, "train")
    data, labels = next(ds.iter_batches(8, shuffle=False))
    assert data["stim"].shape[1] == 8  # batch on axis 1
    assert labels["target"].shape[1] == 8


def test_labels_are_not_model_inputs(sine_dir):
    ds = load_split(DatasetConfig(name="sine", data_dir=str(sine_dir)), "train")
    data, labels = next(ds.iter_batches(8))
    assert "target" not in data and "target" in labels
    # Unless the label is also the input (e.g. an autoencoder).
    ds.label_key = "stim"
    data, labels = ds.full()
    assert "stim" in data and "stim" in labels


def test_train_val_split(sine_dir):
    ds = load_split(DatasetConfig(name="sine", data_dir=str(sine_dir)), "train")
    train, val = ds.split(0.25, seed=0)
    assert train.n + val.n == ds.n and val.n == round(0.25 * ds.n)
    # Disjoint trials: no trial of val appears in train.
    flat = lambda d: {tuple(d.tensors["target"][:, i, 0].tolist()) for i in range(d.n)}  # noqa: E731
    assert not flat(train) & flat(val)
    assert ds.split(0.0)[1] is None


def test_load_split_tabular_batching(blobs_dir):
    cfg = DatasetConfig(name="blobs", data_dir=str(blobs_dir))
    ds = load_split(cfg, "train")
    data, labels = next(ds.iter_batches(8, shuffle=False))
    assert data["features"].shape[0] == 8  # batch on axis 0
    assert labels["label"].dtype == torch.long
