"""Shared test fixtures.

wandb runs in *disabled* mode for the whole suite so tests need no account: the
library still calls wandb, but ``WANDB_MODE=disabled`` makes those calls no-ops.
"""

import os

os.environ.setdefault("WANDB_MODE", "disabled")

from pathlib import Path

import pytest

from modular_ml.data.data_gen import generate_blobs, generate_sine


@pytest.fixture(scope="session")
def sine_dir(tmp_path_factory) -> Path:
    """A tiny sequential dataset on disk."""
    d = tmp_path_factory.mktemp("data") / "sine_t"
    generate_sine(d, n_trials=48, seq_len=20, seed=1)
    return d


@pytest.fixture(scope="session")
def blobs_dir(tmp_path_factory) -> Path:
    """A tiny tabular dataset on disk."""
    d = tmp_path_factory.mktemp("data") / "blobs_t"
    generate_blobs(d, n_samples=160, n_classes=3, seed=1)
    return d
