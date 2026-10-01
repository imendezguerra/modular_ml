"""Reproducibility: the same seed must give bit-identical data, metrics and weights."""

import numpy as np
import pytest
import torch

from modular_ml.data.data_gen import generate_sine
from modular_ml.pipeline.runner import Runner
from modular_ml.tools.loaders import h5_load
from test_pipeline import _sine_config


def _train(data_dir, seed):
    cfg = _sine_config(data_dir, epochs=3)
    cfg.seed = seed
    runner = Runner(cfg, use_wandb=False)
    runner.run_train()
    metrics = runner.run_test()
    state = {k: v.clone() for k, v in runner.model.state_dict().items()}
    return metrics, state


def test_data_generation_is_deterministic(tmp_path):
    for name in ("a", "b"):
        generate_sine(tmp_path / name, n_trials=32, seq_len=10, seed=3)
    for split in ("train.h5", "test.h5"):
        a, b = h5_load(tmp_path / "a" / split), h5_load(tmp_path / "b" / split)
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


def test_training_is_deterministic(sine_dir):
    metrics_a, state_a = _train(sine_dir, seed=0)
    metrics_b, state_b = _train(sine_dir, seed=0)
    assert metrics_a == metrics_b
    for k in state_a:
        torch.testing.assert_close(state_a[k], state_b[k], rtol=0, atol=0)


def test_different_seeds_differ(sine_dir):
    _, state_a = _train(sine_dir, seed=0)
    _, state_b = _train(sine_dir, seed=1)
    assert any(not torch.equal(state_a[k], state_b[k]) for k in state_a)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
