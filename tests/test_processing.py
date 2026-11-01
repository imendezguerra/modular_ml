"""Post-training analyses: linear probes."""

import numpy as np
import pytest

from modular_ml.processing.probing import linear_probe


def _sequence_outputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    rep = rng.standard_normal((20, 30, 8)).astype(np.float32)  # [T, B, H]
    w = rng.standard_normal((8, 2))
    return {
        "act": rep,
        "decodable": rep @ w,  # [T, B, 2], linear in rep
        "noise": rng.standard_normal((20, 30, 2)).astype(np.float32),
        "label.cls": (rep[..., 0] > 0).astype(np.int64),  # [T, B]
    }


def test_probe_scores_decodable_vs_noise():
    m = linear_probe(_sequence_outputs(), ["act"], ["decodable", "noise", "label.cls"])
    assert m["probe/r2.act->decodable"] > 0.99
    assert m["probe/r2.act->noise"] < 0.1
    assert m["probe/acc.act->label.cls"] > 0.95


def test_probe_tabular_and_missing_keys():
    rng = np.random.default_rng(1)
    rep = rng.standard_normal((100, 4))
    out = {"repr": rep, "label.y": (rep[:, 1] > 0).astype(np.int64)}
    m = linear_probe(out, ["repr", "absent"], ["label.y", "absent"])
    assert list(m) == ["probe/acc.repr->label.y"]


def test_probe_shape_mismatch_raises():
    out = {"act": np.zeros((5, 4, 3)), "bad": np.zeros((4, 1))}
    with pytest.raises(ValueError):
        linear_probe(out, ["act"], ["bad"])
