"""Synthetic dataset generators.

Two self-contained tasks share the same on-disk format so a single pipeline can
train on either:

* **sequential** (``generate_sine``): a frequency/amplitude-cued sine-generation
  task. A constant cue per trial encodes the target wave; the model must emit the
  sine over time. Tensors are time-major ``[T, B, F]``. Showcases recurrence and
  time-series logging.
* **tabular** (``generate_blobs``): a nonlinear multi-class classification task
  (interleaved gaussian "moons"-style clusters). Tensors are ``[B, F]`` with
  integer labels. Showcases MLP blocks + cross-entropy + accuracy.

Each dataset directory holds ``train.h5``, ``test.h5`` and a ``meta.yaml`` read
back by :class:`~modular_ml.configs.dataset_configs.DatasetConfig`.
"""

from pathlib import Path
from typing import Dict

import numpy as np

from modular_ml.tools.loaders import h5_save, yaml_save


def _split(n: int, test_frac: float, rng: np.random.Generator):
    idx = rng.permutation(n)
    n_test = round(n * test_frac)
    return idx[n_test:], idx[:n_test]


def generate_sine(
    out_dir: Path,
    n_trials: int = 512,
    seq_len: int = 100,
    dt: float = 0.01,
    freq_range=(1.0, 4.0),
    amp_range=(0.5, 1.5),
    noise: float = 0.02,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Dict:
    """Generate the frequency-cued sine task and write it to ``out_dir``.

    Input features (constant over time): [freq_cue, amp_cue]. Target: the sine
    wave ``amp * sin(2*pi*freq*t)`` with additive noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len) * dt  # [T]

    freqs = rng.uniform(*freq_range, size=n_trials)
    amps = rng.uniform(*amp_range, size=n_trials)

    # Cues normalised to ~[0, 1], broadcast over time -> [T, B, 2].
    freq_cue = (freqs - freq_range[0]) / (freq_range[1] - freq_range[0])
    amp_cue = (amps - amp_range[0]) / (amp_range[1] - amp_range[0])
    cue = np.stack([freq_cue, amp_cue], axis=-1)  # [B, 2]
    stim = np.broadcast_to(cue, (seq_len, n_trials, 2)).copy()

    target = amps[None, :, None] * np.sin(2 * np.pi * freqs[None, :, None] * t[:, None, None])
    target = target + noise * rng.standard_normal(target.shape)
    stim = stim.astype(np.float32)
    target = target.astype(np.float32)

    train_idx, test_idx = _split(n_trials, test_frac, rng)
    out_dir = Path(out_dir)
    h5_save({"stim": stim[:, train_idx], "target": target[:, train_idx]}, out_dir / "train.h5")
    h5_save({"stim": stim[:, test_idx], "target": target[:, test_idx]}, out_dir / "test.h5")
    meta = {
        "name": out_dir.name,
        "task_type": "regression",
        "layout": "TBF",
        "in_dim": 2,
        "out_dim": 1,
        "dt": dt,
        "seq_len": seq_len,
        "in_key": "stim",
        "out_key": "target",
        "label_key": "target",
    }
    yaml_save(meta, out_dir / "meta.yaml")
    return meta


def generate_blobs(
    out_dir: Path,
    n_samples: int = 2000,
    n_classes: int = 4,
    n_features: int = 2,
    cluster_std: float = 0.6,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Dict:
    """Generate a nonlinear multi-class tabular task and write it to ``out_dir``.

    Classes are placed on a ring (so the boundary is nonlinear in feature space),
    then perturbed with gaussian noise.
    """
    rng = np.random.default_rng(seed)
    per = n_samples // n_classes
    feats, labels = [], []
    for c in range(n_classes):
        angle = 2 * np.pi * c / n_classes
        centre = 2.5 * np.array([np.cos(angle), np.sin(angle)])
        base = rng.normal(0.0, cluster_std, size=(per, 2)) + centre
        if n_features > 2:  # pad with informative-but-correlated extra dims
            extra = 0.5 * base[:, :1] + rng.normal(0.0, cluster_std, size=(per, n_features - 2))
            base = np.concatenate([base, extra], axis=1)
        feats.append(base)
        labels.append(np.full(per, c))
    x = np.concatenate(feats, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)

    n = x.shape[0]
    train_idx, test_idx = _split(n, test_frac, rng)
    out_dir = Path(out_dir)
    h5_save({"features": x[train_idx], "label": y[train_idx]}, out_dir / "train.h5")
    h5_save({"features": x[test_idx], "label": y[test_idx]}, out_dir / "test.h5")
    meta = {
        "name": out_dir.name,
        "task_type": "classification",
        "layout": "BF",
        "in_dim": n_features,
        "out_dim": n_classes,
        "dt": 0.0,
        "in_key": "features",
        "out_key": "logits",
        "label_key": "label",
    }
    yaml_save(meta, out_dir / "meta.yaml")
    return meta
