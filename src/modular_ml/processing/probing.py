"""Linear probes logged after testing.

For each (representation, target) pair, fit a linear model from the
representation to the target and report its cross-validated score: R^2 for
float targets (ridge regression), accuracy for integer targets (logistic
regression). Folds split by trial, so the time steps of one sequence never sit
in both the fit and the score set. Entirely domain-agnostic.
"""

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def linear_probe(
    outputs: Dict[str, np.ndarray],
    keys: List[str],
    targets: List[str],
    n_folds: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """Decode each target from each representation with a linear model.

    Args:
        outputs: Arrays by key, e.g. the test outputs (labels as ``label.<key>``).
        keys: Representations ``[..., B, H]`` to probe (e.g. ``rnn.activity``).
        targets: Variables to decode, shaped ``[..., B, F]`` or ``[..., B]``
            with the same leading dims as the representation.
        n_folds: Cross-validation folds over trials.
        seed: Seed for assigning trials to folds.

    Returns:
        ``probe/r2.<key>-><target>`` or ``probe/acc.<key>-><target>`` per pair.
    """
    metrics: Dict[str, float] = {}
    for key in keys:
        if key not in outputs:
            continue
        rep = np.asarray(outputs[key])
        # Assign whole trials (axis -2) to folds.
        n_trials = rep.shape[-2]
        trial_fold = np.random.default_rng(seed).permutation(n_trials) % n_folds
        cv = PredefinedSplit(np.broadcast_to(trial_fold, rep.shape[:-1]).ravel())
        x = rep.reshape(-1, rep.shape[-1])
        for target in targets:
            if target not in outputs:
                continue
            y = np.asarray(outputs[target])
            if y.ndim == rep.ndim - 1:
                y = y[..., None]
            if y.shape[:-1] != rep.shape[:-1]:
                raise ValueError(
                    f"Probe target '{target}' {y.shape} does not match '{key}' {rep.shape}"
                )
            y = y.reshape(-1, y.shape[-1])
            if np.issubdtype(y.dtype, np.integer):
                name, scoring, y = "acc", "accuracy", y.ravel()
                probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            else:
                name, scoring = "r2", "r2"
                probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            score = cross_val_score(probe, x, y, cv=cv, scoring=scoring).mean()
            metrics[f"probe/{name}.{key}->{target}"] = float(score)
    return metrics
