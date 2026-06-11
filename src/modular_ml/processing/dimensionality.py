"""PCA / dimensionality analysis logged after testing.

This is the *analysis-logging* showcase: given the saved test activations, run
PCA per key and report the number of principal components needed to reach an
explained-variance threshold, plus a 2-PC scatter figure. Entirely
domain-agnostic.
"""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA


def _flatten(activation: np.ndarray) -> np.ndarray:
    """Collapse all but the last (feature) axis -> [samples, features]."""
    arr = np.asarray(activation)
    return arr.reshape(-1, arr.shape[-1])


def pca_analysis(
    activations: Dict[str, np.ndarray],
    keys: List[str],
    pca_dims: int = 10,
    evr_threshold: float = 0.95,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Run PCA on each key.

    Returns:
        metrics: ``pca/n_pcs.<key>`` (PCs to reach the threshold) and
            ``pca/evr_top.<key>`` (variance explained by PC0) per key.
        projections: 2-PC projections per key for plotting.
    """
    metrics: Dict[str, float] = {}
    projections: Dict[str, np.ndarray] = {}
    for key in keys:
        if key not in activations:
            continue
        x = _flatten(activations[key])
        if x.shape[0] < 2 or x.shape[1] < 2:
            continue
        n_comp = min(pca_dims, x.shape[0], x.shape[1])
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(x)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        n_pcs = int(np.searchsorted(cum, evr_threshold) + 1)
        metrics[f"pca/n_pcs.{key}"] = float(min(n_pcs, n_comp))
        metrics[f"pca/evr_top.{key}"] = float(evr[0])
        projections[key] = proj[:, :2]
    return metrics, projections


def plot_projections(projections: Dict[str, np.ndarray]):
    """Build a matplotlib figure of the 2-PC projections (one panel per key)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(projections.keys())
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        proj = projections[key]
        ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.4)
        ax.set_title(key)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig
