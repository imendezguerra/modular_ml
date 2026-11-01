"""Regenerate the figures used in the docs (``docs/assets/figures/``).

Trains both example configs locally (wandb off) and plots the data, training
curves, predictions and post-training analysis. Run after
``python scripts/generate_data.py``::

    python scripts/make_doc_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from modular_ml.configs.main_configs import load_config
from modular_ml.constants import CONFIG_DIR, DATA_DIR, ROOT_DIR
from modular_ml.pipeline.runner import Runner
from modular_ml.processing.dimensionality import pca_analysis
from modular_ml.tools.loaders import h5_load

FIG_DIR = ROOT_DIR / "docs" / "assets" / "figures"


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


def _scatter_classes(ax, xy, labels) -> None:
    sc = ax.scatter(*xy.T, c=labels, cmap="tab10", vmin=0, vmax=9, s=6, alpha=0.7)
    ax.legend(*sc.legend_elements(), title="class", fontsize=8)


def plot_data() -> None:
    """Example inputs and targets of both synthetic datasets."""
    sine = h5_load(DATA_DIR / "sine" / "train.h5")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for b in range(3):
        (line,) = ax[0].plot(sine["target"][:, b, 0], label=f"trial {b}")
        ax[1].plot(sine["stim"][:, b, 0], c=line.get_color())
        ax[1].plot(sine["stim"][:, b, 1], "--", c=line.get_color())
    ax[0].set(title="target sine per trial", xlabel="time step (T)")
    ax[1].set(title="input cues: frequency (solid), amplitude (dashed)", xlabel="time step (T)")
    ax[1].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=8)
    _save(fig, "data_sine.png")

    blobs = h5_load(DATA_DIR / "blobs" / "train.h5")
    fig, ax = plt.subplots(figsize=(4, 3.6))
    _scatter_classes(ax, blobs["features"], blobs["label"])
    ax.set(title="blobs: features by class", xlabel="feature 0", ylabel="feature 1")
    _save(fig, "data_blobs.png")


def _train(config_name: str) -> Runner:
    runner = Runner(load_config(CONFIG_DIR / "runs" / config_name), use_wandb=False)
    runner.run_train()
    runner.run_test()
    return runner


def plot_sequential() -> None:
    """Loss curve, test predictions and PCA of the RNN activity."""
    runner = _train("config_sequential.yaml")
    out = h5_load(runner.output_dir / "test_outputs.h5")
    hist = runner._metric_history

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(hist["train/total_loss"])
    milestone = runner.cfg.model.architecture.schedule.milestones[0]
    ax[0].axvline(milestone, ls=":", c="grey", label="recurrent weights unfrozen")
    ax[0].set(title="training loss", xlabel="epoch", yscale="log")
    ax[0].legend(fontsize=8)
    for b in range(3):
        (line,) = ax[1].plot(out["label.target"][:, b, 0], lw=1)
        ax[1].plot(out["readout.prediction"][:, b, 0], "--", c=line.get_color())
    ax[1].set(title="test trials: target (solid) vs prediction (dashed)", xlabel="time step")
    _, proj = pca_analysis({"rnn.activity": out["rnn.activity"]}, ["rnn.activity"])
    ax[2].scatter(*proj["rnn.activity"].T, s=2, alpha=0.3)
    ax[2].set(title="analysis: PCA of rnn.activity", xlabel="PC1", ylabel="PC2")
    _save(fig, "results_sequential.png")


def plot_tabular() -> None:
    """Loss curve, test confusion matrix and PCA of the learned encoder representation."""
    runner = _train("config_tabular.yaml")
    out = h5_load(runner.output_dir / "test_outputs.h5")
    hist = runner._metric_history

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(hist["train/total_loss"])
    ax[0].set(title="training loss", xlabel="epoch", yscale="log")
    ConfusionMatrixDisplay.from_predictions(
        out["label.label"], out["readout.logits"].argmax(-1), ax=ax[1], colorbar=False, cmap="Blues"
    )
    ax[1].set(title="test confusion matrix")
    _, proj = pca_analysis({"encoder.repr": out["encoder.repr"]}, ["encoder.repr"])
    _scatter_classes(ax[2], proj["encoder.repr"], out["label.label"])
    ax[2].set(title="analysis: PCA of encoder.repr", xlabel="PC1", ylabel="PC2")
    _save(fig, "results_tabular.png")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_data()
    plot_sequential()
    plot_tabular()
