"""The Runner: builds everything from a Config and drives train/test + logging.

Responsibilities:
* build the modular model, loss and optimizer from config,
* run the training loop (forward -> weighted loss -> backward -> clip ->
  sparsity-mask enforce -> runtime schedule -> wandb log -> checkpoints),
* run the test loop (eval metrics, save outputs, PCA analysis, save model),
* compute a scalar sweep objective from the logged metric history.

Everything logs to wandb; metrics are namespaced ``train/*``, ``test/*``,
``pca/*`` so dashboards and sweep objectives can select by group.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import wandb
from loguru import logger

from modular_ml.configs.main_configs import Config, load_config
from modular_ml.constants import OUTPUT_DIR, PRINT_EPOCH
from modular_ml.data.datasets import TensorDataset, load_split
from modular_ml.losses.loss import build_modular_loss
from modular_ml.models.network import build_modular_model
from modular_ml.pipeline.runtime_schedule import resolve_runtime_schedule_stage
from modular_ml.processing.dimensionality import pca_analysis, plot_projections
from modular_ml.tools import logging as wlog
from modular_ml.tools.loaders import h5_save

_OPTIMIZERS = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD}


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


class Runner:
    """Drive one train/test run from a resolved :class:`Config`."""

    def __init__(
        self,
        config: Union[str, Path, Config],
        use_wandb: bool = True,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        is_sweep: bool = False,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.cfg: Config = config
        self.use_wandb = use_wandb
        self.is_sweep = is_sweep

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        # Build model + loss + optimizer from config.
        self.model = build_modular_model(self.cfg.model).to(self.device)
        self.loss = build_modular_loss(self.cfg.optim).to(self.device)
        self.optimizer = self._build_optimizer()

        # Identify the supervised prediction/label keys for metric reporting.
        self.pred_key, self.label_key = self._resolve_pred_label()
        self.task_type = self.cfg.dataset.task_type.value

        self._metric_history: Dict[str, List[float]] = {}
        self._last_epoch = 0
        self.train_loss = None

        if self.use_wandb:
            wlog.init_wandb_run(
                config=_config_to_dict(self.cfg),
                name=run_name,
                tags=tags,
                notes=notes,
                group=run_name if is_sweep else None,
            )
        self.run_name = getattr(getattr(wandb, "run", None), "name", None) or "local"
        self.output_dir = OUTPUT_DIR / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- setup helpers -------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.optim
        cls = _OPTIMIZERS[getattr(opt_cfg.optim, "value", opt_cfg.optim)]
        params = [p for p in self.model.parameters() if p.requires_grad]
        kwargs = {"lr": opt_cfg.lr}
        if opt_cfg.weight_decay:
            kwargs["weight_decay"] = opt_cfg.weight_decay
        return cls(params, **kwargs)

    def _resolve_pred_label(self):
        for module in self.loss.loss_modules:
            if module.label_key is not None and module.in_keys:
                return module.in_keys[0], module.label_key
        # Fall back to model/dataset config.
        first = next(iter(self.cfg.model.architecture.modules.values()))
        return f"{first.id}.{first.out_keys[0]}", self.cfg.dataset.label_key

    def _load_dataset(self, split: str) -> TensorDataset:
        ds = load_split(self.cfg.dataset, split, dtype=self.dtype)
        ds.tensors = {k: v.to(self.device) for k, v in ds.tensors.items()}
        return ds

    # --- metrics -------------------------------------------------------------
    def _supervised_metric(self, data: Dict[str, torch.Tensor],
                           labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        pred = data.get(self.pred_key)
        target = labels.get(self.label_key)
        if pred is None or target is None:
            return {}
        if self.task_type == "classification":
            logits = pred.reshape(-1, pred.shape[-1])
            preds = logits.argmax(dim=-1)
            acc = (preds == target.reshape(-1)).float().mean().item()
            return {"accuracy": acc}
        # regression: Pearson correlation between flattened pred and target
        p = pred.reshape(-1).float()
        t = target.reshape(-1).float()
        if p.numel() > 1:
            corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
        else:
            corr = float("nan")
        return {"corrcoef": corr}

    def _log(self, epoch: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and v is not None:
                if not (isinstance(v, float) and np.isnan(v)):
                    self._metric_history.setdefault(k, []).append(float(v))
        if self.use_wandb:
            wlog.log_metrics(metrics, step=epoch)

    # --- training ------------------------------------------------------------
    def run_train(self) -> None:
        """Run the full training loop."""
        ds = self._load_dataset("train")
        gen = torch.Generator().manual_seed(self.cfg.seed)
        schedule = self.cfg.model.architecture.schedule
        self.model.train()

        for epoch in range(self.cfg.model.epochs):
            self._maybe_apply_schedule(schedule, epoch)
            epoch_losses, epoch_metrics = [], []
            for data, labels in ds.iter_batches(self.cfg.model.batch_size, generator=gen):
                self.optimizer.zero_grad()
                out = self.model(data)
                total, terms = self.loss(out, labels, model=self.model, epoch=epoch, mode="train")
                total.backward()
                if self.cfg.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.clip_grad
                    )
                self.optimizer.step()
                self.model.apply_sparsity_masks()
                epoch_losses.append(total.item())
                epoch_metrics.append(self._supervised_metric(out, labels))

            self.train_loss = float(np.mean(epoch_losses))
            if epoch % PRINT_EPOCH == 0:
                metrics = {"train/total_loss": self.train_loss, "train/epoch": float(epoch)}
                metrics.update(_flatten_terms("train", terms))
                metrics.update({f"train/{k}": _mean_metric(epoch_metrics, k)
                                for k in (epoch_metrics[0] if epoch_metrics else {})})
                self._log(epoch, metrics)
        self._last_epoch = self.cfg.model.epochs

    def _maybe_apply_schedule(self, schedule, epoch: int) -> None:
        resolved = resolve_runtime_schedule_stage(schedule, epoch)
        if resolved is None:
            return
        stage_idx, state = resolved
        applied = self.model.apply_runtime_schedule(state)
        # Rebuild optimizer so newly-unfrozen params are included.
        self.optimizer = self._build_optimizer()
        if applied:
            logger.info(f"Runtime schedule stage {stage_idx} @ epoch {epoch}: {applied}")
            self._log(epoch, {"runtime_schedule/stage_index": float(stage_idx)})
        if state.get("save_checkpoint"):
            self._save_model(epoch, f"model_stage_{stage_idx}_epoch_{epoch}.pt")

    # --- testing -------------------------------------------------------------
    def run_test(self) -> Dict[str, float]:
        """Evaluate on the test split, save outputs, run analysis, save model."""
        ds = self._load_dataset("test")
        self.model.eval()
        with torch.no_grad():
            data, labels = ds.full()
            out = self.model(data)
            total, terms = self.loss(out, labels, model=self.model,
                                     epoch=self._last_epoch, mode="test")

        metrics = {"test/total_loss": total.item()}
        metrics.update(_flatten_terms("test", terms))
        metrics.update({f"test/{k}": v for k, v in self._supervised_metric(out, labels).items()})
        self._log(self._last_epoch, metrics)

        self._save_outputs(out, labels)
        self._run_analysis(out)
        self._save_model(self._last_epoch, "model_trained.pt")
        return metrics

    def _save_outputs(self, out: Dict[str, torch.Tensor],
                      labels: Dict[str, torch.Tensor]) -> None:
        payload = {k: _to_numpy(v) for k, v in out.items() if torch.is_tensor(v)}
        for k, v in labels.items():
            payload[f"label.{k}"] = _to_numpy(v)
        path = self.output_dir / "test_outputs.h5"
        h5_save(payload, path)
        if self.use_wandb:
            wlog.log_artifact(path, "outputs", "test_outputs")

    def _run_analysis(self, out: Dict[str, torch.Tensor]) -> None:
        pcfg = self.cfg.processing
        if not pcfg.run_pca or not pcfg.pca_keys:
            return
        activations = {k: _to_numpy(out[k]) for k in pcfg.pca_keys if k in out}
        metrics, projections = pca_analysis(
            activations, list(activations.keys()), pcfg.pca_dims, pcfg.evr_threshold
        )
        self._log(self._last_epoch, metrics)
        if self.use_wandb and projections:
            fig = plot_projections(projections)
            if fig is not None:
                wlog.log_figure("pca/projections", fig, step=self._last_epoch)

    # --- checkpoints ---------------------------------------------------------
    def _save_model(self, epoch: int, filename: str) -> None:
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "last_epoch": self._last_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": self.train_loss,
            },
            path,
        )
        if self.use_wandb:
            wlog.log_artifact(path, "model", filename.replace(".pt", ""))

    def load_model(self, path: Union[str, Path], weights_only_state: bool = True) -> None:
        """Load weights (and optionally optimizer state) from a checkpoint dict."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if not weights_only_state and ckpt.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # --- sweep objective -----------------------------------------------------
    def compute_objective(self) -> Optional[float]:
        """Combine the configured metrics into a scalar sweep objective."""
        obj_cfg = self.cfg.logger.objective
        if obj_cfg is None or not obj_cfg.metrics:
            return None
        reducers = {"last": lambda v: v[-1], "max": max, "min": min,
                    "mean": lambda v: float(np.mean(v))}
        values, weights = [], []
        for spec in obj_cfg.metrics:
            hist = self._metric_history.get(spec.name, [])
            val = reducers[spec.epoch_reduction](hist) if hist else spec.missing_value
            values.append(val)
            weights.append(spec.weight)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if obj_cfg.combination == "sum":
            objective = float(np.sum(weights * values))
        elif obj_cfg.combination == "mean":
            objective = float(np.mean(values))
        else:  # weighted_mean
            denom = np.sum(np.abs(weights)) or 1.0
            objective = float(np.sum(weights * values) / denom)
        if self.use_wandb and wandb.run is not None:
            wandb.run.summary["sweep/objective"] = objective
        return objective


# --- module-level helpers ----------------------------------------------------
def _config_to_dict(cfg: Config) -> dict:
    """Best-effort conversion of the config to a wandb-friendly dict."""
    from dataclasses import asdict

    try:
        return asdict(cfg)
    except Exception:
        return {"seed": cfg.seed}


def _flatten_terms(prefix: str, terms: Dict) -> Dict[str, float]:
    """Flatten the ModularLoss breakdown into ``prefix/term.key`` scalars."""
    flat: Dict[str, float] = {}
    for term_id, payload in terms.items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                flat[f"{prefix}/{term_id}.{key}"] = float(value.detach())
        elif torch.is_tensor(payload):
            flat[f"{prefix}/{term_id}"] = float(payload.detach())
    return flat


def _mean_metric(metric_dicts: List[Dict[str, float]], key: str) -> float:
    vals = [m[key] for m in metric_dicts if key in m and not np.isnan(m[key])]
    return float(np.mean(vals)) if vals else float("nan")
