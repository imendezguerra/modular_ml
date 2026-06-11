"""Modular loss: a config-built, weighted, schedulable sum of loss terms.

``build_modular_loss`` instantiates one :class:`LossModule` per ``LossSpec``.
Each module reads its ``in_keys`` from the data dict (or, when ``requires_model``
is set, reads a module's parameters), applies a criterion, and returns a per-key
dict. :class:`ModularLoss` weights each term by ``reg_weight * use_reg *
schedule(epoch)`` and sums them.
"""

import bisect
import math
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import hydra
import torch
from torch import nn

from modular_ml.configs.optim_configs import LossSpec, OptimizerConfig
from modular_ml.losses.regularisers import (
    _cross_entropy,
    _effective_rank,
    _monotonicity_loss,
    _norm_per_row,
    _temporal_smoothness_loss,
    _trend_loss,
)
from modular_ml.models.blocks import Block
from modular_ml.models.network import ModularNet

DataDict = Dict[str, torch.Tensor]


def _schedule_weight(cfg: Optional[Dict], epoch: Optional[int], mode: str = "train") -> float:
    """Return the scalar schedule weight for one loss term at ``epoch``.

    Supported ``type`` values: ``constant`` (default), ``linear_ramp``,
    ``cosine``, ``piecewise``. Schedules apply in training only unless the
    config sets ``train_only: false``.
    """
    cfg = cfg or {}
    if mode != "train" and bool(cfg.get("train_only", True)):
        return 1.0

    e = int(epoch) if epoch is not None else 0
    start = int(cfg.get("start_epoch", 0))
    if e < start:
        return 0.0
    sched = str(cfg.get("type", "constant")).lower()

    if sched == "constant":
        return float(cfg.get("value", 1.0))

    if sched in ("linear_ramp", "cosine"):
        end = cfg.get("end_epoch", None)
        hold = bool(cfg.get("hold_after_end", True))
        v0 = float(cfg.get("start_value", 0.0))
        v1 = float(cfg.get("end_value", 1.0))
        if end is None or int(end) <= start:
            return v1
        if e > int(end):
            return v1 if hold else 0.0
        t = min(max((e - start) / float(int(end) - start), 0.0), 1.0)
        if sched == "cosine":
            t = 0.5 * (1.0 - math.cos(math.pi * t))
        return v0 + t * (v1 - v0)

    if sched == "piecewise":
        milestones = [int(x) for x in cfg.get("milestones", [])]
        values = [float(x) for x in cfg.get("values", [])]
        if len(values) != len(milestones) + 1:
            raise ValueError("piecewise schedule needs len(values) == len(milestones)+1")
        return values[bisect.bisect_right(milestones, e)]

    raise ValueError(f"Unknown loss schedule type: '{sched}'")


class LossModule(nn.Module):
    """A single loss term applied to one or more data-dict keys."""

    CRITERION_FUNCS: ClassVar[dict] = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "cross_entropy": _cross_entropy,
        "norm2": lambda x: x.norm(2),
        "norm_row": _norm_per_row,
        "pow2": lambda x: x.pow(2).mean(),
        "var": lambda x: x.var(),
        "monotonicity": _monotonicity_loss,
        "trend": _trend_loss,
        "temporal_smoothness": _temporal_smoothness_loss,
        "effective_rank": _effective_rank,
    }
    SUPERVISED: ClassVar[set] = {"mse", "l1", "cross_entropy"}

    def __init__(
        self,
        id: str,
        criterion: str,
        in_keys: List[str],
        label_key: Optional[str],
        requires_model: bool,
        reg_weight: float,
        use_reg: float = 1.0,
    ) -> None:
        super().__init__()
        # Accept either a LossCriterion enum or a plain string.
        criterion = getattr(criterion, "value", criterion)
        if criterion not in self.CRITERION_FUNCS:
            raise ValueError(f"Unknown criterion '{criterion}'")
        self.id = id
        self.criterion = criterion
        self.in_keys = in_keys
        self.requires_model = requires_model
        self.reg_weight = reg_weight
        self.use_reg = use_reg
        self.label_key = label_key if criterion in self.SUPERVISED else None
        self.schedule: Dict = {}

    @property
    def criterion_fun(self):
        return self.CRITERION_FUNCS[self.criterion]

    def _apply_criterion(self, x, labels=None):
        fn = self.criterion_fun
        return fn(x, labels) if labels is not None else fn(x)

    def forward(
        self,
        data_dict: DataDict,
        label_dict: Optional[DataDict] = None,
        model: Optional[ModularNet] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute this term for each ``in_key``, returning a per-key dict."""
        labels = None
        if self.label_key is not None:
            labels = (label_dict or {}).get(self.label_key)
            if labels is None:
                labels = data_dict.get(self.label_key)

        out: Dict[str, torch.Tensor] = {}
        if self.requires_model:
            if model is None:
                raise ValueError(f"Loss '{self.id}' requires the model.")
            for key in self.in_keys:
                module: Block = model.get_module(key)
                terms = [self._apply_criterion(p) for _, p in module.named_parameters()]
                out[key] = torch.stack(terms).sum()
        else:
            for key in self.in_keys:
                out[key] = self._apply_criterion(data_dict[key], labels)
        return out


class ModularLoss(nn.Module):
    """Weighted, schedulable sum of :class:`LossModule` terms."""

    def __init__(self, loss_modules: List[LossModule], loss_ids: List[str]) -> None:
        super().__init__()
        self.loss_modules = nn.ModuleList(loss_modules)
        self.loss_ids = loss_ids

    def forward(
        self,
        data_dict: DataDict,
        label_dict: Optional[DataDict] = None,
        model: Optional[ModularNet] = None,
        epoch: Optional[int] = None,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, Union[Dict, torch.Tensor]]]:
        """Return (total_loss, per-term breakdown for logging)."""
        terms: Dict[str, Union[Dict, torch.Tensor]] = {}
        weighted: List[torch.Tensor] = []
        for lid, module in zip(self.loss_ids, self.loss_modules):
            block = module(data_dict, label_dict, model)
            terms[lid] = block
            mean_loss = torch.stack(list(block.values())).mean()
            sched_w = _schedule_weight(module.schedule, epoch, mode)
            weighted.append(module.reg_weight * module.use_reg * sched_w * mean_loss)
            terms[f"schedules/{lid}"] = torch.as_tensor(
                sched_w, device=mean_loss.device, dtype=mean_loss.dtype
            )
        total = torch.stack(weighted).sum()
        return total, terms


def _instantiate_loss(spec: LossSpec, idx: int) -> Tuple[str, LossModule]:
    """Hydra-instantiate one loss term from its spec."""
    kwargs: Dict = {
        "_target_": spec._loss_target_,
        "id": spec.id or f"loss_{idx}",
        "criterion": spec.criterion,
        "in_keys": spec.in_keys,
        "label_key": spec.label_key,
        "requires_model": spec.requires_model,
        "reg_weight": spec.reg_weight,
        "use_reg": spec.use_reg,
    }
    kwargs.update(getattr(spec, "params", {}) or {})
    module = hydra.utils.instantiate(kwargs)
    module.schedule = getattr(spec, "schedule", {}) or {}
    return kwargs["id"], module


def build_modular_loss(optim_cfg: OptimizerConfig) -> ModularLoss:
    """Build a ``ModularLoss`` from an optimizer config's loss dict."""
    specs: List[LossSpec] = list(optim_cfg.losses.values())
    if not specs:
        raise ValueError("OptimizerConfig must define at least one loss.")
    modules, ids = [], []
    for idx, spec in enumerate(specs):
        lid, module = _instantiate_loss(spec, idx)
        ids.append(lid)
        modules.append(module)
    return ModularLoss(modules, ids)
