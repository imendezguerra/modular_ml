"""Modular loss: criteria, scheduling, weighting, parameter regularisers."""

import torch

from modular_ml.configs.model_configs import ArchitectureConfig, BlockSpec, ModelConfig
from modular_ml.configs.optim_configs import LossSpec, OptimizerConfig
from modular_ml.losses.loss import _schedule_weight, build_modular_loss
from modular_ml.models.network import build_modular_model


def test_schedule_weight_linear_ramp():
    cfg = {"type": "linear_ramp", "start_epoch": 0, "end_epoch": 10,
           "start_value": 0.0, "end_value": 1.0}
    assert _schedule_weight(cfg, 0) == 0.0
    assert abs(_schedule_weight(cfg, 5) - 0.5) < 1e-9
    assert _schedule_weight(cfg, 10) == 1.0
    # Schedules apply in training only by default.
    assert _schedule_weight(cfg, 5, mode="test") == 1.0


def test_supervised_and_regulariser_terms():
    ocfg = OptimizerConfig(losses={
        "mse": LossSpec(id="mse", criterion="mse", in_keys=["pred"], label_key="y"),
        "smooth": LossSpec(id="smooth", criterion="temporal_smoothness",
                           in_keys=["act"], reg_weight=0.5),
    })
    loss = build_modular_loss(ocfg)
    data = {"pred": torch.randn(6, 4, 1), "act": torch.randn(6, 4, 8)}
    total, terms = loss(data, {"y": torch.randn(6, 4, 1)}, epoch=0)
    assert total.dim() == 0
    assert "mse" in terms and "smooth" in terms


def test_requires_model_param_regulariser():
    arch = ArchitectureConfig(modules={
        "lin": BlockSpec(_block_target_="modular_ml.models.blocks.LinearBlock", id="lin",
                         in_keys=["x"], out_keys=["y"], in_dim=4, out_dim=2),
    })
    net = build_modular_model(ModelConfig(n_inputs=4, n_outputs=2, in_key="x", architecture=arch))
    loss = build_modular_loss(OptimizerConfig(losses={
        "w": LossSpec(id="w", criterion="norm2", in_keys=["lin"],
                      requires_model=True, reg_weight=1.0),
    }))
    out = net({"x": torch.randn(5, 4)})
    total, terms = loss(out, model=net)
    assert total.item() > 0
    assert "lin" in terms["w"]


def test_cross_entropy_criterion():
    loss = build_modular_loss(OptimizerConfig(losses={
        "ce": LossSpec(id="ce", criterion="cross_entropy", in_keys=["logits"], label_key="lab"),
    }))
    logits = torch.randn(8, 3)
    total, _ = loss({"logits": logits}, {"lab": torch.randint(0, 3, (8,))})
    assert total.item() > 0
