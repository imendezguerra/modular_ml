"""Modular loss: criteria, scheduling, weighting, parameter regularisers."""

import pytest
import torch

from modular_ml.configs.enums import LossCriterion
from modular_ml.configs.model_configs import ArchitectureConfig, BlockSpec, ModelConfig
from modular_ml.configs.optim_configs import LossSpec, OptimizerConfig
from modular_ml.losses.loss import LossModule, _schedule_weight, build_modular_loss
from modular_ml.models.network import build_modular_model


def test_schedule_weight_linear_ramp():
    cfg = {
        "type": "linear_ramp",
        "start_epoch": 0,
        "end_epoch": 10,
        "start_value": 0.0,
        "end_value": 1.0,
    }
    assert _schedule_weight(cfg, 0) == 0.0
    assert abs(_schedule_weight(cfg, 5) - 0.5) < 1e-9
    assert _schedule_weight(cfg, 10) == 1.0
    # Schedules apply in training only by default.
    assert _schedule_weight(cfg, 5, mode="test") == 1.0


def test_supervised_and_regulariser_terms():
    ocfg = OptimizerConfig(
        losses={
            "mse": LossSpec(id="mse", criterion="mse", in_keys=["pred"], label_key="y"),
            "smooth": LossSpec(
                id="smooth", criterion="temporal_smoothness", in_keys=["act"], reg_weight=0.5
            ),
        }
    )
    loss = build_modular_loss(ocfg)
    data = {"pred": torch.randn(6, 4, 1), "act": torch.randn(6, 4, 8)}
    total, terms = loss(data, {"y": torch.randn(6, 4, 1)}, epoch=0)
    assert total.dim() == 0
    assert "mse" in terms and "smooth" in terms


def test_requires_model_param_regulariser():
    arch = ArchitectureConfig(
        modules={
            "lin": BlockSpec(
                _block_target_="modular_ml.models.blocks.LinearBlock",
                id="lin",
                in_keys=["x"],
                out_keys=["y"],
                in_dim=4,
                out_dim=2,
            ),
        }
    )
    net = build_modular_model(ModelConfig(n_inputs=4, n_outputs=2, in_key="x", architecture=arch))
    loss = build_modular_loss(
        OptimizerConfig(
            losses={
                "w": LossSpec(
                    id="w", criterion="norm2", in_keys=["lin"], requires_model=True, reg_weight=1.0
                ),
            }
        )
    )
    out = net({"x": torch.randn(5, 4)})
    total, terms = loss(out, model=net)
    assert total.item() > 0
    assert "lin" in terms["w"]


def test_cross_entropy_criterion():
    loss = build_modular_loss(
        OptimizerConfig(
            losses={
                "ce": LossSpec(
                    id="ce", criterion="cross_entropy", in_keys=["logits"], label_key="lab"
                ),
            }
        )
    )
    logits = torch.randn(8, 3)
    total, _ = loss({"logits": logits}, {"lab": torch.randint(0, 3, (8,))})
    assert total.item() > 0


def test_criterion_enum_matches_table():
    assert {c.value for c in LossCriterion} == set(LossModule.CRITERION_FUNCS)


@pytest.mark.parametrize(
    "criterion, labels, params",
    [
        ("huber", torch.randn(6, 4, 2), {"delta": 0.5}),
        ("bce_with_logits", torch.randint(0, 2, (6, 4, 2)), {"pos_weight": [1.0, 3.0]}),
        ("cosine", torch.randn(6, 4, 2), {}),
    ],
)
def test_new_supervised_criteria(criterion, labels, params):
    loss = build_modular_loss(
        OptimizerConfig(
            losses={
                "t": LossSpec(
                    id="t", criterion=criterion, in_keys=["p"], label_key="y", params=params
                )
            }
        )
    )
    total, _ = loss({"p": torch.randn(6, 4, 2)}, {"y": labels})
    assert total.dim() == 0 and torch.isfinite(total)


def test_criterion_params_reach_the_function():
    logits, labels = torch.randn(8, 3), torch.randint(0, 3, (8,))
    spec = {"id": "ce", "criterion": "cross_entropy", "in_keys": ["z"], "label_key": "y"}
    plain = build_modular_loss(OptimizerConfig(losses={"ce": LossSpec(**spec)}))
    smooth = build_modular_loss(
        OptimizerConfig(
            losses={"ce": LossSpec(**spec, params={"label_smoothing": 0.2, "weight": [1, 2, 3]})}
        )
    )
    assert plain({"z": logits}, {"y": labels})[0] != smooth({"z": logits}, {"y": labels})[0]
