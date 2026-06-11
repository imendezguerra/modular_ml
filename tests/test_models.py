"""Model construction, forward shapes, init/freeze and runtime schedule."""

import torch

from modular_ml.configs.init_configs import InitNormalRec
from modular_ml.configs.model_configs import ArchitectureConfig, BlockSpec, ModelConfig
from modular_ml.models.network import build_modular_model


def _rnn_model() -> ModelConfig:
    arch = ArchitectureConfig(modules={
        "rnn": BlockSpec(
            _block_target_="modular_ml.models.blocks.RNNBlock", id="rnn",
            in_keys=["stim"], out_keys=["activity"], in_dim=3, out_dim=8,
            params={"alpha": 0.2, "activation": "tanh"},
            init={"linear_hh.weight": InitNormalRec(g=1.2)},
            freeze=["linear_hh.weight"],
        ),
        "readout": BlockSpec(
            _block_target_="modular_ml.models.blocks.LinearBlock", id="readout",
            in_keys=["rnn.activity"], out_keys=["prediction"], in_dim=8, out_dim=2,
        ),
    })
    return ModelConfig(n_inputs=3, n_outputs=2, in_key="stim", architecture=arch, alpha=0.2)


def test_build_and_forward_shapes():
    net = build_modular_model(_rnn_model())
    out = net({"stim": torch.randn(10, 4, 3)})
    assert out["rnn.activity"].shape == (10, 4, 8)
    assert out["readout.prediction"].shape == (10, 4, 2)


def test_freeze_applied_from_spec():
    net = build_modular_model(_rnn_model())
    rnn = net.get_module("rnn")
    assert rnn.linear_hh.weight.requires_grad is False
    assert rnn.linear_ih.weight.requires_grad is True


def test_runtime_schedule_unfreezes():
    net = build_modular_model(_rnn_model())
    applied = net.apply_runtime_schedule({"modules": {"rnn": {"unfreeze": ["linear_hh.weight"]}}})
    assert applied["rnn"]["unfreeze"] == ["linear_hh.weight"]
    assert net.get_module("rnn").linear_hh.weight.requires_grad is True


def test_multi_input_modes():
    from modular_ml.models.blocks import MultiInputLinearBlock

    block = MultiInputLinearBlock(
        id="mix", in_keys=["a", "b"], out_keys=["y"], in_dims=[3, 2], out_dim=4,
        input_modes=["active", "disconnect"],
    )
    out = block({"a": torch.randn(5, 3), "b": torch.randn(5, 2)})
    assert out["mix.y"].shape == (5, 4)
    # Disconnected columns are masked out.
    assert torch.allclose(block.mask_active[3:], torch.zeros(2))
