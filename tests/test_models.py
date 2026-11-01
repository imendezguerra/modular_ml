"""Model construction, forward shapes, init/freeze and runtime schedule."""

import pytest
import torch

from modular_ml.configs.init_configs import InitNormalRec
from modular_ml.configs.model_configs import ArchitectureConfig, BlockSpec, ModelConfig
from modular_ml.models.blocks import Conv1dBlock, LinearBlock, RecurrentBlock, TransformerBlock
from modular_ml.models.network import build_modular_model


def _rnn_model() -> ModelConfig:
    arch = ArchitectureConfig(
        modules={
            "rnn": BlockSpec(
                _block_target_="modular_ml.models.blocks.RNNBlock",
                id="rnn",
                in_keys=["stim"],
                out_keys=["activity"],
                in_dim=3,
                out_dim=8,
                params={"alpha": 0.2, "activation": "tanh"},
                init={"linear_hh.weight": InitNormalRec(g=1.2)},
                freeze=["linear_hh.weight"],
            ),
            "readout": BlockSpec(
                _block_target_="modular_ml.models.blocks.LinearBlock",
                id="readout",
                in_keys=["rnn.activity"],
                out_keys=["prediction"],
                in_dim=8,
                out_dim=2,
            ),
        }
    )
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


def test_init_applied_once_per_block(monkeypatch):
    from modular_ml.models.blocks import Block

    calls = []
    original = Block.set_parameters

    def spy(self, specs=None):
        calls.append(self.id)
        original(self, specs)

    monkeypatch.setattr(Block, "set_parameters", spy)
    build_modular_model(_rnn_model())
    assert calls == ["rnn", "readout"]  # each block once, in its constructor


def test_parameter_free_block_from_config():
    arch = ArchitectureConfig(
        modules={
            "tile": BlockSpec(
                _block_target_="modular_ml.models.blocks.BroadcastBlock",
                id="tile",
                in_keys=["x"],
                out_keys=["y"],
                in_dim=2,
                out_dim=6,
            )
        }
    )
    net = build_modular_model(ModelConfig(architecture=arch))
    assert net({"x": torch.randn(4, 2)})["tile.y"].shape == (4, 6)


def test_multi_input_modes():
    from modular_ml.models.blocks import MultiInputLinearBlock

    block = MultiInputLinearBlock(
        id="mix",
        in_keys=["a", "b"],
        out_keys=["y"],
        in_dims=[3, 2],
        out_dim=4,
        input_modes=["active", "disconnect"],
    )
    out = block({"a": torch.randn(5, 3), "b": torch.randn(5, 2)})
    assert out["mix.y"].shape == (5, 4)
    # Disconnected columns are masked out.
    assert torch.allclose(block.mask_active[3:], torch.zeros(2))


@pytest.mark.parametrize("cell", ["gru", "lstm", "rnn"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_recurrent_block_shapes(cell, bidirectional):
    block = RecurrentBlock(
        "r", ["x"], ["seq", "last"], 3, 8, cell=cell, num_layers=2, bidirectional=bidirectional
    )
    out = block({"x": torch.randn(10, 4, 3)})
    assert out["r.seq"].shape == (10, 4, 8)
    assert out["r.last"].shape == (4, 8)


def test_recurrent_block_freeze_by_torch_name():
    block = RecurrentBlock("r", ["x"], ["seq"], 3, 8, freeze_specs=["rnn.weight_hh_l0"])
    assert block.rnn.weight_hh_l0.requires_grad is False
    assert block.rnn.weight_ih_l0.requires_grad is True


@pytest.mark.parametrize("kernel_size", [2, 3])
def test_conv1d_block_preserves_time_and_is_causal(kernel_size):
    block = Conv1dBlock("c", ["x"], ["y"], 3, 5, kernel_size, hidden_dims=[6], causal=True)
    x = torch.randn(12, 2, 3)
    y = block({"x": x})["c.y"]
    assert y.shape == (12, 2, 5)
    # Changing the last step must not affect earlier outputs.
    x2 = x.clone()
    x2[-1] += 1.0
    y2 = block({"x": x2})["c.y"]
    assert torch.allclose(y[:-1], y2[:-1])


def test_transformer_block_causal():
    block = TransformerBlock("t", ["x"], ["y"], 3, 8, nhead=2, num_layers=1, causal=True)
    block.eval()
    x = torch.randn(7, 2, 3)
    y = block({"x": x})["t.y"]
    assert y.shape == (7, 2, 8)
    x2 = x.clone()
    x2[-1] += 1.0
    assert torch.allclose(y[:-1], block({"x": x2})["t.y"][:-1], atol=1e-6)


def test_dropout_active_in_train_only():
    block = LinearBlock("l", ["x"], ["y"], 16, 4, dropout=0.5)
    x = torch.randn(32, 16)
    block.train()
    assert not torch.allclose(block({"x": x})["l.y"], block({"x": x})["l.y"])
    block.eval()
    assert torch.allclose(block({"x": x})["l.y"], block({"x": x})["l.y"])


def test_unknown_activation_raises():
    with pytest.raises(KeyError):
        LinearBlock("l", ["x"], ["y"], 2, 2, activation="sigmoid")
