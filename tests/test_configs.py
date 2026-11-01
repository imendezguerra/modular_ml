"""Config composition + instantiation tests."""

import pytest
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError

from modular_ml.configs.main_configs import Config, instantiate_config, load_config, merge_overrides
from modular_ml.constants import CONFIG_DIR


def test_sequential_config_instantiates():
    cfg = load_config(CONFIG_DIR / "runs" / "config_sequential.yaml")
    assert isinstance(cfg, Config)
    assert cfg.model.n_inputs == 2
    assert cfg.model.n_outputs == 1
    # alpha = dt_s / tau_s resolved via the `divide` resolver (0.01 / 0.05).
    assert abs(cfg.model.alpha - 0.2) < 1e-9
    assert list(cfg.optim.losses.keys()) == ["mse", "smooth"]
    assert cfg.logger.objective.metrics[0].name == "val/corrcoef"


def test_tabular_config_instantiates():
    cfg = load_config(CONFIG_DIR / "runs" / "config_tabular.yaml")
    assert cfg.model.n_outputs == 4
    enc = cfg.model.architecture.modules["encoder"]
    # Interpolation into a list must preserve the int type.
    assert enc.params["hidden_dims"] == [64]
    assert isinstance(enc.params["hidden_dims"][0], int)


def test_uninstantiated_returns_dictconfig():
    raw = load_config(CONFIG_DIR / "runs" / "config_sequential.yaml", instantiate=False)
    assert isinstance(raw, DictConfig)


@pytest.mark.parametrize("model", ["model_rnn", "model_gru", "model_transformer"])
def test_sequential_models_are_interchangeable(model):
    cfg = load_config(CONFIG_DIR / "runs" / "config_sequential.yaml", overrides=[f"model={model}"])
    # Shared optim / processing / logger configs follow the model's hidden key.
    hidden = cfg.model.hidden_key
    assert cfg.optim.losses["smooth"].in_keys == [hidden]
    assert cfg.processing.pca_keys == [hidden]


def test_merge_overrides_dotted_keys():
    base = load_config(
        CONFIG_DIR / "runs" / "config_sequential.yaml", resolve=False, instantiate=False
    )
    raw = merge_overrides(base, {"optim.lr": 0.123, "model.architecture.global_params.g": 2.0})
    cfg = instantiate_config(raw)
    assert cfg.optim.lr == 0.123
    # Applied before resolution, so interpolations pick it up.
    assert cfg.model.architecture.modules["rnn"].init["linear_hh.weight"].g == 2.0
    with pytest.raises(ConfigAttributeError):  # typo in a sweep key
        merge_overrides(base, {"optim.lrr": 1.0})
