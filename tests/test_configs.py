"""Config composition + instantiation tests."""

from omegaconf import DictConfig

from modular_ml.configs.main_configs import Config, load_config
from modular_ml.constants import CONFIG_DIR


def test_sequential_config_instantiates():
    cfg = load_config(CONFIG_DIR / "runs" / "config_sequential.yaml")
    assert isinstance(cfg, Config)
    assert cfg.model.n_inputs == 2
    assert cfg.model.n_outputs == 1
    # alpha = dt_s / tau_s resolved via the `divide` resolver (0.01 / 0.05).
    assert abs(cfg.model.alpha - 0.2) < 1e-9
    assert list(cfg.optim.losses.keys()) == ["mse", "smooth"]
    assert cfg.logger.objective.metrics[0].name == "test/corrcoef"


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
