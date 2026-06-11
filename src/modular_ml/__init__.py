"""modular_ml: a Hydra + dataclasses template for modular, wandb-tracked ML.

Importing the package registers the custom OmegaConf resolvers used inside the
YAML configs (``divide``, ``multiply``, ``list_repeat``) so that ``load_config``
works from anywhere.
"""

from modular_ml.tools import hydra_utils as _hydra_utils  # noqa: F401  (registers resolvers)

__all__ = ["__version__"]
__version__ = "0.1.0"
