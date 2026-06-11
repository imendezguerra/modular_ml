"""Custom OmegaConf resolvers used inside the YAML configs.

These let configs express derived values directly, e.g.::

    alpha: ${divide:${model.dt_s},${model.tau_s}}
    out_dim: ${multiply:${...dim_base},${...dim_ratio}}
    layers: ${list_repeat:128,3}

Resolvers are registered on import (idempotently) so that ``load_config`` works
no matter which module triggers the import first.
"""

from omegaconf import OmegaConf


def _register(name: str, fn) -> None:
    """Register a resolver once, tolerating re-imports."""
    try:
        OmegaConf.register_new_resolver(name, fn)
    except ValueError:
        # Already registered (e.g. the module was imported twice).
        pass


_register("divide", lambda a, b: float(a) / float(b))
_register("multiply", lambda a, b: a * b)
_register("list_repeat", lambda x, n: OmegaConf.create([x] * int(n)))
