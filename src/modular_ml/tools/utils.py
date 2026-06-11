"""Small shared utilities."""

from typing import Any, Optional

import torch.nn as nn


def resolve_attr(module: nn.Module, name: str) -> Optional[Any]:
    """Resolve a possibly-dotted ``name`` to a parameter or submodule.

    Tries ``named_parameters`` first (so e.g. ``linear.weight`` returns the
    Parameter), then falls back to ``get_submodule`` for whole submodules.
    Returns None if nothing matches.
    """
    params = dict(module.named_parameters())
    if name in params:
        return params[name]
    try:
        return module.get_submodule(name)
    except AttributeError:
        return None
