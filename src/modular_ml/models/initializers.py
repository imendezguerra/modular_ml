"""Apply :class:`InitSpecs` to parameter tensors.

``_init_param`` is the dispatch entry point used by ``Block.set_parameters``.
It supports whole-tensor methods (uniform/normal/normal_rec/fill) plus the
``by_inputs`` method, which slices a 2D weight along its input (column) axis and
initialises each chunk with its own nested spec.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from modular_ml.configs.init_configs import InitSpecs


def _attr(spec: Optional[Any], name: str, default: Any) -> Any:
    """Read an attribute from a dataclass spec or a plain dict, with fallback."""
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _apply_init(tensor: torch.Tensor, method: Optional[str], spec: Any) -> None:
    """Initialise ``tensor`` in place according to ``method`` and apply sparsity."""
    if method is None:
        return
    if method == "uniform":
        nn.init.uniform_(tensor, _attr(spec, "scale_min", -0.1), _attr(spec, "scale_max", 0.1))
    elif method == "normal":
        nn.init.normal_(tensor, _attr(spec, "mean", 0.0), _attr(spec, "std", 0.05))
    elif method == "normal_rec":
        g = _attr(spec, "g", 1.2)
        n = tensor.shape[0]
        nn.init.normal_(tensor, 0.0, g / (n ** 0.5))
    elif method == "fill":
        tensor.fill_(_attr(spec, "value", 0.0))
    else:
        raise ValueError(f"Unknown init method: {method}")

    # Optional column-wise sparsity (zero out whole input columns).
    sparse_col = _attr(spec, "sparse_col", 0.0)
    if sparse_col > 0 and tensor.ndim == 2:
        with torch.no_grad():
            mask = (torch.rand(tensor.shape[1]) > sparse_col).to(tensor.dtype)
            tensor *= mask

    # Optional element-wise sparsity.
    sparse_gen = _attr(spec, "sparse_gen", 0.0)
    if sparse_gen > 0:
        with torch.no_grad():
            mask = (torch.rand_like(tensor) > sparse_gen).to(tensor.dtype)
            tensor *= mask


def _init_param(tensor: torch.Tensor, spec: Optional[InitSpecs]) -> None:
    """Initialise ``tensor`` according to ``spec`` (no-op when spec is None)."""
    if spec is None:
        return
    method = _attr(spec, "method", None)

    if method == "by_inputs":
        offset = 0
        for chunk in _attr(spec, "chunks", []) or []:
            in_dim = _attr(chunk, "dim", None)
            in_spec = _attr(chunk, "spec", None)
            if in_dim is None or in_spec is None:
                raise ValueError("InitByInputs requires dim and spec per chunk.")
            if tensor.ndim == 2:
                sub = tensor[:, offset:offset + in_dim]
            else:
                sub = tensor[offset:offset + in_dim]
            _apply_init(sub, _attr(in_spec, "method", None), in_spec)
            offset += in_dim
    else:
        _apply_init(tensor, method, spec)
