"""Modular network blocks.

Every block is a ``Block``: it declares ``in_keys`` it reads from a shared
``DataDict`` and ``out_keys`` it writes back (namespaced as ``id.out_key``).
``forward`` handles the dict plumbing; subclasses implement ``forward_block``.
This uniform contract is what lets ``ModularNet`` chain arbitrary blocks
purely from config.

Only general-purpose blocks live here (RNN, linear, multi-input linear,
broadcast, fixed encoder, discrete connectivity). No domain-specific layers.
"""

from math import sqrt
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from modular_ml.configs.init_configs import InitSpecs
from modular_ml.models.initializers import _init_param
from modular_ml.tools.utils import resolve_attr

DataDict = Dict[str, torch.Tensor]


class Block(nn.Module):
    """Base block: shared dict I/O, init, sparsity masks and freezing."""

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.id = id
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._sparsity_mask_names: Dict[str, str] = {}

    # --- dict plumbing -------------------------------------------------------
    def _get_inputs(self, data_dict: DataDict) -> Union[List[torch.Tensor], torch.Tensor]:
        inputs = [data_dict[k] for k in self.in_keys]
        return inputs[0] if len(inputs) == 1 else inputs

    def forward_block(self, inputs):
        raise NotImplementedError

    def forward(self, data_dict: DataDict) -> DataDict:
        """Read ``in_keys``, run ``forward_block``, write ``id.out_key`` entries."""
        inputs = self._get_inputs(data_dict)
        outputs = self.forward_block(inputs)
        for key, out in zip(self.out_keys, outputs):
            data_dict[f"{self.id}.{key}"] = out
        return data_dict

    # --- initialisation / sparsity ------------------------------------------
    def set_parameters(self, init_specs: Optional[Dict[str, InitSpecs]] = None) -> None:
        """Apply per-parameter init specs and register sparsity masks."""
        init_specs = init_specs or {}
        with torch.no_grad():
            for name, param in self.state_dict().items():
                spec = init_specs.get(name, None)
                _init_param(param, spec)
                if spec is None:
                    continue
                if not _requests_sparsity(spec):
                    continue
                mask = (param != 0).to(param.dtype)
                mask_name = f"_sparse_mask__{name.replace('.', '_')}"
                self.register_buffer(mask_name, mask, persistent=True)
                self._sparsity_mask_names[name] = mask_name

    def apply_sparsity_masks(self, apply_grad: bool = True) -> None:
        """Re-zero masked weights (and their grads) to keep sparsity fixed."""
        if not self._sparsity_mask_names:
            return
        for name, param in self.named_parameters():
            mask_name = self._sparsity_mask_names.get(name)
            if mask_name is None:
                continue
            mask = getattr(self, mask_name, None)
            if mask is None or mask.shape != param.shape:
                continue
            param.data.mul_(mask)
            if apply_grad and param.grad is not None:
                param.grad.data.mul_(mask)

    # --- freezing ------------------------------------------------------------
    def freeze(self, freeze_list: Optional[List[str]]) -> None:
        """Freeze the named parameters/submodules."""
        self.set_requires_grad(freeze_list, False)

    def unfreeze(self, unfreeze_list: Optional[List[str]]) -> None:
        """Unfreeze the named parameters/submodules."""
        self.set_requires_grad(unfreeze_list, True)

    def set_requires_grad(self, names: Optional[List[str]], requires_grad: bool) -> None:
        """Set ``requires_grad`` on named parameters or whole submodules."""
        if not names:
            return
        for name in names:
            target = resolve_attr(self, name)
            if target is None:
                raise ValueError(f"'{name}' not found in block '{self.id}'")
            if isinstance(target, nn.Parameter):
                target.requires_grad = requires_grad
            elif isinstance(target, nn.Module):
                for p in target.parameters():
                    p.requires_grad = requires_grad
            else:
                raise TypeError(f"Cannot update '{name}' in block '{self.id}'")


def _requests_sparsity(spec) -> bool:
    """Whether an init spec asks for a persistent sparsity mask."""
    if spec is None:
        return False
    if not getattr(spec, "enforce_sparse", False):
        return False
    return getattr(spec, "sparse_col", 0.0) > 0 or getattr(spec, "sparse_gen", 0.0) > 0


class RNNBlock(Block):
    """Leaky-integrator (vanilla) RNN producing hidden states over time."""

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
        alpha: float,
        bias_in: bool = False,
        bias_rec: bool = False,
        activation: Literal["tanh", "relu"] = "tanh",
        noise: Optional[float] = None,
        hidden_init_val: float = 0.1,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.alpha = alpha
        self.noise_amp = noise or 0.0
        self.activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}.get(activation, nn.Tanh())
        self.linear_ih = nn.Linear(in_dim, out_dim, bias=bias_in)
        self.linear_hh = nn.Linear(out_dim, out_dim, bias=bias_rec)
        self._init_default()
        self.hidden_init_val = hidden_init_val
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def _init_default(self) -> None:
        with torch.no_grad():
            k = sqrt(1.0 / self.out_dim)
            nn.init.uniform_(self.linear_ih.weight, -k, k)
            nn.init.uniform_(self.linear_hh.weight, -k, k)
            if self.linear_ih.bias is not None:
                nn.init.uniform_(self.linear_ih.bias, -k, k)
            if self.linear_hh.bias is not None:
                nn.init.uniform_(self.linear_hh.bias, -k, k)

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Run the RNN over time. Input/Output shape: [T, B, N]."""
        T, B, _ = inputs.shape
        device, dtype = self.linear_ih.weight.device, self.linear_ih.weight.dtype
        x = torch.rand(B, self.out_dim, device=device, dtype=dtype)
        x = (x - 0.5) * 2 * self.hidden_init_val
        h = self.activation(x)
        h_states = torch.zeros(T, B, self.out_dim, device=device, dtype=dtype)
        for t in range(T):
            noise = self.noise_amp * torch.randn_like(x) if self.noise_amp else 0.0
            x = x + self.alpha * (-x + self.linear_ih(inputs[t]) + self.linear_hh(h) + noise)
            h = self.activation(x)
            h_states[t] = h
        return [h_states]


class LinearBlock(Block):
    """Single linear layer with optional activation."""

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        activation: Literal["tanh", "relu", "none"] = "none",
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = {
            "tanh": nn.Tanh(), "relu": nn.ReLU(), "none": nn.Identity(),
        }[activation]
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return [self.activation(self.linear(inputs))]


class FixedEncoderBlock(Block):
    """MLP encoder, optionally frozen and/or L2-normalised."""

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Literal["tanh", "relu", "none"] = "relu",
        bias: bool = True,
        normalize: bool = False,
        trainable: bool = True,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.normalize = normalize
        hidden_dims = hidden_dims or []
        dims = [in_dim, *hidden_dims, out_dim]
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "none": nn.Identity}[activation]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.encoder = nn.Sequential(*layers)
        self.set_parameters(init_specs)
        if trainable:
            self.freeze(freeze_specs)
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        out = self.encoder(inputs)
        if self.normalize:
            out = F.normalize(out, dim=-1)
        return [out]


class BroadcastBlock(Block):
    """Tile the input along the feature dim (no parameters)."""

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
    ) -> None:
        if out_dim % in_dim != 0:
            raise ValueError(
                f"BroadcastBlock '{id}': out_dim ({out_dim}) must be a multiple of "
                f"in_dim ({in_dim})"
            )
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.repeat_factor = out_dim // in_dim

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return [inputs.repeat_interleave(self.repeat_factor, dim=-1)]


class ConnectivityLinear(nn.Module):
    """Linear layer with weights quantised to {-1, 0, +1} (straight-through)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = False,
                 thr: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = scale
        self.thr = thr
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        wq = torch.zeros_like(w)
        wq[w > self.thr] = 1.0
        wq[w < -self.thr] = -1.0
        return wq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = self.quantize(self.weight)
        w_eff = self.scale * (wq + self.weight - self.weight.detach())  # STE
        return x @ w_eff.T + (0 if self.bias is None else self.bias)


class MultiInputLinearBlock(Block):
    """Linear layer over several concatenated inputs with per-input modes.

    Each input slice has a mode: ``active`` (learnable), ``freeze`` (connected
    but detached), or ``disconnect`` (zeroed). Modes can be changed at runtime
    via :meth:`set_input_modes` (used by the runtime schedule).
    """

    _MODES = ("active", "disconnect", "freeze")

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dims: List[int],
        out_dim: int,
        input_modes: Optional[List[str]] = None,
        bias: bool = False,
        linear_type: Literal["base", "connectivity"] = "base",
        activation: Literal["tanh", "relu", "none"] = "none",
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        self.in_dims = list(in_dims)
        in_dim = sum(in_dims)
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        if linear_type == "connectivity":
            self.linear = ConnectivityLinear(in_dim, out_dim, bias)
        else:
            self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = {
            "tanh": nn.Tanh(), "relu": nn.ReLU(), "none": nn.Identity(),
        }[activation]
        self.input_modes = self._resolve_modes(input_modes)
        self._build_mask(self.input_modes)
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def _resolve_modes(self, modes: Optional[List[str]]) -> List[str]:
        if modes is None:
            return ["active"] * len(self.in_dims)
        assert len(modes) == len(self.in_dims), "input_modes must match in_keys length"
        bad = set(modes) - set(self._MODES)
        if bad:
            raise ValueError(f"input_modes must be in {self._MODES}, got {sorted(bad)}")
        return list(modes)

    def _build_mask(self, modes: List[str]) -> None:
        active, frozen = [], []
        for mode, dim in zip(modes, self.in_dims):
            active.append(torch.ones(dim) if mode == "active" else torch.zeros(dim))
            frozen.append(torch.ones(dim) if mode == "freeze" else torch.zeros(dim))
        # Column masks [in_dim] broadcast over output rows.
        self.register_buffer("mask_active", torch.cat(active), persistent=False)
        self.register_buffer("mask_frozen", torch.cat(frozen), persistent=False)

    def set_input_modes(self, modes: Optional[List[str]]) -> None:
        """Update per-input connectivity modes at runtime."""
        self.input_modes = self._resolve_modes(modes)
        self._build_mask(self.input_modes)

    def forward_block(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        x = torch.cat(inputs, dim=-1)
        w = self.linear.weight
        # Active columns learn; frozen columns contribute with detached weights.
        w_eff = w * self.mask_active + w.detach() * self.mask_frozen
        out = F.linear(x, w_eff, self.linear.bias)
        return [self.activation(out)]
