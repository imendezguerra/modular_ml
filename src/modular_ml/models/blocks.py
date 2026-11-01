"""Modular network blocks.

Every block is a ``Block``: it declares ``in_keys`` it reads from a shared
``DataDict`` and ``out_keys`` it writes back (namespaced as ``id.out_key``).
``forward`` handles the dict plumbing; subclasses implement ``forward_block``.
This uniform contract is what lets ``ModularNet`` chain arbitrary blocks
purely from config. Each block applies its config ``init`` and ``freeze`` specs
once, at the end of its constructor, via :meth:`Block.set_parameters` and
:meth:`Block.freeze`.

Only general-purpose blocks live here (leaky RNN, GRU/LSTM, linear, MLP
encoder, temporal conv, transformer, multi-input linear, broadcast, discrete
connectivity). No domain-specific layers. Sequence blocks use ``[T, B, F]``.
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
Activation = Literal["tanh", "relu", "gelu", "none"]

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "none": nn.Identity}
_RNN_CELLS = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}


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
        activation: Activation = "tanh",
        noise: Optional[float] = None,
        hidden_init_val: float = 0.1,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.alpha = alpha
        self.noise_amp = noise or 0.0
        self.activation = _ACTIVATIONS[activation]()
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


class RecurrentBlock(Block):
    """PyTorch GRU / LSTM / Elman RNN; outputs ``[sequence, final hidden]``.

    ``out_dim`` is the output feature size (split across directions when
    ``bidirectional``). ``dropout`` applies between stacked layers.
    """

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
        cell: Literal["gru", "lstm", "rnn"] = "gru",
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        dropout: float = 0.0,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.n_dir = 2 if bidirectional else 1
        if out_dim % self.n_dir != 0:
            raise ValueError(f"RecurrentBlock '{id}': out_dim must be even when bidirectional")
        self.rnn = _RNN_CELLS[cell](
            in_dim,
            out_dim // self.n_dir,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Input ``[T, B, in_dim]`` -> ``[T, B, out_dim]`` and last-layer state ``[B, out_dim]``."""
        out, h_n = self.rnn(inputs)
        if isinstance(h_n, tuple):  # LSTM returns (h, c)
            h_n = h_n[0]
        return [out, torch.cat(list(h_n[-self.n_dir :]), dim=-1)]


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
        activation: Activation = "none",
        dropout: float = 0.0,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)  # on the input
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = _ACTIVATIONS[activation]()
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return [self.activation(self.linear(self.dropout(inputs)))]


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
        activation: Activation = "relu",
        bias: bool = True,
        normalize: bool = False,
        trainable: bool = True,
        dropout: float = 0.0,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.normalize = normalize
        hidden_dims = hidden_dims or []
        dims = [in_dim, *hidden_dims, out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:
                # One parameter-free slot keeps Linear names at encoder.0, encoder.2, ...
                layers.append(nn.Sequential(_ACTIVATIONS[activation](), nn.Dropout(dropout)))
        self.encoder = nn.Sequential(*layers)
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        out = self.encoder(inputs)
        if self.normalize:
            out = F.normalize(out, dim=-1)
        return [out]


class Conv1dBlock(Block):
    """Stack of 1D convolutions over time; ``[T, B, F]`` in and out, T preserved.

    ``causal`` pads on the left only, so step t never sees later steps.
    ``dropout`` applies after each hidden activation.
    """

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        hidden_dims: Optional[List[int]] = None,
        activation: Activation = "relu",
        bias: bool = True,
        causal: bool = False,
        dropout: float = 0.0,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        dims = [in_dim, *(hidden_dims or []), out_dim]
        self.convs = nn.ModuleList(
            nn.Conv1d(dims[i], dims[i + 1], kernel_size, bias=bias) for i in range(len(dims) - 1)
        )
        left = kernel_size - 1 if causal else (kernel_size - 1) // 2
        self.pad = (left, kernel_size - 1 - left)
        self.activation = _ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout)
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        x = inputs.permute(1, 2, 0)  # [T, B, F] -> [B, F, T]
        for i, conv in enumerate(self.convs):
            x = conv(F.pad(x, self.pad))
            if i < len(self.convs) - 1:
                x = self.dropout(self.activation(x))
        return [x.permute(2, 0, 1)]


class TransformerBlock(Block):
    """Input projection + sinusoidal positions + ``nn.TransformerEncoder``.

    Works on ``[T, B, F]``; ``out_dim`` is the model width (divisible by
    ``nhead``). ``causal`` masks attention to later steps.
    """

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        activation: Literal["relu", "gelu"] = "gelu",
        norm_first: bool = True,
        causal: bool = False,
        dropout: float = 0.0,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.causal = causal
        self.proj = nn.Linear(in_dim, out_dim)
        layer = nn.TransformerEncoderLayer(
            out_dim,
            nhead,
            dim_feedforward or 4 * out_dim,
            dropout,
            activation,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers,
            norm=nn.LayerNorm(out_dim) if norm_first else None,
            enable_nested_tensor=False,
        )
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        x = self.proj(inputs)
        T, _, d = x.shape
        x = x + _sinusoidal_positions(T, d, x.device, x.dtype)
        mask = None
        if self.causal:
            mask = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype).triu(1)
        return [self.encoder(x, mask=mask)]


def _sinusoidal_positions(T: int, d: int, device, dtype) -> torch.Tensor:
    """Fixed sine/cosine position encoding, shape ``[T, 1, d]``."""
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
    freq = 10000 ** (-torch.arange(0, d, 2, device=device, dtype=dtype) / d)
    pe = torch.zeros(T, d, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)[:, : d // 2]
    return pe.unsqueeze(1)


class BroadcastBlock(Block):
    """Tile the input along the feature dim (no parameters)."""

    def __init__(
        self,
        id: str,
        in_keys: List[str],
        out_keys: List[str],
        in_dim: int,
        out_dim: int,
        init_specs: Optional[Dict[str, InitSpecs]] = None,
        freeze_specs: Optional[List[str]] = None,
    ) -> None:
        if out_dim % in_dim != 0:
            raise ValueError(
                f"BroadcastBlock '{id}': out_dim ({out_dim}) must be a multiple of "
                f"in_dim ({in_dim})"
            )
        super().__init__(id, in_keys, out_keys, in_dim, out_dim)
        self.repeat_factor = out_dim // in_dim
        self.set_parameters(init_specs)
        self.freeze(freeze_specs)

    def forward_block(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return [inputs.repeat_interleave(self.repeat_factor, dim=-1)]


class ConnectivityLinear(nn.Module):
    """Linear layer with weights quantised to {-1, 0, +1} (straight-through)."""

    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = False, thr: float = 0.0, scale: float = 1.0
    ):
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
        activation: Activation = "none",
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
        self.activation = _ACTIVATIONS[activation]()
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
