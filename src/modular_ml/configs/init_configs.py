"""Per-parameter initialisation specs.

Each block declares how to initialise individual parameters via a mapping
``{param_name: InitSpec}`` in its YAML ``init:`` block. Specs carry a ``method``
string consumed by :func:`modular_ml.models.initializers._init_param` and an
optional sparsity request. ``InitByInputs`` lets a single weight matrix be
initialised in column chunks (one per concatenated input).
"""

from dataclasses import dataclass, field
from typing import Union


class InitBase:
    """Base initializer (carries common sparsity flags)."""

    method: str
    enforce_sparse: bool = True
    sparse_col: float = 0.0
    sparse_gen: float = 0.0


@dataclass
class InitNormal(InitBase):
    """Normal-distribution initializer."""

    mean: float = 0.0
    std: float = 0.05
    method: str = "normal"
    enforce_sparse: bool = True
    sparse_col: float = 0.0
    sparse_gen: float = 0.0
    _target_: str = "modular_ml.configs.init_configs.InitNormal"


@dataclass
class InitUniform(InitBase):
    """Uniform-distribution initializer."""

    scale_min: float = -0.1
    scale_max: float = 0.1
    method: str = "uniform"
    enforce_sparse: bool = True
    sparse_col: float = 0.0
    sparse_gen: float = 0.0
    _target_: str = "modular_ml.configs.init_configs.InitUniform"


@dataclass
class InitNormalRec(InitBase):
    """Gain-scaled normal init for recurrent matrices (std = g / sqrt(N))."""

    g: float = 1.2
    method: str = "normal_rec"
    enforce_sparse: bool = True
    sparse_col: float = 0.0
    sparse_gen: float = 0.0
    _target_: str = "modular_ml.configs.init_configs.InitNormalRec"


@dataclass
class InitFill(InitBase):
    """Constant-fill initializer."""

    value: float = 0.0
    method: str = "fill"
    enforce_sparse: bool = True
    sparse_col: float = 0.0
    sparse_gen: float = 0.0
    _target_: str = "modular_ml.configs.init_configs.InitFill"


@dataclass
class InitChunk:
    """One column chunk for :class:`InitByInputs` (dim + nested spec)."""

    dim: int
    spec: InitBase


@dataclass
class InitByInputs(InitBase):
    """Initialise a weight matrix in column chunks, one per concatenated input."""

    method: str = "by_inputs"
    enforce_sparse: bool = True
    chunks: list = field(default_factory=list)
    _target_: str = "modular_ml.configs.init_configs.InitByInputs"

    def __post_init__(self):
        self.chunks = [
            c if isinstance(c, InitChunk) else InitChunk(**c) for c in self.chunks
        ]


InitSpecs = Union[InitNormal, InitUniform, InitNormalRec, InitFill, InitByInputs]
