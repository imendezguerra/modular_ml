"""Post-training analysis config dataclass."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProcessingConfig:
    """Configuration for the analysis logged after testing.

    The default analysis is PCA over the listed activation keys (the
    representation-dimensionality showcase). ``evr_threshold`` is the explained
    variance ratio used to report the number of PCs needed.
    """

    _target_: str = "modular_ml.configs.processing_configs.ProcessingConfig"

    run_pca: bool = True
    pca_keys: List[str] = field(default_factory=list)
    pca_dims: int = 10
    evr_threshold: float = 0.95
