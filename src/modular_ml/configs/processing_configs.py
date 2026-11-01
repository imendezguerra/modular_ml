"""Post-training analysis config dataclass."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProcessingConfig:
    """Configuration for the analysis logged after testing.

    Two analyses, each switched on independently:

    * PCA over ``pca_keys`` (representation dimensionality). ``evr_threshold``
      is the explained variance ratio used to report the number of PCs needed.
    * Linear probes decoding each of ``probe_targets`` from each of
      ``probe_keys``, cross-validated over ``probe_folds`` folds of trials.
      Labels are addressed as ``label.<key>``.
    """

    _target_: str = "modular_ml.configs.processing_configs.ProcessingConfig"

    run_pca: bool = True
    pca_keys: List[str] = field(default_factory=list)
    pca_dims: int = 10
    evr_threshold: float = 0.95

    run_probe: bool = False
    probe_keys: List[str] = field(default_factory=list)
    probe_targets: List[str] = field(default_factory=list)
    probe_folds: int = 5
