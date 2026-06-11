"""Dataset config dataclass.

Datasets are synthetic and stored as HDF5 (see :mod:`modular_ml.data`). The
config points at a generated dataset directory and records the tensor keys the
pipeline should read. Dimensions/dt are read back from the dataset's own
``meta.yaml`` when present so the model config can interpolate against them.
"""

from dataclasses import dataclass
from pathlib import Path

from modular_ml.configs import enums
from modular_ml.constants import DATA_DIR
from modular_ml.tools.loaders import yaml_load


@dataclass
class DatasetConfig:
    """Where the data lives and how to read it."""

    _target_: str = "modular_ml.configs.dataset_configs.DatasetConfig"

    name: str = "dataset"
    data_dir: str = "path_to_data"
    task_type: enums.TaskType = enums.TaskType.regression

    in_dim: int = 1
    out_dim: int = 1
    dt: float = 0.01

    # Keys inside the HDF5 files.
    in_key: str = "stim"
    out_key: str = "target"
    label_key: str = "target"

    def __post_init__(self):
        if isinstance(self.task_type, str):
            self.task_type = enums.TaskType(self.task_type)

        # Resolve data_dir relative to the project data directory if not absolute.
        data_dir = Path(self.data_dir)
        if not data_dir.is_absolute():
            data_dir = DATA_DIR / data_dir
        self.data_dir = data_dir

        # Pull dimensions/dt from the dataset's metadata when available.
        meta_path = data_dir / "meta.yaml"
        if meta_path.exists():
            meta = yaml_load(meta_path)
            self.in_dim = meta.get("in_dim", self.in_dim)
            self.out_dim = meta.get("out_dim", self.out_dim)
            self.dt = meta.get("dt", self.dt)
            self.task_type = enums.TaskType(meta.get("task_type", self.task_type.value))
            self.in_key = meta.get("in_key", self.in_key)
            self.out_key = meta.get("out_key", self.out_key)
            self.label_key = meta.get("label_key", self.label_key)
