"""Lightweight IO helpers: YAML and HDF5 round-trips.

The HDF5 helpers store a flat ``Dict[str, np.ndarray]`` plus scalar/string
attributes, which is all the synthetic datasets and test outputs need.
"""

from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
import yaml

PathLike = Union[str, Path]


def yaml_load(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def yaml_save(obj: Dict[str, Any], path: PathLike) -> None:
    """Save a dict to YAML, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def h5_save(data: Dict[str, Any], path: PathLike) -> None:
    """Save a flat dict of arrays/scalars to HDF5.

    Array-like values become datasets; scalars and strings become file
    attributes so they survive the round-trip via :func:`h5_load`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            if isinstance(value, (str, bytes, int, float, bool)):
                f.attrs[key] = value
            else:
                f.create_dataset(key, data=np.asarray(value))


def h5_load(path: PathLike) -> Dict[str, Any]:
    """Load an HDF5 file written by :func:`h5_save` into a flat dict."""
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            out[key] = f[key][()]
        for key, value in f.attrs.items():
            out[key] = value
    return out
