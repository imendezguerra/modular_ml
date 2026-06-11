"""Generate the synthetic datasets from the data_gen configs.

Usage::

    python scripts/generate_data.py            # generate both datasets
    python scripts/generate_data.py gen_sine   # generate one by config stem
"""

import sys
from pathlib import Path

from modular_ml.constants import CONFIG_DIR, DATA_DIR
from modular_ml.data.data_gen import generate_blobs, generate_sine
from modular_ml.tools.loaders import yaml_load

_GENERATORS = {"sine": generate_sine, "blobs": generate_blobs}


def run_one(cfg_path: Path) -> None:
    cfg = yaml_load(cfg_path)
    gen = _GENERATORS[cfg.pop("generator")]
    out_dir = DATA_DIR / cfg.pop("out_dir")
    meta = gen(out_dir=out_dir, **cfg)
    print(f"[{cfg_path.stem}] wrote {meta['name']} -> {out_dir}")


def main(argv) -> None:
    gen_dir = CONFIG_DIR / "data_gen"
    stems = argv or ["gen_sine", "gen_tabular"]
    for stem in stems:
        run_one(gen_dir / f"{stem}.yaml")


if __name__ == "__main__":
    main(sys.argv[1:])
