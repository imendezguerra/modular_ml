# Repository structure

The repository separates **what you configure** (`configs/`, YAML) from **what
you implement** (`src/modular_ml/`, Python). Each component in the diagram has a
config group and a matching Python module.

--8<-- "README.md:diagram"

## Folder layout

```
modular_ml/
├── configs/                     # all YAML (the things you edit most)
│   ├── runs/                    # run configs + Hydra config groups
│   │   ├── dataset/             #   group: which dataset + how to read it
│   │   ├── model/               #   group: architecture (which blocks, how connected)
│   │   ├── optim/               #   group: optimiser + loss terms
│   │   ├── processing/          #   group: post-training analyses
│   │   ├── logger/              #   group: wandb logging + sweep objective
│   │   ├── config_sequential.yaml   # concrete run (RNN task)
│   │   └── config_tabular.yaml      # concrete run (MLP task)
│   ├── data_gen/                # params for the synthetic example data
│   ├── sweeps/                  # wandb sweep specs
│   └── experiments/             # multi-stage experiment specs
├── src/modular_ml/              # the Python package
│   ├── configs/                 # dataclass schemas + load_config (Hydra)
│   ├── data/                    # data_gen.py (example generators), datasets.py
│   ├── models/                  # blocks.py, network.py, initializers.py
│   ├── losses/                  # loss.py, regularisers.py
│   ├── pipeline/                # runner.py, pipeline.py, runtime_schedule.py,
│   │                            # experiment_tracking.py
│   ├── processing/              # dimensionality.py (PCA), probing.py (linear probes), io.py
│   ├── tools/                   # hydra_utils.py, logging.py, loaders.py, utils.py
│   └── constants.py             # paths + wandb project/entity
├── scripts/                     # thin entry points
│   ├── generate_data.py
│   ├── run_pipeline.py
│   ├── run_sweep.py
│   ├── run_experiment.py
│   └── make_doc_figures.py      # regenerates the figures in these docs
├── notebooks/tutorial_colab.ipynb   # runnable Colab version of the tutorial
├── data/                        # datasets in pipeline format (gitignored)
├── outputs/                     # per-run outputs + checkpoints (gitignored)
├── tests/                       # pytest suite (wandb disabled)
├── docs/                        # this site (user guide + API reference)
├── .github/                     # CI/CD workflows + RELEASING.md
├── ci/constraints-min.txt       # minimum dependency pins for CI
├── mkdocs.yml                   # docs site config
├── pyproject.toml
└── environment.yaml
```

## Package responsibilities

| Module | Responsibility |
|--------|----------------|
| `configs/` | Dataclass schema for every subsystem + `load_config` (compose → resolve → instantiate). |
| `data/` | Gets data into the pipeline format: loaders, converters or generators (here, the synthetic example generators), plus the layout-aware dataset/batching wrapper. |
| `models/` | `Block` base class + general blocks; `ModularNet` and the `build_modular_model` factory. |
| `losses/` | `LossModule` / `ModularLoss` + `build_modular_loss`; supervised criteria and regularisers. |
| `pipeline/` | The `Runner` (train/test/log) and the `run_pipeline` / `run_sweep` / `run_experiment` entry points. |
| `processing/` | Analyses that run after a model is trained, on the model or its outputs (e.g. interpretability, representation analysis; PCA is the included example), plus loaders to read a run's artifacts back. |
| `tools/` | Hydra resolvers, wandb wrappers, HDF5/YAML IO, module attribute lookup. |

Next: the Concepts pages cover each component in depth, starting with
[Configuration](configs.md). The [API reference](../api/configs.md) documents every module.
