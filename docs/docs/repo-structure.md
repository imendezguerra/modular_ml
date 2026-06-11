# Repository structure

```
modular_ml_pipe/
├── configs/                     # all YAML (the things you edit most)
│   ├── runs/                    # run configs + Hydra config groups
│   │   ├── dataset/             #   group: which dataset + how to read it
│   │   ├── model/               #   group: architecture (blocks)
│   │   ├── optim/               #   group: optimizer + loss terms
│   │   ├── logger/              #   group: wandb logging + sweep objective
│   │   ├── processing/          #   group: post-test analysis (PCA)
│   │   ├── config_sequential.yaml   # concrete run (RNN task)
│   │   └── config_tabular.yaml       # concrete run (MLP task)
│   ├── data_gen/                # synthetic data generation params
│   ├── sweeps/                  # wandb sweep specs
│   └── experiments/             # multi-stage experiment specs
├── data/                        # generated datasets (gitignored)
├── outputs/                     # per-run outputs + checkpoints (gitignored)
├── scripts/                     # thin entry points
│   ├── generate_data.py
│   ├── run_pipeline.py
│   ├── run_sweep.py
│   └── run_experiment.py
├── src/modular_ml/
│   ├── configs/                 # dataclass configs + load_config (Hydra)
│   ├── models/                  # blocks.py, network.py, initializers.py
│   ├── losses/                  # loss.py, regularisers.py
│   ├── data/                    # data_gen.py, datasets.py
│   ├── pipeline/                # runner.py, pipeline.py, runtime_schedule.py,
│   │                            # experiment_tracking.py
│   ├── processing/              # dimensionality.py (PCA), io.py
│   ├── tools/                   # hydra_utils.py, logging.py, loaders.py
│   └── constants.py
├── tests/                       # pytest suite (wandb disabled)
└── docs/                        # this site
```

## Package responsibilities

| Module | Responsibility |
|--------|----------------|
| `configs/` | Dataclass schema for every subsystem + `load_config` (compose → resolve → instantiate). |
| `models/` | `Block` base + general blocks; `ModularNet` and the `build_modular_model` factory. |
| `losses/` | `LossModule` / `ModularLoss` + `build_modular_loss`; general regulariser criteria. |
| `data/` | Synthetic generators and the dataset/batching wrapper. |
| `pipeline/` | The `Runner` (train/test/log) and the `run_pipeline` / `run_sweep` / `run_experiment` entry points. |
| `processing/` | Post-test analysis (PCA) and run-artifact loaders. |
| `tools/` | Hydra resolvers, wandb wrappers, HDF5/YAML IO. |
