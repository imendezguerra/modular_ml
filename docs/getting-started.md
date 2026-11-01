# Getting started

## Install

--8<-- "README.md:install"

The pre-commit hooks lint and format your staged files on every commit and run
the tests before every push. See `CONTRIBUTING.md` for details.

## Quickstart

--8<-- "README.md:quickstart"

`generate_data.py` creates `data/sine/` and `data/blobs/`, each with `train.h5`,
`test.h5` and a `meta.yaml`. After a run, open it in wandb to see the `train/*`,
`test/*` and analysis (`pca/*`) metrics, the PCA figure, and the logged config,
model and outputs artifacts.

Next: the [Tutorial](user-guide/tutorial.md) walks through every element of a
run, step by step.
