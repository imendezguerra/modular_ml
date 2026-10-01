.PHONY: install lint test data train sweep experiment docs clean

install:
	pip install -e ".[dev,docs]"

lint:
	ruff check src tests

test:
	pytest

data:
	python scripts/generate_data.py

train:
	python scripts/run_pipeline.py

sweep:
	python scripts/run_sweep.py

experiment:
	python scripts/run_experiment.py

docs:
	mkdocs build --strict -f docs/mkdocs.yml

clean:
	rm -rf outputs wandb site **/__pycache__ .pytest_cache
