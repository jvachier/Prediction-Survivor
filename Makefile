UV ?= uv

.PHONY: install lint black ruff test test-cov test-fast

install:
	$(UV) sync --group dev

lint:
	$(UV) run pylint --disable=R,C src/

black:
	$(UV) run black src/

ruff:
	$(UV) run ruff check src/
	$(UV) run ruff check --fix src/
	$(UV) run ruff format src/

test:
	$(UV) run pytest tests/ -v

test-cov:
	$(UV) run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	$(UV) run pytest tests/ -v -x --ff