UV ?= uv

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