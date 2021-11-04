PYTHON ?= python
PYTEST ?= pytest
BANDIT ?= bandit
FLAKE8 ?= flake8
BLACK ?= black
PIP ?= pip

all: clean lint test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

init:
	$(PIP) install -r requirements.txt


lint:
	$(FLAKE8) src/ tests/
	$(BANDIT) -r src/ tests/

format-code:
	$(BLACK) src/
	$(BLACK) tests/

test:
	PYTHONPATH=./src $(PYTEST)

.PHONY: clean init test