PYTHON ?= python
PYTEST ?= pytest
BANDIT ?= bandit
FLAKE8 ?= flake8
BLACK ?= black
TOX ?= tox
PIP ?= pip

all: clean lint test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

init:
	$(PIP) install -r requirements.txt


lint:
	$(FLAKE8) src/ tests/
	$(BANDIT) -c .bandit.yml -r src/ tests/

format-code:
	$(BLACK) src/
	$(BLACK) tests/

test:
	PYTHONPATH=./src $(PYTEST)

test-matrix:
	$(TOX) -r

.PHONY: clean init test