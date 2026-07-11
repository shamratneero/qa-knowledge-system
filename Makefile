PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help install run test test-fast test-slow benchmark smoke preflight release-prep

help:
	@echo "Available targets:"
	@echo "  install       Install dependencies into .venv"
	@echo "  run           Start local API server"
	@echo "  test          Run all tests"
	@echo "  test-fast     Run fast test subset"
	@echo "  test-slow     Run slow test subset"
	@echo "  benchmark     Run benchmark harness"
	@echo "  smoke         Run API smoke checks"
	@echo "  preflight     Run release/deploy preflight checks"
	@echo "  release-prep  Prepare release tag (usage: make release-prep VERSION=v1.0.1)"

install:
	$(PIP) install -r requirements.txt

run:
	bash scripts/run/start_local.sh

test:
	$(PYTHON) -m pytest -v

test-fast:
	$(PYTHON) -m pytest -m "not slow" -v

test-slow:
	$(PYTHON) -m pytest -m slow -v

benchmark:
	$(PYTHON) scripts/benchmark.py

smoke:
	bash scripts/run/smoke_test.sh

preflight:
	bash scripts/release/preflight.sh

release-prep:
	@test -n "$(VERSION)" || (echo "Usage: make release-prep VERSION=v1.0.1" && exit 1)
	bash scripts/release/prepare_release.sh $(VERSION)
