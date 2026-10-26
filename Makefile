.PHONY: all help lint format typecheck test test-unit test-integration test-benchmark test-chaos build-docker clean

PYTHON ?= python

all: lint typecheck test

help:
	@echo "Available targets:"
	@echo "  lint             - Run ruff linter checks"
	@echo "  format           - Run ruff code formatting checks and fixes"
	@echo "  typecheck        - Run mypy strict type checks"
	@echo "  test             - Run full pytest test suite (unit, integration, benchmark, chaos)"
	@echo "  test-unit        - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-benchmark   - Run benchmark and load sweep tests"
	@echo "  test-chaos       - Run chaos and resilience tests"
	@echo "  build-docker     - Build Triton serving container image"
	@echo "  clean            - Remove Python bytecode, pytest, mypy, and ruff caches"

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy scripts api internal test

test:
	$(PYTHON) -m pytest test

test-unit:
	$(PYTHON) -m pytest test/unit

test-integration:
	$(PYTHON) -m pytest test/integration

test-benchmark:
	$(PYTHON) -m pytest test/benchmark

test-chaos:
	$(PYTHON) -m pytest test/chaos

build-docker:
	docker build -f deployments/docker/Dockerfile.triton -t rag-inference-serving:latest .

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p) for p in pathlib.Path('.').glob('.*cache*')]"
