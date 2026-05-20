# Makefile to recreate pyproject.toml using uv commands

.PHONY: create-pyproject clean help tests tests-fast tests-slow tests-cov measurement-validation lint format typecheck setup-pre-commit test-voila-e2e serve-voila kill-voila ci

JUPYTER_MATH_IMAGE ?= itisfoundation/jupyter-math:3.0.5

# Default target
help:
	@echo "Available targets:"
	@echo "  tests                - Run all tests (use target=<test_file>::<test_function> for specific test)"
	@echo "  tests-fast           - Run only fast tests (excludes slow and validation markers)"
	@echo "  tests-slow           - Run slow and integration tests (excludes measurement-validation)"
	@echo "  tests-cov            - Run all tests with coverage report"
	@echo "  measurement-validation - Run measurement validation in parallel with xdist"
	@echo "  lint                 - Run Ruff lint checks"
	@echo "  format               - Run Ruff formatter"
	@echo "  typecheck            - Run ty type checks"
	@echo "  setup-pre-commit     - Install and set up pre-commit hooks"
	@echo "  run-pre-commit       - Run pre-commit hooks on staged files"
	@echo "  run-pre-commit-all   - Run pre-commit hooks on all files"
	@echo "  ci                   - Run exactly what CI runs (lint + type-check + fast tests + slow tests + E2E)"
	@echo "  help                 - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make tests                                           # Run all tests"
	@echo "  make tests-fast                                      # Run only fast tests"
	@echo "  make tests-slow                                      # Run only slow tests"
	@echo "  make tests target=test_image_loader.py               # Run all tests in a file"
	@echo "  make tests target=test_image_loader.py::test_parse_and_build_no_resample  # Run specific test"
	@echo "  make lint                                            # Run Ruff linter"
	@echo "  make measurement-validation                          # Run measurement validation with xdist"
	@echo "  make format                                          # Format with Ruff"
	@echo "  make typecheck                                       # Run ty type checker"
	@echo "  make setup-pre-commit                                # Set up pre-commit hooks"
	@echo "  make run-pre-commit                                  # Run pre-commit hooks on staged files"

# Run tests with optional target specification
tests:
	@echo "Running tests..."
ifdef target
	@echo "Target: $(target)"
	uv run pytest -v tests/$(target)
else
	@echo "Running all tests"
	uv run pytest -v tests/
endif

# Run only fast tests (exclude slow, validation, and notebook_smoke markers)
tests-fast:
	@echo "Running fast tests (excluding slow, validation, and notebook_smoke tests)..."
	uv run pytest -v -m "not slow and not validation and not notebook_smoke" tests/

# Run slow and integration tests (excludes measurement-validation)
tests-slow:
	@echo "Running slow and integration tests (excluding measurement-validation)..."
	uv run pytest -v --run-slow --run-validation -m "slow or validation" --ignore=tests/test_measurement_validation.py tests/

# Run tests with coverage (matches CI behavior)
tests-cov:
	@echo "Running tests with coverage..."
	uv run pytest -v --cov=src/sar_pattern_validation --cov-report=xml:coverage.xml --cov-report=term --cov-report=html tests/
	@echo "Coverage report generated: coverage.xml and htmlcov/"

measurement-validation:
	@echo "Running measurement validation with xdist..."
	uv run pytest -v -n auto --dist loadscope --run-validation tests/test_measurement_validation.py

lint:
	@echo "Running Ruff linter..."
	uv run ruff check . --fix

format:
	@echo "Formatting code with Ruff..."
	uv run ruff format .

typecheck:
	@echo "Running ty type checks..."
	uv run ty check src tests

# Set up pre-commit hooks
setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	uv sync
	uv run pre-commit install
	@echo "Pre-commit hooks installed successfully!"
	@echo "Use make run-pre-commit to check staged files, or run-pre-commit-all to check all files"

# Run pre-commit hooks on staged files
run-pre-commit: setup-pre-commit
	@echo "Running pre-commit hooks on staged files..."
	uv run pre-commit run

# Run pre-commit hooks on all files (useful for CI or manual checks)
run-pre-commit-all: setup-pre-commit
	@echo "Running pre-commit hooks on all files..."
	uv run pre-commit run --all-files

# --------------------------------------------------------------------------
# Container-first voila harness — runs inside itisfoundation/jupyter-math:3.0.5
# with the repo bind-mounted. See scripts/voila_docker.sh and
# scripts/run_in_jupyter_math.sh.
# --------------------------------------------------------------------------

test-voila-e2e:
	JUPYTER_MATH_IMAGE=$(JUPYTER_MATH_IMAGE) ./scripts/voila_docker.sh test

serve-voila:
	JUPYTER_MATH_IMAGE=$(JUPYTER_MATH_IMAGE) ./scripts/voila_docker.sh shell

kill-voila:
	@docker ps --filter "name=sar-voila-jm-" -q | xargs -r docker kill 2>/dev/null || true
	@docker ps --filter "ancestor=$(JUPYTER_MATH_IMAGE)" -q | xargs -r docker kill 2>/dev/null || true
	@echo "kill-voila: done"

# Run exactly what CI runs, locally.
# Mirrors the four CI jobs: type-check, tests, slow-tests, e2e-tests.
# Note: slow-tests and e2e-tests require git-lfs data (run `git lfs pull` first).
ci:
	@echo "=== [1/4] lint + type-check ==="
	uv run ruff check src/ tests/
	uv run ty check src/
	@echo "=== [2/4] fast tests ==="
	uv run pytest -v -m "not slow and not validation and not notebook_smoke" \
		--cov=src/sar_pattern_validation --cov-report=xml:coverage.xml --cov-report=term \
		tests/
	@echo "=== [3/4] slow + integration tests ==="
	uv run pytest -v -m "slow or validation" \
		--basetemp=test-artifacts \
		--ignore=tests/test_measurement_validation.py \
		--cov=src/sar_pattern_validation --cov-report=xml:coverage-validation.xml --cov-report=term \
		tests/
	@echo "=== [4/4] e2e tests (voila + playwright) ==="
	JUPYTER_MATH_IMAGE=$(JUPYTER_MATH_IMAGE) ./scripts/voila_docker.sh test
	@echo "=== CI complete ==="
