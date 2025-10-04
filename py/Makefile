# Makefile for IMP Research Environment
# Provides convenient shortcuts for common development tasks

.PHONY: help install install-dev test test-fast test-research test-notebook \
        format lint type-check clean docs validate jupyter pre-commit

# Default target
help:
	@echo "IMP Research Environment - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package with all dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo "  make validate         Validate environment setup"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-fast        Run fast tests only (skip slow tests)"
	@echo "  make test-research    Run research framework tests"
	@echo "  make test-notebook    Run notebook execution tests"
	@echo "  make coverage         Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format           Format code with black and isort"
	@echo "  make lint             Run flake8 linter"
	@echo "  make type-check       Run mypy type checker"
	@echo "  make pre-commit       Run all pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make jupyter          Start Jupyter Lab"
	@echo "  make docs             Build documentation"
	@echo "  make clean            Clean build artifacts and cache"
	@echo ""

# Installation targets
install:
	pip install -e ".[dev,optimization,research]"

install-dev:
	pip install -e ".[dev,optimization,research]"
	pip install pre-commit
	pre-commit install
	python -m ipykernel install --user --name=imp-research --display-name="IMP Research Environment"

validate:
	python -m imp.utils.env_validator

# Testing targets
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-research:
	pytest tests/ -v -m research

test-notebook:
	pytest tests/ -v -m notebook

test-integration:
	pytest tests/ -v -m integration

coverage:
	pytest tests/ --cov=imp --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality targets
format:
	black imp/ tests/
	isort imp/ tests/
	@echo "Code formatted successfully"

format-check:
	black --check imp/ tests/
	isort --check-only imp/ tests/

lint:
	flake8 imp/ tests/ --max-line-length=88 --extend-ignore=E203,W503

type-check:
	mypy imp/ --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

# Development targets
jupyter:
	jupyter lab

docs:
	@echo "Documentation files:"
	@echo "  - docs/DEVELOPMENT_SETUP.md"
	@echo "  - docs/DEVELOPMENT_WORKFLOW.md"
	@echo "  - docs/QUICK_REFERENCE.md"
	@echo "  - tests/RESEARCH_TESTING_GUIDE.md"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "Cleaned build artifacts and cache"

clean-notebooks:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
	@echo "Cleared notebook outputs"

# CI/CD targets
ci-test:
	pytest tests/ -v --tb=short -m "not slow"

ci-lint:
	black --check imp/ tests/
	isort --check-only imp/ tests/
	flake8 imp/ tests/ --max-line-length=88 --extend-ignore=E203,W503

ci-all: ci-lint ci-test

# Development workflow targets
dev-setup: install-dev validate
	@echo ""
	@echo "Development environment setup complete!"
	@echo "Run 'make jupyter' to start Jupyter Lab"

dev-check: format lint type-check test-fast
	@echo ""
	@echo "All checks passed! Ready to commit."

# Research workflow targets
research-start: jupyter

research-test:
	pytest tests/test_research_framework.py -v
	pytest tests/test_notebook_integration.py -v

# Utility targets
list-kernels:
	jupyter kernelspec list

remove-kernel:
	jupyter kernelspec remove imp-research

reinstall-kernel:
	python -m ipykernel install --user --name=imp-research --display-name="IMP Research Environment"

# Version info
version:
	@python -c "import imp; print(f'IMP Version: {imp.__version__ if hasattr(imp, \"__version__\") else \"unknown\"}')"
	@python --version
	@pip --version

# Dependencies
deps-list:
	pip list

deps-outdated:
	pip list --outdated

deps-update:
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e ".[dev,optimization,research]"

# Benchmarking
benchmark:
	pytest tests/ -v --benchmark-only

# Profile tests
profile:
	pytest tests/ -v --profile

# Generate requirements.txt
freeze:
	pip freeze > requirements.txt
	@echo "Requirements saved to requirements.txt"
