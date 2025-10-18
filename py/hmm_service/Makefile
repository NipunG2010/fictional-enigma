# HMM Microservice Makefile

.PHONY: help install dev start test clean lint format docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Start development server with reload"
	@echo "  start       - Start production server"
	@echo "  test        - Run basic functionality test"
	@echo "  test-full   - Run comprehensive tests (when implemented)"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"

# Install dependencies
install:
	pip install -r requirements.txt

# Development server with auto-reload
dev:
	python start.py --reload --log-level DEBUG

# Production server
start:
	python start.py

# Basic functionality test
test:
	@echo "Starting service for testing..."
	@timeout 2s python start.py --port 8002 > /dev/null 2>&1 & \
	sleep 1 && \
	python test_basic.py http://127.0.0.1:8002 && \
	pkill -f "start.py --port 8002" || true

# Comprehensive tests (placeholder for future implementation)
test-full:
	@echo "Comprehensive tests not yet implemented (will be added in task 8)"
	@echo "Run 'make test' for basic functionality tests"

# Linting (when tools are available)
lint:
	@echo "Linting tools not installed. Install with:"
	@echo "pip install flake8 mypy"
	@echo "Then run: flake8 . && mypy ."

# Code formatting (when tools are available)
format:
	@echo "Formatting tools not installed. Install with:"
	@echo "pip install black isort"
	@echo "Then run: black . && isort ."

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Docker build targets
docker-build:
	docker build -t hmm-service:latest .

docker-build-dev:
	docker build -f Dockerfile.development -t hmm-service:dev .

# Docker compose targets
docker-run:
	docker-compose up -d

docker-run-dev:
	docker-compose -f docker-compose.development.yml up -d

docker-run-prod:
	docker-compose -f docker-compose.production.yml up -d

# Docker stop targets
docker-stop:
	docker-compose down

docker-stop-dev:
	docker-compose -f docker-compose.development.yml down

docker-stop-prod:
	docker-compose -f docker-compose.production.yml down

# Docker management
docker-clean:
	docker-compose down -v
	docker system prune -f

docker-logs:
	docker-compose logs -f hmm-service

docker-logs-dev:
	docker-compose -f docker-compose.development.yml logs -f hmm-service-dev

# Environment setup
setup-env-dev:
	cp .env.development .env
	@echo "Development environment configured"

setup-env-prod:
	cp .env.production .env
	@echo "Production environment configured (update secrets!)"

# Show service status
status:
	@echo "Checking if service is running..."
	@curl -s http://localhost:8000/health > /dev/null && echo "✓ Service is running" || echo "✗ Service is not running"

# Health checks
health:
	@echo "Service Health:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Service not responding"
	@echo "\nReadiness Check:"
	@curl -s http://localhost:8000/health/ready | python -m json.tool || echo "Service not ready"

# Quick deployment commands
deploy-dev: setup-env-dev docker-build-dev docker-run-dev
	@echo "Development environment deployed"

deploy-prod: setup-env-prod docker-build docker-run-prod
	@echo "Production environment deployed (remember to update secrets!)"