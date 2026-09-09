# ==============================================================================
# CVGuard Phase 0 Root Automation Makefile
# Offline Computer Vision Integrity Assurance Platform
# ==============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Environment configuration discovery
COMPOSE_FILE := infra/docker-compose.yml
ENV_FILE := $(if $(wildcard infra/.env),infra/.env,infra/.env.example)

.PHONY: help up down test lint schema-check clean

help:
	@echo "CVGuard Automation Tasks (Phase 1: Governance Spine):"
	@echo "  make up           - Build and start all 6 planes, Postgres, Redis, and MinIO via docker-compose"
	@echo "  make down         - Stop and tear down all docker-compose containers and networks"
	@echo "  make test         - Run pytest across schemas, health probes, and Phase 1 security tests"
	@echo "  make test-security- Run only the 4 Phase 1 tamper-evident ledger security tests"
	@echo "  make lint         - Run ruff (lint & format check) and mypy static analysis across all services"
	@echo "  make schema-check - Validate that libs/schemas installs and imports cleanly in all services"
	@echo "  make clean        - Remove Python cache files, .pytest_cache, and build artifacts"

up:
	@echo "==> Starting CVGuard infrastructure and plane services using $(ENV_FILE)..."
	docker compose -f $(COMPOSE_FILE) --env-file $(ENV_FILE) up --build

down:
	@echo "==> Tearing down CVGuard containers..."
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

test:
	@echo "==> Running test suite across all services, shared schemas, and security tests..."
	PYTHONPATH=libs/schemas:services/gateway:services/data-plane:services/model-plane:services/inference-plane:services/drift-plane:services/governance \
	pytest -v tests/

test-security:
	@echo "==> Running Phase 1 Governance Spine Security Test Suite..."
	PYTHONPATH=libs/schemas:services/governance \
	pytest -v tests/test_security.py

test-e2e:
	@echo "==> Running Phase 2 End-to-End Vertical Slice Pipeline Test..."
	PYTHONPATH=libs/schemas:services/data-plane:services/gateway:services/governance \
	pytest -v tests/test_e2e_pipeline.py

test-detectors:
	@echo "==> Running Phase 3 Data Plane Detector Unit Tests..."
	PYTHONPATH=libs/schemas:services/data-plane:services/governance \
	pytest -v tests/test_data_plane_detectors.py

test-phase3:
	@echo "==> Running Phase 3 Multi-Detector E2E & Source Aggregation Integration Test..."
	PYTHONPATH=libs/schemas:services/data-plane:services/gateway:services/governance \
	pytest -v tests/test_e2e_phase3.py

load-sanity:
	@echo "==> Running Phase 2 Load Sanity Benchmark (200 Synthetic Images)..."
	python3 scripts/load_sanity.py --batch-size 200

lint:
	@echo "==> Checking code style and formatting with ruff..."
	ruff check .
	ruff format --check .
	@echo "==> Running static type analysis with mypy..."
	MYPYPATH=libs/schemas:services/governance mypy libs/ services/ tests/

schema-check:
	@echo "==> Validating that libs/schemas is importable across all 6 service plane runtimes..."
	python3 -c "import sys; sys.path.insert(0, 'libs/schemas'); import cvguard_schemas; from cvguard_schemas import Finding, SignedFinding, AssetType, Report, Severity, Disposition; print('  [OK] cvguard_schemas v' + cvguard_schemas.__version__ + ' imported successfully')"
	python3 -c "\
import sys, importlib.util; \
sys.path.insert(0, 'libs/schemas'); \
planes = ['gateway', 'data-plane', 'model-plane', 'inference-plane', 'drift-plane', 'governance']; \
for p in planes: \
    path = f'services/{p}/main.py'; \
    spec = importlib.util.spec_from_file_location(p, path); \
    mod = importlib.util.module_from_spec(spec); \
    spec.loader.exec_module(mod); \
    assert hasattr(mod, 'app'), f'{p} missing FastAPI app instance'; \
    print(f'  [OK] Service \"{p}\" imported cvguard_schemas without error.'); \
print('==> All 6 services successfully validated with shared schemas!')"

clean:
	@echo "==> Cleaning cache and temporary build directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist build .coverage
