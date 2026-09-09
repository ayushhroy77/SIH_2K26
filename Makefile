# ==============================================================================
# CVGuard Phase 0 Root Automation Makefile
# ==============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE_FILE := cvguard/infra/docker-compose.yml
ENV_FILE := $(if $(wildcard cvguard/infra/.env),cvguard/infra/.env,cvguard/infra/.env.example)

.PHONY: help up down test lint schema-check clean

help:
	@echo "CVGuard Phase 0 Automation Tasks:"
	@echo "  make up           - Build and start all 6 planes, Postgres, Redis, and MinIO via docker-compose"
	@echo "  make down         - Stop and tear down all docker-compose containers and networks"
	@echo "  make test         - Run pytest across shared schemas, health endpoints, and plane stubs"
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
	@echo "==> Running test suite across all services and shared schemas..."
	cd cvguard && PYTHONPATH=libs/schemas:services/gateway:services/data-plane:services/model-plane:services/inference-plane:services/drift-plane:services/governance \
	pytest -v tests/

lint:
	@echo "==> Checking code style and formatting with ruff..."
	cd cvguard && ruff check .
	cd cvguard && ruff format --check .
	@echo "==> Running static type analysis with mypy..."
	cd cvguard && MYPYPATH=libs/schemas mypy libs/ services/ tests/

schema-check:
	@echo "==> Validating that libs/schemas is importable across all 6 service plane runtimes..."
	cd cvguard && make schema-check

clean:
	@echo "==> Cleaning cache and temporary build directories..."
	cd cvguard && make clean
