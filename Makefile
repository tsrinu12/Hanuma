# Common dev ops for distrebute-ml.
#
# All targets assume MODEL_PROFILE=lite unless you override:
#   make test MODEL_PROFILE=prod

SHELL := /bin/bash
SERVICES := cold_start_bandit semantic_search moderation live_moderation
MODEL_PROFILE ?= lite

.PHONY: help install test test-% e2e up down logs clean docker-build push-local

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install python deps for every service.
	@for svc in $(SERVICES); do \
	  echo ">> installing $$svc"; \
	  pip install -r services/$$svc/requirements.txt; \
	done
	@pip install websockets

test:  ## Run unit tests for every service.
	@for svc in $(SERVICES); do \
	  echo ">> testing $$svc"; \
	  ( cd services/$$svc && \
	    PYTHONPATH=$$PWD:$$PWD/../.. \
	    MODEL_PROFILE=$(MODEL_PROFILE) \
	    pytest -q --tb=short ) || exit 1; \
	done

test-%:  ## Run unit tests for one service, e.g. `make test-moderation`.
	@cd services/$* && \
	  PYTHONPATH=$$PWD:$$PWD/../.. \
	  MODEL_PROFILE=$(MODEL_PROFILE) \
	  pytest -q --tb=short

e2e:  ## Run end-to-end harness (boots all services on localhost).
	@MODEL_PROFILE=$(MODEL_PROFILE) python scripts/e2e.py

up:  ## docker compose up the whole stack.
	@MODEL_PROFILE=$(MODEL_PROFILE) docker compose up --build

down:  ## Tear down compose stack.
	@docker compose down

logs:  ## Tail compose logs.
	@docker compose logs -f

clean:  ## Remove pycache and pytest artifacts.
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name .pytest_cache -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -f scripts/*.e2e.log scripts/e2e_report.json

docker-build:  ## Build all four service images.
	@for svc in $(SERVICES); do \
	  echo ">> building $$svc"; \
	  docker build -f services/$$svc/Dockerfile -t distrebute/$$svc:local . || exit 1; \
	done
