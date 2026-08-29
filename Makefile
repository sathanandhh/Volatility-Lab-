# ── Volatility MCP — Makefile ───────────────────────────────────────
# Common development tasks. Run `make help` for a list of targets.

.PHONY: help install install-dev test test-fast lint format typecheck run run-sse docker-build docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install dev dependencies (pytest, ruff, mypy, jupyter)
	pip install -e ".[dev]"

test: ## Run full test suite
	pytest tests/ -v --tb=short

test-fast: ## Run fast tests only (skip model fitting)
	pytest tests/ -v -k "not arch_family and not order_selector and not integration" --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ --cov=core --cov=session_store --cov=mcp_server --cov-report=term-missing

lint: ## Lint with ruff
	ruff check . --fix

format: ## Format with ruff
	ruff format .

typecheck: ## Type-check with mypy
	mypy mcp_server/ core/ --ignore-missing-imports

run: ## Start MCP server (stdio transport — for Claude Desktop)
	python -m mcp_server.server --transport stdio

run-sse: ## Start MCP server (SSE transport — for remote access)
	python -m mcp_server.server --transport sse --host 0.0.0.0 --port 8765

run-http: ## Start MCP server (streamable-http)
	python -m mcp_server.server --transport streamable-http --host 0.0.0.0 --port 8765

docker-build: ## Build Docker image
	docker build -t volatility-mcp:latest -f docker/Dockerfile .

docker-up: ## Start all services via docker-compose
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all docker services
	docker compose -f docker/docker-compose.yml down

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
