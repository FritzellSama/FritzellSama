# Quantum Trader AI - Production Makefile
# Institutional Trading Platform - Manage billions in real-time
# Python 3.12+, Polars, AsyncIO, FastAPI, PyTorch 2.0+

.PHONY: help install install-dev test test-unit test-integration test-coverage lint format type-check security-check \
		build build-docker clean clean-cache run-dev run-prod deploy deploy-staging deploy-prod \
		db-migrate db-rollback db-seed backup restore monitor logs stop

# Load environment variables from .env file
include .env
export

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Quantum Trader AI - Makefile Commands$(NC)"
	@echo "$(YELLOW)Environment: $(ENVIRONMENT)$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-30s$(NC) %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	@echo "$(GREEN)Production dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)Development dependencies installed$(NC)"

# Testing targets
test: test-unit test-integration ## Run all tests
	@echo "$(GREEN)All tests completed$(NC)"

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit -v --tb=short --maxfail=1

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration -v --tb=short --maxfail=1

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=src/quantum_trader --cov-report=html --cov-report=term --cov-fail-under=80

# Code quality targets
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	ruff check src/ tests/
	pylint src/quantum_trader

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)Code formatted$(NC)"

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy src/quantum_trader --strict

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src/quantum_trader -ll
	safety check --json
	@echo "$(GREEN)Security checks passed$(NC)"

# Build targets
build: clean ## Build Python package
	@echo "$(BLUE)Building Python package...$(NC)"
	python -m build
	@echo "$(GREEN)Package built successfully$(NC)"

build-docker: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose -f docker/docker-compose.yml build
	@echo "$(GREEN)Docker images built$(NC)"

# Cleanup targets
clean: clean-cache ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Build artifacts cleaned$(NC)"

clean-cache: ## Clean cache files
	@echo "$(BLUE)Cleaning cache files...$(NC)"
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	@echo "$(GREEN)Cache files cleaned$(NC)"

# Runtime targets
run-dev: ## Run development server
	@echo "$(BLUE)Starting development server...$(NC)"
	python -m src.quantum_trader.main --config config/environments/development.yaml

run-prod: ## Run production server
	@echo "$(BLUE)Starting production server...$(NC)"
	python -m src.quantum_trader.main --config config/environments/production.yaml

# Deployment targets
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	ansible-playbook -i ansible/inventories/staging/hosts ansible/playbooks/deploy.yml
	@echo "$(GREEN)Deployed to staging$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(YELLOW)WARNING: Deploying to production!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		ansible-playbook -i ansible/inventories/production/hosts ansible/playbooks/deploy.yml; \
		echo "$(GREEN)Deployed to production$(NC)"; \
	else \
		echo "$(RED)Deployment cancelled$(NC)"; \
	fi

# Database targets
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)Migrations completed$(NC)"

db-rollback: ## Rollback last database migration
	@echo "$(YELLOW)Rolling back last migration...$(NC)"
	alembic downgrade -1
	@echo "$(GREEN)Rollback completed$(NC)"

db-seed: ## Seed database with test data
	@echo "$(BLUE)Seeding database...$(NC)"
	python scripts/db/seed_data.py
	@echo "$(GREEN)Database seeded$(NC)"

# Backup and restore targets
backup: ## Backup database and configs
	@echo "$(BLUE)Creating backup...$(NC)"
	python scripts/backup/create_backup.py
	@echo "$(GREEN)Backup created$(NC)"

restore: ## Restore from backup
	@echo "$(YELLOW)Restoring from backup...$(NC)"
	python scripts/backup/restore_backup.py --backup-id $(BACKUP_ID)
	@echo "$(GREEN)Restore completed$(NC)"

# Monitoring targets
monitor: ## Open monitoring dashboard
	@echo "$(BLUE)Opening monitoring dashboard...$(NC)"
	xdg-open http://localhost:$${GRAFANA_PORT:-3000}

logs: ## Tail application logs
	@echo "$(BLUE)Tailing logs...$(NC)"
	tail -f logs/quantum_trader.log

stop: ## Stop all running services
	@echo "$(BLUE)Stopping all services...$(NC)"
	docker-compose -f docker/docker-compose.yml down
	pkill -f "python -m src.quantum_trader" || true
	@echo "$(GREEN)All services stopped$(NC)"

# Analysis targets
analyze-performance: ## Run performance analysis
	@echo "$(BLUE)Running performance analysis...$(NC)"
	python scripts/analysis/performance_analysis.py

analyze-profit: ## Generate profit report
	@echo "$(BLUE)Generating profit report...$(NC)"
	python scripts/analysis/profit_report.py

analyze-risk: ## Generate risk report
	@echo "$(BLUE)Generating risk report...$(NC)"
	python scripts/analysis/risk_report.py

analyze-trades: ## Analyze trade data
	@echo "$(BLUE)Analyzing trades...$(NC)"
	python scripts/analysis/trade_analysis.py
