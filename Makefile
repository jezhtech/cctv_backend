.PHONY: help install install-dev test test-cov lint format clean start stop build up down logs

help: ## Show this help message
	@echo "Smart Attendance Backend - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	poetry install --only main

install-dev: ## Install all dependencies including development
	poetry install --with dev

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=app --cov-report=html

lint: ## Run linting
	poetry run flake8 app/
	poetry run mypy app/

format: ## Format code
	poetry run black app/
	poetry run isort app/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/

start: ## Start the application
	poetry run python start.py

start-dev: ## Start the application in development mode
	poetry run python -m app.main

build: ## Build Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View logs
	docker-compose logs -f

logs-app: ## View application logs
	docker-compose logs -f app

logs-db: ## View database logs
	docker-compose logs -f db

logs-redis: ## View Redis logs
	docker-compose logs -f redis

logs-celery: ## View Celery logs
	docker-compose logs -f celery

db-migrate: ## Run database migrations
	poetry run alembic upgrade head

db-rollback: ## Rollback last migration
	poetry run alembic downgrade -1

db-revision: ## Create new migration
	poetry run alembic revision --autogenerate -m "Update"

db-reset: ## Reset database (WARNING: This will delete all data)
	docker-compose down -v
	docker-compose up -d db
	sleep 10
	poetry run python -c "from app.core.database import create_tables; create_tables()"

setup: ## Initial setup
	@echo "Setting up Smart Attendance Backend..."
	@echo "1. Installing dependencies..."
	poetry install
	@echo "2. Creating necessary directories..."
	mkdir -p logs uploads models
	@echo "3. Setting up environment..."
	@if [ ! -f .env ]; then \
		echo "Creating .env file from template..."; \
		cp env.example .env; \
		echo "Please edit .env file with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi
	@echo "4. Starting services..."
	docker-compose up -d db redis
	@echo "5. Waiting for services to be ready..."
	sleep 15
	@echo "6. Creating database tables..."
	poetry run python -c "from app.core.database import create_tables; create_tables()"
	@echo "Setup complete! Run 'make start' to start the application"

test-setup: ## Test if the application setup is correct
	poetry run python test_setup.py

celery-start: ## Start Celery worker
	poetry run celery -A app.core.celery_app worker --loglevel=info

celery-flower: ## Start Celery Flower monitoring
	poetry run celery -A app.core.celery_app flower --port=5555

monitor: ## Start monitoring services
	@echo "Starting monitoring services..."
	@echo "Celery Flower: http://localhost:5555"
	@echo "Application: http://localhost:8000"
	@echo "Health Check: http://localhost:8000/health"
	@echo "API Docs: http://localhost:8000/docs"
	make celery-flower &

production: ## Production deployment
	@echo "Production deployment..."
	@echo "1. Building images..."
	make build
	@echo "2. Starting services..."
	docker-compose -f docker-compose.yml up -d
	@echo "3. Scaling workers..."
	docker-compose up -d --scale celery=3
	@echo "Production deployment complete!"

production-down: ## Stop production services
	docker-compose down

status: ## Show service status
	@echo "Service Status:"
	@docker-compose ps
	@echo ""
	@echo "Active streams:"
	@curl -s http://localhost:8000/api/v1/cameras/streams/status | python -m json.tool 2>/dev/null || echo "Application not running"
