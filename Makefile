.PHONY: help install dev-install test lint format clean docker-up docker-down migrate

help:
	@echo "Social Media Radar - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make dev-install   Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make dev           Start development server"
	@echo "  make worker        Start Celery worker"
	@echo "  make beat          Start Celery beat scheduler"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     Start all services with Docker Compose"
	@echo "  make docker-down   Stop all services"
	@echo "  make docker-logs   View logs"
	@echo "  make docker-build  Rebuild Docker images"
	@echo ""
	@echo "Database:"
	@echo "  make migrate       Run database migrations"
	@echo "  make migration     Create new migration"
	@echo "  make db-reset      Reset database (WARNING: deletes all data)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo "  make type-check    Run type checker"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove generated files"

install:
	poetry install --only main

dev-install:
	poetry install

dev:
	poetry run uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

worker:
	poetry run celery -A app.ingestion.celery_app worker --loglevel=info

beat:
	poetry run celery -A app.ingestion.celery_app beat --loglevel=info

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

migrate:
	poetry run alembic upgrade head

migration:
	@read -p "Enter migration message: " msg; \
	poetry run alembic revision --autogenerate -m "$$msg"

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		poetry run alembic downgrade base; \
		poetry run alembic upgrade head; \
	fi

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=app --cov-report=html --cov-report=term

lint:
	poetry run ruff check app tests
	poetry run black --check app tests

format:
	poetry run black app tests
	poetry run ruff check --fix app tests

type-check:
	poetry run mypy app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

