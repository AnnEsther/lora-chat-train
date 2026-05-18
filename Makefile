.PHONY: help up build build-model down db worker backend frontend test lint clean prune reset-all

help:
	@echo "LoRA Chat & Train — development commands"
	@echo ""
	@echo "  make up           Start all services (no rebuild)"
	@echo "  make build        Rebuild all Docker images (uses layer cache)"
	@echo "  make build-model  Rebuild only the model_server image"
	@echo "  make down         Stop all services"
	@echo "  make db           Start only postgres + redis"
	@echo "  make init-db      Initialise the database schema"
	@echo "  make reset-all    Clear all sessions and adapters (start fresh)"
	@echo "  make prune        Remove ALL unused images + build cache (frees disk)"
	@echo "  make backend      Start FastAPI backend (local, no Docker)"
	@echo "  make worker       Start Celery worker (local, no Docker)"
	@echo "  make frontend     Start Next.js frontend (local)"
	@echo "  make test         Run pytest unit tests"
	@echo "  make lint         Run ruff linter over Python code"
	@echo "  make clean        Remove __pycache__ directories"

up:
	docker compose up -d

build:
	docker compose build

build-model:
	docker compose build model_server

down:
	docker compose down

prune:
	@echo "Pruning unused Docker images, build cache, and stopped containers..."
	docker image prune -a -f
	docker builder prune -a -f
	docker container prune -f
	@echo "Done. Run 'df -h /' to check disk usage."

db:
	docker compose up -d postgres redis

init-db:
	docker compose run --rm backend python -m scripts.init_db

reset-all:
	docker compose run --rm backend python -m scripts.reset_all

backend:
	cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

worker:
	cd worker && celery -A worker.tasks worker --loglevel=info --pool=solo

frontend:
	cd frontend && npm run dev

test:
	cd backend && python -m pytest tests/ -v

lint:
	ruff check backend/ worker/ training/ shared/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
