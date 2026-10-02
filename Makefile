.PHONY: help up build build-model down db worker backend frontend test lint clean prune reset-all backup glyph-chats glyph-chat glyph-export

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
	@echo "  make prune        Remove ALL unused images + build cache (frees disk; keeps volumes)"
	@echo "  make backup       Back up the database (local copy + S3) — see docs/backups.md"
	@echo "  make glyph-chats  List the 20 most recent Glyph Chat conversations"
	@echo "  make glyph-chat ID=<id>  Show one Glyph Chat conversation"
	@echo "  make glyph-export Export every Glyph Chat message to a CSV file"
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

backup:
	./scripts/backup_db.sh

PSQL = docker compose exec -T postgres psql -U lora -d lora

glyph-chats:
	$(PSQL) -c "SELECT c.id, c.created_at, c.updated_at, count(m.id) AS messages FROM glyph_conversations c LEFT JOIN glyph_messages m ON m.conversation_id = c.id GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 20;"

glyph-chat:
	@test -n "$(ID)" || (echo "usage: make glyph-chat ID=<conversation id>"; exit 1)
	$(PSQL) -c "SELECT created_at, role, adapter_id, content, error FROM glyph_messages WHERE conversation_id = '$(ID)' ORDER BY created_at;"

glyph-export:
	$(PSQL) -c "\copy (SELECT conversation_id, created_at, role, content, adapter_id, adapter_run_id, error FROM glyph_messages ORDER BY conversation_id, created_at) TO STDOUT WITH CSV HEADER" > glyph_chats_$$(date +%Y%m%d).csv
	@echo "Wrote glyph_chats_$$(date +%Y%m%d).csv"

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
