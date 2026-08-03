.PHONY: backend-sync backend-run backend-test frontend-install frontend-run fix check pre-commit tree
IGNORE := .git|__pycache__|node_modules|uploads|qdrant

SNAPSHOT_FILE := project_snapshot.md

# ==========================================================
# Backend
# ==========================================================
backend-sync:
	cd backend && uv lock && uv sync

backend-run:
	$(MAKE) backend-sync
	cd backend && docker compose up -d qdrant
	@echo "Waiting for Qdrant to start..."
	@sleep 3
	@-cmd.exe /c start "http://localhost:6333/dashboard#/collections" 2>/dev/null || python3 -m webbrowser "http://localhost:6333/dashboard#/collections" >/dev/null 2>&1 || true
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0

backend-test:
	$(MAKE) backend-sync
	cd backend && uv run pytest

# ==========================================================
# Frontend
# ==========================================================
frontend-install:
	cd frontend && pnpm install

frontend-run:
	cd frontend && pnpm dev

# ==========================================================
# Quality
# ==========================================================
fix:
	cd backend && uv run ruff format .
	cd backend && uv run ruff check . --fix

check:
	cd backend && uv run ruff format --check .
	cd backend && uv run ruff check .
	cd backend && uv run mypy .

# 	cd frontend && pnpm typecheck
# 	cd frontend && pnpm lint

pre-commit:
	cd backend && uv run pre-commit run --config ../.pre-commit-config.yaml --all-files

# =========================================================
# Misc
# ==========================================================
tree:
	rm -rf .ruff_cache backend/__pycache__ backend/.pytest_cache backend/.mypy_cache
	tree -I '$(IGNORE)' backend frontend

tree-s:
	rm -rf .ruff_cache backend/__pycache__ backend/.pytest_cache backend/.mypy_cache
	tree -I '$(IGNORE)' backend frontend > $(SNAPSHOT_FILE)
	@echo "✅ Tree saved to $(SNAPSHOT_FILE)"
