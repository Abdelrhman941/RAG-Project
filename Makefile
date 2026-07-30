.PHONY: backend-sync backend-run frontend-install frontend-run fix check pre-commit

# ==========================================================
# Backend
# ==========================================================
backend-sync:
	cd backend && uv lock && uv sync

backend-run:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0

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

	cd frontend && pnpm typecheck
	cd frontend && pnpm lint

pre-commit:
	cd backend && uv run pre-commit run --config ../.pre-commit-config.yaml --all-files
