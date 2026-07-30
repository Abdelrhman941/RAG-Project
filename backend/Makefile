.PHONY: sync run fix check pre-commit

# ---- Install / update dependencies --------
sync:
	uv lock
	uv sync

# ---- Run development server --------
run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0

# ---- Automatically fix lint issues and format code --------
fix:
	uv run ruff format .
	uv run ruff check . --fix

# ---- CI-style checks (no modifications) --------
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy .
# 	uv run pytest

# ---- Run all pre-commit hooks --------
pre-commit:
	uv run pre-commit run --all-files
