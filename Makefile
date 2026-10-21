init:
	uv sync --extra dev

run:
	uv run python app.py

chat:
	uv run python chat_cli.py

reindex:
	rm -rf chroma_db
	uv run python -m scripts.reload_db

check:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

test-live:
	RUN_LIVE_TESTS=1 uv run pytest -m live

.PHONY: chat check init run reindex test test-live
