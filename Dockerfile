# FROM baseimage: baseimage is a starting point
FROM python:3.12-slim

WORKDIR /app

RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD is used to START the container AFTER building
# there can be only 1 CMD command in a dockerfile
