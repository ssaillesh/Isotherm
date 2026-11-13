# Build from the repo root:  docker build -f experiments/Dockerfile -t isotherm-experiments .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app/experiments
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH

COPY experiments/pyproject.toml experiments/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY proto /app/proto
COPY experiments /app/experiments
RUN uv run --no-project --with grpcio-tools python -m grpc_tools.protoc \
      -Igen=../proto --python_out=. --grpc_python_out=. ../proto/isotherm.proto

ENTRYPOINT ["python", "simulate.py"]
