# Build from the repo root:  docker build -f ml/Dockerfile -t isotherm-ml .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app/ml
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH

COPY ml/pyproject.toml ml/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY proto /app/proto
COPY ml /app/ml
RUN uv run --no-project --with grpcio-tools python -m grpc_tools.protoc \
      -Igen=../proto --python_out=. --grpc_python_out=. ../proto/isotherm.proto

ENTRYPOINT ["python", "dataset.py"]
