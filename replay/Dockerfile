# Build from the repo root:  docker build -f replay/Dockerfile -t isotherm-replay .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app/replay
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH

COPY replay/pyproject.toml replay/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY proto /app/proto
# loader.py reads cluster capacity from ../configs (its --config-dir default).
COPY configs /app/configs
COPY replay /app/replay
RUN uv run --no-project --with grpcio-tools python -m grpc_tools.protoc \
      -Igen=../proto --python_out=. --grpc_python_out=. ../proto/isotherm.proto

ENTRYPOINT ["python", "replayer.py"]
