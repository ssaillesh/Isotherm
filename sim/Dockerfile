# Build from the repo root:  docker build -f sim/Dockerfile -t isotherm-sim .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app/sim
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH \
    ISOTHERM_CONFIG_DIR=/app/configs

COPY sim/pyproject.toml sim/uv.lock ./
RUN uv sync --frozen --no-dev --extra learned --no-install-project

COPY proto /app/proto
COPY configs /app/configs
COPY sim /app/sim
# Stubs are generated at build time from the contract, never copied in.
RUN uv run --no-project --with grpcio-tools python -m grpc_tools.protoc \
      -Igen=../proto --python_out=. --grpc_python_out=. ../proto/isotherm.proto

EXPOSE 50051 9100
ENTRYPOINT ["python", "server.py"]
CMD ["--port", "50051", "--metrics-port", "9100"]
