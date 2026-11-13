# Build from the repo root:  docker build -f scheduler/Dockerfile -t isotherm-scheduler .
FROM golang:1.23 AS build

RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
 && rm -rf /var/lib/apt/lists/* \
 && go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35.2 \
 && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

WORKDIR /src/scheduler
COPY scheduler/go.mod scheduler/go.sum ./
RUN go mod download

COPY proto /src/proto
COPY scheduler /src/scheduler
# Stubs are generated at build time from the contract, never copied in.
RUN mkdir -p gen/isothermpb && protoc -I../proto \
      --go_out=gen/isothermpb --go_opt=paths=source_relative \
      --go-grpc_out=gen/isothermpb --go-grpc_opt=paths=source_relative \
      ../proto/isotherm.proto \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/isotherm-scheduler ./cmd/isotherm-scheduler

FROM gcr.io/distroless/static:nonroot
COPY --from=build /out/isotherm-scheduler /usr/local/bin/isotherm-scheduler
ENTRYPOINT ["/usr/local/bin/isotherm-scheduler"]
