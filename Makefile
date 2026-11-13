# Isotherm task runner. `make help` lists targets.
#
# Each component stands alone: `make test-<component>` syncs its own env, regenerates its
# own stubs from proto/isotherm.proto, lints, and tests just that component.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PY_COMPONENTS := sim replay experiments ml
COMPONENTS    := $(PY_COMPONENTS) scheduler deploy
PROTO         := proto/isotherm.proto

SIM_ADDR      ?= localhost:50051
CLUSTER       ?= isotherm
TRACE         ?= data/synthetic.csv
RESULTS       ?= docs/results

PROTOC_GEN_GO_VERSION      := v1.35.2
PROTOC_GEN_GO_GRPC_VERSION := v1.5.1
K8S_VERSION                := 1.31.4

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_%-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@printf "  \033[36m%-20s\033[0m %s\n" "test-<component>" "Test one of: $(COMPONENTS)" \
	  "proto-<component>" "Regenerate one Python component's stubs"

# ---------------------------------------------------------------- setup

.PHONY: setup tools-go
setup: tools-go ## Install every component's dependencies
	@for c in $(PY_COMPONENTS); do echo "== $$c"; (cd $$c && uv sync); done
	cd sim && uv sync --extra learned
	cd scheduler && go mod download

tools-go: ## Install protoc plugins for Go
	go install google.golang.org/protobuf/cmd/protoc-gen-go@$(PROTOC_GEN_GO_VERSION)
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@$(PROTOC_GEN_GO_GRPC_VERSION)

pin-k8s: ## Re-pin k8s.io staging modules (K8S_VERSION=1.31.4)
	cd scheduler && ./hack/pin-k8s.sh $(K8S_VERSION)

# ---------------------------------------------------------------- contract

.PHONY: proto proto-go $(addprefix proto-,$(PY_COMPONENTS))
proto: $(addprefix proto-,$(PY_COMPONENTS)) proto-go ## Regenerate all stubs from the proto

$(addprefix proto-,$(PY_COMPONENTS)): proto-%:
	cd $* && uv run python -m grpc_tools.protoc -Igen=../proto \
	  --python_out=. --grpc_python_out=. --pyi_out=. ../$(PROTO)

proto-go:
	mkdir -p scheduler/gen/isothermpb
	protoc -Iproto \
	  --go_out=scheduler/gen/isothermpb --go_opt=paths=source_relative \
	  --go-grpc_out=scheduler/gen/isothermpb --go-grpc_opt=paths=source_relative \
	  $(PROTO)

# ---------------------------------------------------------------- tests

.PHONY: test lint $(addprefix test-,$(COMPONENTS))
test: $(addprefix test-,$(COMPONENTS)) ## Test every component, one at a time

$(addprefix test-,$(PY_COMPONENTS)): test-%: ## Test one Python component
	cd $* && uv sync --quiet
	$(MAKE) --no-print-directory proto-$*
	cd $* && uv run ruff format --check . && uv run ruff check . && uv run pytest -q

test-scheduler: proto-go ## Test the Go scheduler plugin
	cd scheduler && test -z "$$(gofmt -l $$(find . -name '*.go' -not -path './gen/*'))" \
	  || { gofmt -l .; echo "gofmt needed"; exit 1; }
	cd scheduler && go vet ./... && go test ./...

test-deploy: ## Test the kwok node generator
	cd deploy/kwok && uv run --with pytest --with pyyaml pytest -q test_gen_nodes.py

# ---------------------------------------------------------------- run locally

.PHONY: sim trace baselines plots dataset train
sim: proto-sim ## Run the thermal simulator (gRPC :50051, metrics :9100)
	cd sim && uv run python server.py

trace: ## Generate a synthetic trace into $(TRACE)
	cd replay && uv run python loader.py synthetic --jobs 500 --out ../$(TRACE)

baselines: proto-experiments ## Run every policy against a running sim (needs `make sim`)
	cd experiments && uv run python simulate.py --sim $(SIM_ADDR) --trace ../$(TRACE) \
	  --out ../$(RESULTS)/baselines.csv
	cd experiments && uv run python plots.py ../$(RESULTS)/baselines.csv --out ../$(RESULTS)

dataset: proto-ml ## Build a surrogate dataset from the sim's rollout oracle (needs `make sim`)
	cd ml && uv run python dataset.py --sim $(SIM_ADDR) --out ../data/surrogate.npz

train: ## Train the learned surrogate; serve it with `make sim-learned`
	cd ml && uv run python train.py ../data/surrogate.npz --out ../data/surrogate.joblib

sim-learned: proto-sim ## Run the sim with the learned surrogate as default predictor
	cd sim && uv run --extra learned python server.py --predictor learned \
	  --model-path ../data/surrogate.joblib

# ---------------------------------------------------------------- kwok cluster

.PHONY: cluster-up cluster-down nodes scheduler replay
cluster-up: ## Create the kwok cluster and its fake GPU nodes
	kwokctl create cluster --name $(CLUSTER)
	$(MAKE) --no-print-directory nodes

nodes: ## (Re)apply kwok nodes generated from configs/
	uv run deploy/kwok/gen_nodes.py | kubectl --context kwok-$(CLUSTER) apply -f -

cluster-down: ## Delete the kwok cluster
	kwokctl delete cluster --name $(CLUSTER)

scheduler: proto-go ## Run isotherm-scheduler against the kwok cluster (needs `make sim`)
	kwokctl get kubeconfig --name $(CLUSTER) > deploy/k8s/kubeconfig.local
	KUBECONFIG_PATH=$(CURDIR)/deploy/k8s/kubeconfig.local \
	  envsubst < deploy/k8s/scheduler-config.yaml > deploy/k8s/scheduler-config.local.yaml
	cd scheduler && ISOTHERM_SIM_ADDR=$(SIM_ADDR) go run ./cmd/isotherm-scheduler \
	  --config ../deploy/k8s/scheduler-config.local.yaml

replay: proto-replay ## Replay $(TRACE) into kwok (SCHEDULER=default-scheduler for baseline)
	cd replay && uv run python replayer.py ../$(TRACE) --sim $(SIM_ADDR) \
	  --scheduler $${SCHEDULER:-isotherm-scheduler} \
	  --kubeconfig ../deploy/k8s/kubeconfig.local \
	  --out ../$(RESULTS)/replay-$${SCHEDULER:-isotherm-scheduler}.csv

# ---------------------------------------------------------------- containers

.PHONY: images up down
images: ## Build every component's image
	@for c in sim replay experiments ml scheduler; do \
	  docker build -f $$c/Dockerfile -t isotherm-$$c . || exit 1; done

up: ## Start sim + Prometheus + Grafana (http://localhost:3000)
	docker compose -f deploy/compose.yaml up --build -d

down: ## Stop the compose stack
	docker compose -f deploy/compose.yaml down

# ---------------------------------------------------------------- housekeeping

.PHONY: clean
clean: ## Remove generated stubs and caches
	for c in $(PY_COMPONENTS); do \
	  find $$c/gen -type f ! -name '__init__.py' -delete; done
	rm -rf scheduler/gen
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
