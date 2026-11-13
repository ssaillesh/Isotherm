# Isotherm

Thermal-aware GPU scheduling. A kube-scheduler plugin (Go) asks a rack-row thermal
simulator (Python) how hot the row would get for each candidate node, and places pods
where the margin to the thermal limits is largest.

```
isotherm/
  proto/          isotherm.proto: the only contract between components
  configs/        rack.yaml, gpu_specs.yaml: the single source of truth for the physical row
  sim/            thermal simulator: physics/, surrogate/, server.py, tests/
  scheduler/      Go kube-scheduler plugin: cmd/, pkg/plugin/, tests
  replay/         trace loader and pod replayer (drives the kwok cluster and the sim clock)
  experiments/    baseline policies, OR-Tools batch placement, plots
  ml/             surrogate dataset generation (from the sim's rollout oracle) and training
  deploy/         kwok node generator, scheduler config, compose, Prometheus, Grafana
  data/           traces, datasets, models (gitignored)
  docs/           design.md, assumptions.md, results/
```

## The four rules

1. **One contract.** Components talk only through `proto/isotherm.proto`. Stubs are generated (`make proto`), gitignored, and never hand-edited.
2. **One config.** The row layout and GPU specs live only in `configs/`. Every value there has a source logged in [docs/assumptions.md](docs/assumptions.md).
3. **One owner for the physics.** Only `sim/` computes temperatures. The scheduler, experiments, ml and replay all ask the sim over gRPC.
4. **Each folder stands alone.** Every component has its own env (uv project or Go module), tests, Dockerfile, and CI job. There are no imports across components.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python envs; one per component)
- Go 1.23+ and `protoc` for the scheduler (`brew install go protobuf`)
- Optional: Docker, [kwok/kwokctl](https://kwok.sigs.k8s.io/) and kubectl for the cluster workflow

## Quick start

```bash
make setup            # install every component's deps + Go protoc plugins
make proto            # generate stubs from proto/isotherm.proto
make test             # each component's lint + tests, one after another

make sim              # terminal 1: simulator on :50051, Prometheus metrics on :9100
make trace            # synthetic trace -> data/synthetic.csv
make baselines        # every policy vs the sim -> docs/results/baselines.csv + plots
```

Train the learned surrogate (the rollout oracle provides the labels):

```bash
make dataset && make train && make sim-learned
```

Full Kubernetes path (kwok fakes the nodes, so nothing actually runs):

```bash
make cluster-up       # kwok cluster + one fake node per server in configs/rack.yaml
make sim              # terminal 1
make scheduler        # terminal 2: isotherm-scheduler alongside default-scheduler
make replay           # terminal 3: replay the trace through isotherm-scheduler
SCHEDULER=default-scheduler make replay   # baseline run on the same cluster
```

Dashboards: `make up` starts the sim with Prometheus and Grafana (http://localhost:3000).

`make help` lists every target.

## Changing things

| You want to... | Do this |
|---|---|
| Change the row or the GPUs | Edit `configs/`, then update the matching rows in `docs/assumptions.md`. |
| Change the contract | Edit `proto/isotherm.proto` (add fields; don't renumber), run `make proto` and `make test`. |
| Change the physics | Only in `sim/physics/`. If surrogate inputs change, bump `FEATURE_SCHEMA` in `sim/surrogate/features.py` and retrain. |
| Bump Kubernetes | `make pin-k8s K8S_VERSION=1.31.x`. Note that the framework API changes in 1.34+. |
