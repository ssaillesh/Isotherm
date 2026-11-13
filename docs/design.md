# Isotherm design

## Goal

Place GPU pods so the row stays furthest from its thermal limits. There are two limits:
server inlet air against the ASHRAE recommended maximum (`inlet_limit_c`), and GPU junction
temperature against each card's throttle point (`throttle_c`).

## Components and data flow

```
                      configs/ (rack.yaml, gpu_specs.yaml)
                         │                         │
                         ▼                         ▼
 replay ──pods──► kube-apiserver (kwok)      deploy/kwok/gen_nodes.py ──► fake Nodes
   │                     │
   │                     ▼
   │              isotherm-scheduler (Go)
   │                PreScore ──ScoreCandidates──┐
   │                PostBind ──ApplyEvent(BIND)─┤
   │                                            ▼
   ├──Step / ApplyEvent(UNBIND)──────────►  sim (Python, gRPC :50051)
   │                                            ▲
 experiments ──GetState/ScoreCandidates/───────┤
               ApplyEvent/Step/Reset            │
 ml ──ScoreCandidates(predictor="rollout")──────┘
```

Every arrow into `sim` is an RPC in `proto/isotherm.proto`. No component imports another.

- **sim** owns all physics: per-server power, the airflow model, GPU junction temperatures, and the predictors.
- **scheduler** sends one `ScoreCandidates` per pod in PreScore. It re-encodes the returned `headroom_c` as an integer score and min-max normalises it. If the sim is unreachable, it returns `Skip`, so other plugins still decide. It never computes a temperature.
- **replay** is the clock in kwok runs. It submits pods, notices binds, deletes finished pods and sends UNBIND, and calls `Step` every tick. Replayed pods carry no duration annotation, so the sim never expires them on its own.
- **experiments** runs the same trace through the sim directly, with no Kubernetes, under each policy. That is fast enough to sweep.
- **ml** trains a fast surrogate on what the rollout oracle returns. Both the inputs (`CandidateScore.features`) and the labels come from the sim. The feature definition lives in `sim/surrogate/features.py`, versioned by `FEATURE_SCHEMA`.

## Physics (sim/physics)

Per substep (≤ 5 s):

1. GPU power = `idle_gpu_fraction·TDP + util·(1 − idle_gpu_fraction)·TDP`, capped so the junction's steady state never exceeds `throttle_c` (throttling).
2. Server power P = `base_w` + Σ GPU power.
3. Inlet temperatures: T_in = T_supply + D·P, where D = (K − AᵀK)⁻¹Aᵀ comes from the recirculation matrix A. See `physics/airflow.py` for the full construction from `max_fraction`, `length_scale`, `height_bias` and `end_bias`.
4. Exhaust T_out = T_in + P / (airflow·c_p).
5. GPU junction: C·dT/dt = P − (T − T_in)/R, stepped exactly with an exponential, so it is stable for any step size.

## Scoring

`ScoreCandidates` returns, per candidate:

| Field | Meaning |
|---|---|
| `inlet_headroom_c` | `inlet_limit_c` minus the hottest inlet in the row, over the horizon. |
| `gpu_headroom_c` | The smallest `throttle_c − T` across the row's GPUs, over the horizon. |
| `headroom_c` | The minimum of the two (the binding margin). |
| `feasible` | Free GPUs exist, no inlet goes over its limit, and no GPU throttles. |

Margins are row-wide because recirculation couples every server.

Predictors:

- **rollout** (default, exact): copies the row, binds the pod, and simulates over `horizon_s` (default 600 s).
- **learned**: a gradient-boosted regressor trained by `ml/` on rollout labels. It predicts both headrooms from the feature vector.

## Open questions

- With the current `configs/`, neither limit binds even at full load (see `assumptions.md`). We need to decide which parameter defines the stressed regime.
- Should infeasible nodes be filtered out (a Filter plugin) and not just scored last?
- Replay-time sim clock vs. scheduler wall clock: binds that arrive between ticks are applied at the next `Step`.
- Calibrating the recirculation parameters needs CFD or inlet sensor data.
