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

## Workload load (replay/loader.py)

The synthetic trace's knob is **offered load**: the average fraction of cluster GPUs the
jobs ask for while they are arriving, `ρ = λ · E[gpu_count · duration] / capacity_gpus`.
Given a target ρ, `synthetic()` sets the Poisson arrival rate λ from the closed-form
expected GPU-seconds per job. Durations are `max(min, Exp(mean))`, so
`E[duration] = min + mean · exp(-min / mean)`. Job sizes do not change with ρ, only arrival
spacing does. `offered_load()` measures ρ back from a trace.

Offered load is not the same as the whole-run average busy fraction. The default trace
(500 jobs, 30 s mean gap) offers about 51% of 320 GPUs. When arrivals stop, the long jobs
still running need about 9,000 s more to finish, and that tail pulls the whole-run average
down to about 32%. Above roughly 100%, or when large jobs fragment nodes, jobs queue, and
achieved utilisation falls below ρ.

Capacity is read from `configs/` (`loader.py --config-dir`, default `../configs`) as
`n_racks × slots_per_rack × gpus_per_node`, today 10 × 4 × 8 = 320, so the trace is always
sized for the row the sim models. `replay` depends on PyYAML for this.

## Load sweep (experiments/sweep.py)

`make sweep-traces` writes one trace per (target load, seed) plus `index.csv`;
`make sweep` runs every policy on each one against the sim. Every sweep result is
simulated, and `sweep.png` is labelled so. Rules, enforced by both CLIs:

- **At least 2,000 jobs per trace and 5 seeds per load.** With 2,000 jobs the measured
  offered load of a seed still lands within about ±6% of the target (0.85 to 0.92 at a
  target of 0.9), which is why it is recorded per seed rather than assumed.
- **Steady state only.** Every metric is taken over the window from 10% to 90% of the
  trace's arrival period (first to last submit). That drops the empty-cluster warm-up and the
  drain after arrivals stop. Ticks count when they end inside the window; waits count for
  jobs submitted inside it. The run stops once the window has ended and all of its jobs
  are placed, so the drain is never simulated.
- **Load is recorded three ways, per seed:** `target_load` (what the generator aimed for),
  `offered_load` (window arrivals' GPU-seconds over window length × capacity), and
  `achieved_utilisation` (mean GPUs allocated per tick over capacity). Capacity comes from
  the sim's `GetState`. The sweep refuses to run if that disagrees with the index.

Utilisation counts the GPUs bound during each tick, not the busy GPUs the sim reports after
`Step`: by then, a job ending on the tick boundary has already expired, and every job would
lose one tick. Below saturation `achieved_utilisation` ≈ `offered_load`. It falls below
once jobs queue or large jobs fragment nodes.

Output: `docs/results/sweep.csv`, one row per (policy, load, seed), appended as each run
finishes (`RESUME=1 make sweep` continues a partial file). Also `sweep_summary.csv` (mean and
sample std over seeds) and `sweep.png`.

## Open questions

- With the current `configs/`, neither limit binds even at full load (see `assumptions.md`). We need to decide which parameter defines the stressed regime.
- Should infeasible nodes be filtered out (a Filter plugin) and not just scored last?
- Replay-time sim clock vs. scheduler wall clock: binds that arrive between ticks are applied at the next `Step`.
- Calibrating the recirculation parameters needs CFD or inlet sensor data.
