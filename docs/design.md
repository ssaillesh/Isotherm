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

**Scenarios.** Every state array in `Cluster` has a leading scenario axis: (scenarios, GPUs)
or (scenarios, nodes). The live row is one scenario. `Cluster.branch(trial, nodes)` returns
one scenario per candidate node, each with the trial workload added there, and `step`
advances them all with the same code. Scenarios share the clock and the workload schedule;
only the trial workload differs.

**What `step` avoids repeating**, without changing any arithmetic:

- *Power and inlets.* They depend only on utilization and on each other. If an update
  returns exactly the inlets it was given, the next one would repeat itself, so it is skipped
  until a bind, unbind or expiry changes utilization. A row that is not throttling settles
  on the second update after a change. A throttling row keeps updating until it stops moving.
- *Expiry.* End times sit in a heap, so each substep looks only at the earliest one, not at
  every workload. An entry whose workload was unbound early is dropped when it surfaces.
- *The decay factor* exp(−h/RC) depends only on the GPU and the substep length, so it is
  computed once per length.

Two places choose the slower of two equivalent formulas, because the faster one rounds
differently in the last bit and results must reproduce exactly: per-server power is a running
total in GPU order (not `sum`), and inlets use one matrix-vector product per scenario (not
one matrix-matrix product). `sim/tests/test_physics.py` checks `step` against the substep
written out longhand, and a branch against separate copies, for exact equality.

## Measured speedup

The speed changes in the server (no Prometheus refresh with metrics off), in
`experiments/simulate.py` (no `GetState` after `Step`) and in `sim/physics` (the three items
above, plus batched rollouts through scenarios) were all committed together in `43fda44`, so
they are measured together, not one by one. Before is `5d260c5`, the parent of `43fda44`.
After is `cda159a`, whose sim and experiments code is the same as `43fda44`'s.

`experiments/check_equivalence.py` exports each commit with `git archive`, starts that
commit's simulator with metrics off, and drives it with that commit's own experiments code.
It runs all six policies on the load 0.3 and load 0.9 seed 0 sweep traces through
`sweep.run_steady_state` (12 runs per commit), then compares every column of every row:

```bash
make sweep-traces   # if data/sweep/ is missing
cd experiments && uv run python check_equivalence.py refs 5d260c5 cda159a \
  --expect ../docs/results/v1_supply18/sweep.csv
```

**Results are unchanged.** All 12 rows are identical in every column between the two commits,
and both match the same 12 rows of the v1 sweep (`docs/results/v1_supply18/sweep.csv`).

**Run time.** Wall-clock seconds per run, from that command on 2026-11-13. Machine: Apple M1,
8 GB, macOS 15.3.1, Python 3.11.14. The 24 runs ran one after another and each was timed once.
The rows are simulated results. The times are real measurements of the simulator code.

| Policy | 30%, before | 30%, after | Speedup | 90%, before | 90%, after | Speedup |
|---|---|---|---|---|---|---|
| random | 49.3 | 18.4 | 2.7x | 19.6 | 10.6 | 1.9x |
| binpack | 46.8 | 17.4 | 2.7x | 17.8 | 10.6 | 1.7x |
| spread | 47.7 | 18.0 | 2.6x | 19.6 | 9.6 | 2.0x |
| coolest-inlet | 45.2 | 17.6 | 2.6x | 17.9 | 8.8 | 2.0x |
| isotherm | 234.1 | 33.4 | 7.0x | 79.7 | 19.2 | 4.1x |
| ortools | 270.8 | 69.7 | 3.9x | 122.6 | 38.1 | 3.2x |
| **all six** | 693.7 | 174.5 | 4.0x | 277.1 | 97.0 | 2.9x |

All 12 runs together: 970.8 s before, 271.5 s after (3.6x).

Single timings move with whatever else the machine is doing. An earlier run the same day,
using a one-off script that made the same calls, gave identical rows and totals of 945.5 s
before and 229.4 s after (4.1x). The largest gap between the two runs was ortools at 30% after
the fixes: 38.8 s then and 69.7 s here. The cause was not investigated. Read the speedups as
about 2 to 3x for the cheap policies and 3 to 7x for isotherm and ortools, not as exact
figures.

**Full sweep estimate (interpolated, not measured).** Loads 0.5 and 0.7 and seeds 1 to 4 were
not timed. Assuming each seed takes as long as seed 0, and that the run time at 0.5 and 0.7
lies on a straight line between 0.3 and 0.9, the 120-run sweep would take
5 × 2 × (174.5 + 97.0) s ≈ 45 minutes after the fixes, against 5 × 2 × (693.7 + 277.1) s
≈ 2.7 hours before. With the earlier run's totals, the after figure is about 38 minutes.
That is under an hour, so parallel simulator instances are not needed.

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

- **rollout** (default, exact): branches the row into one scenario per candidate node, adds the pod to each, and simulates them together over `horizon_s` (default 600 s). The pod stays for the whole horizon; workloads already running expire on schedule.
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
finishes (`RESUME=1 make sweep` continues a partial file; it refuses a file written with
different columns). Also `sweep_summary.csv` (mean and sample std over seeds) and `sweep.png`.

### Limit-violation metrics

How far and how long limits are broken, aggregated in `experiments/` from the temperatures
the sim already returns after each `Step`. The limits are the sim's own (`inlet_limit_c` and
`throttle_c` in each `NodeState`, which it reads from `configs/`), so `experiments/` neither
hardcodes nor recomputes them.

| Column | Definition |
|---|---|
| `inlet_exceedance_cs` | Σ over servers and ticks of max(0, `inlet_c` − `inlet_limit_c`) × tick, in °C·s |
| `time_over_inlet_s` | Seconds in which any server's inlet is strictly above its `inlet_limit_c` |
| `throttled_gpu_s` | GPU-seconds in which a GPU's power was capped (the sim's `throttled` flag) |
| `gpu_exceedance_cs` | Σ over GPUs and ticks of max(0, `temp_c` − `throttle_c`) × tick, in °C·s |

**End-of-tick rule.** A tick covers (t − tick, t], and the sim reports temperatures only at
its end, so each tick's end-of-tick value counts for the whole tick (value × `tick_s`). This
is the same rule `throttled_gpu_s` already used. One server 1°C over its limit at the end of
one 10 s tick is exactly 10 °C·s. A breach that starts and ends inside one tick is not seen,
and a breach that starts mid-tick counts from the tick's start, so at 10 s ticks the totals
are only as fine as 10 s.

**Expect `gpu_exceedance_cs` near zero.** The sim caps GPU power so the junction's steady
state never passes `throttle_c` (`physics/thermal.py`, `throttle`), using the inlet from the
previous substep. A GPU only goes above `throttle_c` by the little its inlet rises within
one substep, or if the inlet itself is above `throttle_c`. A GPU near its limit shows up
as `throttled_gpu_s` (lost power) rather than as degrees over. The column is kept so that
claim is measured rather than assumed.

### Paired comparison (experiments/compare.py)

`make compare RESULTS=<dir>` reads `<dir>/sweep.csv` and writes `<dir>/compare.csv`:
for each target load, other policy and metric, the per-seed difference
`COMPARE_POLICY` (default isotherm) minus other, as mean and sample std over the seeds,
plus two counts. `lower` is the number of seeds where the chosen policy was strictly lower.
`better` uses a direction per metric: lower is better for temperatures, exceedance,
throttling and wait, higher for `min_headroom_c`. It is blank for columns with no agreed
direction (utilisation, `window_placed`, `sim_end_s`). A mean gap no larger than its std is
labelled "no difference" (that includes every difference being 0). Pairing by seed takes
out the trace-to-trace variation that both policies share. Columns that describe the trace (offered load, the window, job
counts) are the same for every policy and are skipped. The CSV is read by column name, so
a sweep written before a metric existed compares on the metrics it has. Each scenario has
its own results directory, so supply temperatures are compared by running it per directory.

## Open questions

- With the current `configs/`, neither limit binds even at full load (see `assumptions.md`). We need to decide which parameter defines the stressed regime.
- Should infeasible nodes be filtered out (a Filter plugin) and not just scored last?
- Replay-time sim clock vs. scheduler wall clock: binds that arrive between ticks are applied at the next `Step`.
- Calibrating the recirculation parameters needs CFD or inlet sensor data.
