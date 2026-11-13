# Results v1: load sweep at 18°C supply air (simulated)

Every number in this folder is simulated. Nothing here was measured on real hardware.

This is the first full load sweep: six placement policies on the same synthetic traces at
four target loads, five seeds each, 120 runs in total.

## Files

| File | What it is |
|---|---|
| `sweep.csv` | One row per (policy, target load, seed): 120 rows |
| `sweep_summary.csv` | Mean and sample standard deviation over the 5 seeds, per (policy, target load): 24 rows |
| `sweep.png` | The summary plotted against target load, one line per policy |
| `configs/` | Copy of `configs/rack.yaml` and `configs/gpu_specs.yaml` as used for this sweep |

## What was run

| Setting | Value |
|---|---|
| Code | commit `43fda44` (the results were committed next, in `9e408bb`) |
| Configs | `configs/` at that commit, copied into `configs/` here |
| Row | 10 racks x 4 servers x 8 GPUs = 320 GPUs (racks 0 to 4 `h100_sxm`, racks 5 to 9 `a100_sxm`) |
| Supply air | 18.0°C (`cooling.supply_temp_c`) |
| Inlet limit | 27.0°C (`cooling.inlet_limit_c`) |
| Throttle temperature | 87°C for `h100_sxm`, 85°C for `a100_sxm` |
| Policies | random, binpack, spread, coolest-inlet, isotherm (rollout predictor), ortools |
| Target loads | 0.3, 0.5, 0.7, 0.9 of the row's GPU capacity |
| Seeds | 0, 1, 2, 3, 4 |
| Trace | synthetic, 2,000 jobs per (load, seed), from `replay/loader.py sweep` |
| Tick | 10 s (`sweep.py --tick-s` default) |
| Measured window | 10% to 90% of each trace's arrival period (`WINDOW` in `experiments/sweep.py`) |
| Python | 3.11.14, dependencies pinned by each component's `uv.lock` (OR-Tools 9.15.6755) |

## Commands

These are the Makefile defaults, and the CSV matches them (120 rows, the loads, seeds and
job counts above).

```bash
make sim METRICS_PORT=0   # terminal 1: simulator, Prometheus refresh off
make sweep-traces         # terminal 2: 20 traces + index.csv -> data/sweep/
make sweep                # 120 runs -> docs/results/sweep.csv, sweep_summary.csv, sweep.png
```

`make sweep-traces` expands to
`loader.py sweep --loads 0.3 0.5 0.7 0.9 --seeds 5 --jobs 2000 --out ../data/sweep`, and
`make sweep` to `sweep.py --sim localhost:50051 --index ../data/sweep/index.csv --out
../docs/results/sweep.csv` followed by `plots.py ../docs/results/sweep_summary.csv --sweep`.
The files were then moved from `docs/results/` into this folder.

## Rerunning it

To rerun without touching this folder, write to a scratch directory and compare:

```bash
git checkout 43fda44      # only needed once the code or configs have changed since v1
make sim METRICS_PORT=0   # terminal 1
make sweep-traces         # terminal 2
make sweep RESULTS=data/v1_rerun
git checkout main
diff data/v1_rerun/sweep.csv docs/results/v1_supply18/sweep.csv && echo identical
diff data/v1_rerun/sweep_summary.csv docs/results/v1_supply18/sweep_summary.csv && echo identical
```

`data/` is gitignored, so the rerun output survives the checkout and never enters git.

## How reproducibility was checked (2026-11-13)

The full sweep was not rerun for this check. What was checked:

- **Traces.** `loader.py sweep` was run again with the arguments above into a scratch
  directory. All 20 traces and `index.csv` are byte-identical to the ones the sweep used.
- **Runs.** 12 of the 120 runs were rerun against a fresh `make sim METRICS_PORT=0`, through
  the same `run_steady_state` function `sweep.py` calls: all six policies on the load 0.9,
  seed 0 trace and all six on the load 0.3, seed 0 trace. Every column of all 12 rows is
  exactly equal to the row in `sweep.csv`, including the ortools rows (the solver gave the
  same placements both times).
- **Not checked.** The other 108 runs, and the plot, were not regenerated.

## Headline numbers

No run came near a thermal limit:

| Quantity | Worst value over all 120 runs | Limit |
|---|---|---|
| Throttled GPU-seconds | 0 | 0 is the goal |
| Peak inlet temperature | 20.43°C | 27°C (smallest headroom 6.57°C) |
| Peak GPU temperature | 72.79°C | 85°C and 87°C throttle temperatures |
| Window jobs left unplaced | 0 | |

Because no limit was ever binding, this sweep cannot show whether isotherm's prediction
helps when cooling is tight. That is the stage 2 question.

## Paired comparison: isotherm minus each other policy

Each difference is taken on the same trace (same target load and seed), so trace-to-trace
variation cancels. Each cell is the mean ± sample standard deviation of the 5 per-seed
differences, then the number of seeds where isotherm was strictly lower. Negative means
isotherm was lower (cooler, or shorter wait). A gap smaller than its standard deviation is
labelled "no difference"; where both are zero the two policies gave identical values.

Peak GPU temperature (°C):

| Isotherm vs | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| binpack | -1.29 ± 0.20 (5/5) | -0.90 ± 0.15 (5/5) | -0.69 ± 0.31 (5/5) | -0.34 ± 0.08 (5/5) |
| spread | -0.47 ± 0.15 (5/5) | -0.39 ± 0.27 (5/5) | no difference: -0.30 ± 0.40 (3/5) | no difference: -0.04 ± 0.20 (2/5) |
| random | -0.69 ± 0.29 (5/5) | -0.46 ± 0.42 (5/5) | -0.26 ± 0.24 (5/5) | no difference: -0.13 ± 0.25 (4/5) |
| coolest-inlet | no difference: +0.02 ± 0.17 (3/5) | no difference: +0.01 ± 0.32 (3/5) | no difference: -0.15 ± 0.31 (3/5) | no difference: -0.18 ± 0.19 (4/5) |
| ortools | no difference: +0.01 ± 0.14 (4/5) | no difference: -0.03 ± 0.28 (3/5) | no difference: -0.03 ± 0.38 (3/5) | no difference: +0.08 ± 0.11 (1/5) |

Mean temperature of busy GPUs (°C):

| Isotherm vs | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| binpack | -2.97 ± 0.05 (5/5) | -2.02 ± 0.14 (5/5) | -1.00 ± 0.14 (5/5) | -0.16 ± 0.15 (5/5) |
| spread | -0.60 ± 0.07 (5/5) | -0.81 ± 0.08 (5/5) | -0.71 ± 0.08 (5/5) | -0.25 ± 0.14 (5/5) |
| random | -0.70 ± 0.10 (5/5) | -0.47 ± 0.04 (5/5) | -0.38 ± 0.10 (5/5) | no difference: -0.10 ± 0.12 (4/5) |
| coolest-inlet | -0.32 ± 0.03 (5/5) | -0.17 ± 0.05 (5/5) | -0.08 ± 0.03 (5/5) | no difference: -0.02 ± 0.03 (3/5) |
| ortools | -0.21 ± 0.07 (5/5) | -0.09 ± 0.06 (5/5) | -0.06 ± 0.02 (5/5) | no difference: +0.06 ± 0.17 (2/5) |

Peak inlet temperature (°C):

| Isotherm vs | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| binpack | -1.12 ± 0.05 (5/5) | -0.78 ± 0.24 (5/5) | no difference: -0.06 ± 0.08 (3/5) | +0.053 ± 0.049 (1/5) |
| spread | -0.65 ± 0.14 (5/5) | -0.48 ± 0.30 (4/5) | no difference: +0.08 ± 0.08 (1/5) | no difference: +0.05 ± 0.09 (1/5) |
| random | -0.68 ± 0.05 (5/5) | -0.54 ± 0.25 (5/5) | no difference: +0.04 ± 0.04 (1/5) | no difference: +0.04 ± 0.04 (1/5) |
| coolest-inlet | -0.05 ± 0.04 (4/5) | no difference: -0.06 ± 0.15 (3/5) | no difference: +0.06 ± 0.10 (2/5) | no difference: -0.04 ± 0.04 (4/5) |
| ortools | no difference: +0.01 ± 0.01 (1/5) | -0.11 ± 0.07 (5/5) | +0.18 ± 0.17 (1/5) | +0.11 ± 0.02 (0/5) |

Mean queue wait (s):

| Isotherm vs | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| binpack | no difference: +0.0 ± 0.0 (0/5) | no difference: +0.0 ± 0.1 (0/5) | no difference: +0.5 ± 0.6 (0/5) | +6.4 ± 5.3 (0/5) |
| spread | -0.5 ± 0.3 (5/5) | -14.7 ± 4.4 (5/5) | -52.5 ± 15.2 (5/5) | -280.4 ± 165.2 (5/5) |
| random | no difference: +0.0 ± 0.0 (0/5) | no difference: -1.1 ± 1.8 (5/5) | -15.2 ± 6.2 (5/5) | -100.8 ± 48.5 (5/5) |
| coolest-inlet | no difference: +0.0 ± 0.0 (0/5) | no difference: +0.0 ± 0.1 (0/5) | no difference: +0.4 ± 0.7 (1/5) | no difference: +1.4 ± 1.7 (1/5) |
| ortools | no difference: +0.0 ± 0.0 (0/5) | no difference: -0.0 ± 0.0 (1/5) | no difference: -1.4 ± 1.7 (5/5) | -76.2 ± 58.8 (5/5) |

What the tables say:

- Against binpack, isotherm is cooler on peak GPU at every load on every seed. The gap
  shrinks from 1.29°C at 30% to 0.34°C at 90%, and at 90% it costs about 6 s of mean wait.
- Against coolest-inlet, peak GPU is no different at any load. Mean busy GPU temperature is
  slightly lower (0.32°C at 30%) and the gap is gone by 90%. Wait is no different.
- Against ortools, peak GPU is no different at any load, and isotherm's mean wait at 90% is
  about 76 s shorter. At 70% and 90% ortools has the lower peak inlet (by 0.18°C and 0.11°C).
- These tables were computed directly from `sweep.csv`. A permanent script for this
  comparison (`experiments/compare.py`) is planned for stage 2.
