# PLAN.md: Isotherm, stage 2

Read with CLAUDE.md. Do **one session per conversation**, in order. CLAUDE.md's ground rules
apply to every session. In addition, in every session:

- **Never commit or push.** When work is ready, print the exact `git add` / `git commit`
  commands for the owner to run.
- **Plan first.** List files to touch and tests to add, then wait for the owner's go-ahead.
- **Stop and ask** before adding a dependency, changing the proto, or adding any physical
  number that has no cited source.
- Finish with `make test` and the summary CLAUDE.md asks for.

## Where we are

The v1 load sweep is done (18°C supply, loads 0.3/0.5/0.7/0.9, 6 policies, 5 seeds, 120 runs).
Paired per-seed differences, isotherm minus baseline:

| vs | Peak GPU, 30% → 90% | Mean busy GPU, 30% → 90% | Wait at 90% |
|---|---|---|---|
| binpack | -1.29 → -0.34°C, 5/5 seeds each load | -2.97 → -0.16°C, 5/5 | +6 s |
| random | -0.69 → -0.13°C | -0.70 → -0.10°C | -101 s |
| coolest-inlet | tie (within noise) | -0.32 → -0.02°C | tie |
| ortools | tie | -0.21 → +0.06°C | -76 s |

Nothing came near a limit: zero throttling, peak inlet about 20.4°C against a 27°C limit,
peak GPU about 72.8°C against throttle. With limits never binding, isotherm's rollout
prediction cannot show value and behaves like coolest-inlet.

**Stage 2 question: when cooling is tight, does isotherm's prediction beat the
coolest-inlet rule on limit violations, without hurting queue wait?**

---

## Session 1: Save v1 results

Goal: v1 is preserved and documented before anything changes.

1. Make sure `docs/results/` holds `sweep.csv`, `sweep_summary.csv` and the sweep plot; move
   them to `docs/results/v1_supply18/` and record the exact command and config used in a
   short `docs/results/v1_supply18/README.md`.
2. Add a "Results v1" section to the main README: the plot, the paired table above (labelled
   simulated), and two sentences on what it shows, including the coolest-inlet tie.
3. Print the git commands for the owner.

Done when: the owner can rerun v1 from the recorded command and get the same CSV.

## Session 2: Speed fixes (skip any already done)

From the profiling report, in this order, each proven by running all six policies on one 30%
and one 90% trace before and after, requiring every column of `sweep.csv` to match exactly:

a. `sim/server.py`: skip the Prometheus refresh when metrics are off.
b. `experiments/simulate.py`: reuse the state `Step` returns instead of a new `GetState`.
c. `sim/physics`: expiry heap instead of scanning every allocation each substep; cache the
   `exp()` decay factor per step size; remove redundant `_update_power` calls.
d. Propose, then implement after approval: batch rollouts by giving `Cluster.step` a batch
   dimension (single state = batch of one), so there is one code path, not a copy.

Report the speedup after each fix. Done when: exact-match checks pass and the sweep time
estimate is updated in CLAUDE.md.

## Session 3: Limit-violation metrics and paired comparison

Goal: measure how far and how long limits are broken, and make the paired analysis permanent.

1. In `experiments/simulate.py`, from temperatures the sim already returns each tick:
   - `inlet_exceedance_cs`: sum over servers and ticks of max(0, T_inlet - inlet_limit_c) x dt
   - `time_over_inlet_s`: seconds in which any inlet is above `inlet_limit_c`
   - keep `throttled_gpu_s`; add `gpu_exceedance_cs` the same way against each GPU's throttle
     temperature from `configs/gpu_specs.yaml`
   Read limits from configs, never hardcode them. This aggregates sim output, so rule 3 holds.
2. Include the new columns in `sweep.csv` and `sweep_summary.csv`.
3. New `experiments/compare.py`: for each load (and supply, once it exists), the per-seed
   difference between a chosen policy and every other policy on every metric, reported as
   mean, standard deviation and wins out of 5. Output `compare.csv` and a printed table.
   A difference smaller than its standard deviation is labelled "no difference".
4. Tests: a fake run 1°C over the inlet limit for 10 s on one server gives exactly 10 °C·s;
   all-cool runs give 0; `compare.py` on a tiny hand-made CSV gives hand-computed results.
5. Run `compare.py` on the v1 sweep and check it reproduces the table above.

Done when: tests pass and the v1 comparison matches.

## Session 4: Variant configs and the stress sweep setup

Goal: run the same sweep with warmer supply air without ever editing `configs/`.

1. Add `--config-dir` (default `../configs`) to the sim server, sweep, loader and any
   component that reads configs; add a `CONFIG_DIR` variable to the Makefile and pass it
   through `make sim`, `make sweep-traces` and `make sweep`. Results go to a per-scenario
   directory (`RESULTS=docs/results/<scenario>`), and the sweep copies the config directory
   it used into that results directory.
2. Choose stress temperatures from the v1 data, not by guessing. The model shifts every
   temperature with the supply air, so:
   - inlet scenario: supply = 18 + (smallest v1 min_headroom_c across loads), rounded down to
     0.5°C. Expected about 24°C: hottest inlets sit right at the limit.
   - GPU scenario: supply = 18 + (smallest gap between v1 peak GPU and its throttle
     temperature), rounded down to 0.5°C. Expected about 30°C: hottest GPUs reach throttle.
   Show the calculation, then create `data/configs/supply<T>/` for each, a copy of
   `configs/` with only the supply temperature changed.
3. Linearity check: run one trace (load 0.5, seed 0, binpack) at 18°C and at the inlet
   scenario. Peak inlet and peak GPU must rise by the supply difference to within 0.05°C. If
   not, stop and report why before going further.
4. Smoke test: one short trace (200 jobs, 1 seed) per scenario for binpack, coolest-inlet and
   isotherm, end to end, to prove the pipeline works. (The 5 seed rule applies to reported
   results, not smoke tests; label smoke output as such and don't save it to results.)
5. Print the exact overnight commands for the owner: for each scenario, start the sim with
   that config dir, then run the sweep with policies binpack, coolest-inlet, isotherm, loads
   0.3 0.5 0.7 0.9, 5 seeds, 2,000 jobs, with RESUME=1 support.

Done when: linearity check passes, smoke tests pass, and the owner has copy-paste commands.
Total stress sweep: 2 scenarios x 4 loads x 3 policies x 5 seeds = 120 runs.

## Session 5: Read the stress results

Goal: answer the stage 2 question honestly.

1. Run `compare.py` with isotherm as the reference for each scenario and load.
2. Make one plot per scenario: inlet exceedance, throttled GPU seconds, peak temperature and
   mean wait against load, one line per policy, mean and standard deviation.
3. Write `docs/results/stage2.md` stating which outcome happened:
   - **A. Isotherm beats coolest-inlet on violations** with no meaningful wait cost. Report
     the size, the loads where it holds, and the wins out of 5.
   - **B. Tie.** State plainly that the cheap rule matches it under stress, and quantify the
     compute cost of isotherm versus coolest-inlet per placement.
   - **C. Isotherm loses.** Go to Session 5b.
4. Never drop seeds, loads or metrics that look bad. Report everything that was run.

### Session 5b (only for outcomes B or C): check the objective

1. Read the isotherm scoring path (`RolloutPredictor` and the score it returns) and explain
   in plain words what it optimises: mean temperature, peak, margin to limit, or something
   else.
2. If it does not target limit violations, propose a score built on predicted exceedance
   over the rollout horizon (for example, predicted inlet and GPU exceedance, then margin as
   a tie-breaker). Wait for approval.
3. Implement with tests, rerun the stress sweep for isotherm only (other policies are
   unchanged), and update `stage2.md` with before and after.

## Session 6: Cooling energy (only with a sourced efficiency curve)

1. Propose two or three published sources for chiller or CRAH efficiency (COP) as a function
   of supply air temperature, and wait for the owner to pick one. Do not add numbers before
   that.
2. Add `cooling.crah_capacity_kw` and the chosen COP curve to `rack.yaml`, cited in
   `docs/assumptions.md`. The sim reports cooling power each step; supply temperature rises
   when heat exceeds capacity. Tests: energy balance, COP lookup, capacity overflow.
3. Report, for each policy, cooling energy at the warmest supply temperature it tolerates
   with zero violations (from the headroom results), compared with binpack.

## Session 7: Real workload

1. Write download instructions for the Alibaba GPU trace (cluster-trace-gpu-v2020) in
   `docs/data.md`; the owner downloads it. Raw data stays in gitignored `data/`.
2. Add a parser to `replay/loader.py` mapping its fields to the trace format (GPU count,
   duration, utilisation, submit time). Document every mapping decision and anything dropped.
   Test with a small hand-made fixture in the same format.
3. Rerun the most informative scenario from Session 5 on a slice of the real trace, 5
   windows as the 5 seeds, and add the result to `stage2.md`.

## Session 8: Finish and present

1. Make `make test-scheduler` pass locally (protoc plugins via `make tools-go`).
2. One kwok end to end run: same trace through isotherm-scheduler and default-scheduler;
   save a Grafana screenshot to `docs/results/`.
3. README: final results section (stage 1 and 2 plots, one table, a "what I learned"
   paragraph, everything labelled simulated) and a 30-second quick start.
4. Draft one resume bullet for the owner, following his rules: only technologies in bold,
   no dashes, concrete numbers from the results, the word "simulated". Do not exaggerate.
5. Write `docs/interview-notes.md` (gitignore it): derivation of D, why the benefit shrinks
   at high load, the coolest-inlet result, the speed fixes and their evidence, the stress
   test design, and three likely interview questions with honest answers.
