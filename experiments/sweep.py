"""Load sweep: every policy on every (target load, seed) trace, measured in steady state.

Traces and their index come from `replay/loader.py sweep` (see `make sweep-traces`).
Requires a running simulator (`make sim`).

    uv run python sweep.py --index ../data/sweep/index.csv --out ../docs/results/sweep.csv

Rules (enforced by the CLI):
- every trace has at least MIN_JOBS jobs, and every target load has at least MIN_SEEDS seeds;
- metrics come only from the steady-state window, WINDOW = 10% to 90% of each trace's
  arrival period (first to last submit), which drops the warm-up and the drain;
- each (policy, load, seed) row records the target load, the offered load measured in the
  window, and the utilisation achieved in the window.

Writes one row per (policy, load, seed) to --out, appending as it goes (--resume skips
rows already there), then a mean/std-over-seeds summary next to it (`*_summary.csv`).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from simulate import Trajectory, simulate
from traces import TraceJob, read_trace

log = logging.getLogger("isotherm.experiments.sweep")

MIN_JOBS = 2000
MIN_SEEDS = 5
WINDOW = (0.1, 0.9)  # fractions of the arrival period


@dataclass(frozen=True)
class IndexRow:
    trace: Path
    target_load: float
    seed: int
    jobs: int
    capacity_gpus: int


@dataclass
class SweepRow:
    policy: str
    target_load: float
    seed: int
    offered_load: float  # measured: window arrivals' GPU-seconds / (window * capacity)
    achieved_utilisation: float  # mean allocated GPUs / capacity over the window's ticks
    jobs: int
    window_start_s: float
    window_end_s: float
    window_jobs: int  # jobs submitted in the window
    window_placed: int  # of those, how many were placed before the run stopped
    mean_wait_s: float
    p95_wait_s: float
    peak_temp_c: float
    mean_busy_temp_c: float
    peak_inlet_c: float
    min_headroom_c: float
    throttled_gpu_s: float
    capacity_gpus: int
    sim_end_s: float


KEY_FIELDS = ("policy", "target_load", "seed")
# Columns that describe the run rather than measure it; not averaged in the summary.
ID_FIELDS = {*KEY_FIELDS, "jobs", "capacity_gpus"}


def read_index(path: Path) -> list[IndexRow]:
    """Read loader.py's index.csv. Trace paths are relative to the index's directory."""
    with path.open(newline="") as f:
        return [
            IndexRow(
                trace=path.parent / r["trace"],
                target_load=float(r["target_load"]),
                seed=int(r["seed"]),
                jobs=int(r["jobs"]),
                capacity_gpus=int(r["capacity_gpus"]),
            )
            for r in csv.DictReader(f)
        ]


def check_rules(rows: list[IndexRow], min_jobs: int = MIN_JOBS, min_seeds: int = MIN_SEEDS):
    """Raise ValueError if the sweep is too small to report."""
    if not rows:
        raise ValueError("index lists no traces")
    small = [r.trace.name for r in rows if r.jobs < min_jobs]
    if small:
        raise ValueError(f"traces with fewer than {min_jobs} jobs: {small}")
    seeds: dict[float, set[int]] = defaultdict(set)
    for r in rows:
        seeds[r.target_load].add(r.seed)
    few = {load: len(s) for load, s in seeds.items() if len(s) < min_seeds}
    if few:
        raise ValueError(f"loads with fewer than {min_seeds} seeds: {few}")


def steady_state_window(
    jobs: list[TraceJob], window: tuple[float, float] = WINDOW
) -> tuple[float, float]:
    """[start, end) in sim seconds: the given fractions of the first-to-last arrival span."""
    first, last = min(j.submit_s for j in jobs), max(j.submit_s for j in jobs)
    if last <= first:
        raise ValueError("need at least two distinct submit times")
    lo, hi = window
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"window fractions must satisfy 0 <= lo < hi <= 1, got {window}")
    span = last - first
    return first + lo * span, first + hi * span


def window_offered_load(
    jobs: list[TraceJob], start_s: float, end_s: float, capacity_gpus: int
) -> float:
    """GPU-seconds asked for by jobs arriving in [start, end), per GPU-second of capacity."""
    work = sum(j.gpu_count * j.duration_s for j in jobs if start_s <= j.submit_s < end_s)
    return work / ((end_s - start_s) * capacity_gpus)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]. 0.0 for an empty list."""
    if not values:
        return 0.0
    ranked = sorted(values)
    return ranked[max(0, math.ceil(q / 100 * len(ranked)) - 1)]


def steady_state_metrics(
    traj: Trajectory,
    jobs: list[TraceJob],
    start_s: float,
    end_s: float,
    policy: str,
    target_load: float,
    seed: int,
) -> SweepRow:
    """Reduce a trajectory to metrics over [start, end) only.

    A tick covers (t - tick_s, t]; it counts if it ends inside (start, end]. Waits are for
    jobs submitted in [start, end), however late they were placed.
    """
    ticks = [k for k in traj.ticks if start_s < k.t_s <= end_s]
    if not ticks:
        raise ValueError(f"no ticks in the window [{start_s}, {end_s}); is tick_s too large?")
    in_window = [j for j in jobs if start_s <= j.submit_s < end_s]
    waits = [traj.placed_at[j.job_id] - j.submit_s for j in in_window if j.job_id in traj.placed_at]
    busy_n = sum(k.busy_gpus for k in ticks)
    allocated = sum(k.allocated_gpus for k in ticks)
    return SweepRow(
        policy=policy,
        target_load=target_load,
        seed=seed,
        offered_load=window_offered_load(jobs, start_s, end_s, traj.capacity_gpus),
        achieved_utilisation=allocated / (len(ticks) * traj.capacity_gpus),
        jobs=len(jobs),
        window_start_s=start_s,
        window_end_s=end_s,
        window_jobs=len(in_window),
        window_placed=len(waits),
        mean_wait_s=statistics.fmean(waits) if waits else 0.0,
        p95_wait_s=percentile(waits, 95),
        peak_temp_c=max(k.peak_temp_c for k in ticks),
        mean_busy_temp_c=sum(k.busy_temp_sum_c for k in ticks) / busy_n if busy_n else 0.0,
        peak_inlet_c=max(k.peak_inlet_c for k in ticks),
        min_headroom_c=min(k.min_headroom_c for k in ticks),
        throttled_gpu_s=sum(k.throttled_gpus for k in ticks) * traj.tick_s,
        capacity_gpus=traj.capacity_gpus,
        sim_end_s=traj.ticks[-1].t_s,
    )


def run_steady_state(
    sim,
    jobs: list[TraceJob],
    policy,
    target_load: float,
    seed: int,
    tick_s: float = 10.0,
    max_sim_s: float = 30 * 86400,
    window: tuple[float, float] = WINDOW,
) -> SweepRow:
    """Run until the window has ended and every job submitted in it has been placed.

    Arrivals after the window keep coming meanwhile, so the window's last jobs queue
    behind the same load they would in a longer trace. Nothing after the window is measured.
    """
    start_s, end_s = steady_state_window(jobs, window)
    need = {j.job_id for j in jobs if start_s <= j.submit_s < end_s}

    def stop(t: float, waiting: int, busy: int, placed_at: dict[str, float]) -> bool:
        return t >= end_s and need <= placed_at.keys()

    traj = simulate(sim, jobs, policy, tick_s, max_sim_s, stop=stop)
    row = steady_state_metrics(traj, jobs, start_s, end_s, policy.name, target_load, seed)
    if row.window_placed < row.window_jobs:
        log.warning(
            "%s load=%.2f seed=%d: %d window jobs still queued at max_sim_s=%.0f",
            policy.name,
            target_load,
            seed,
            row.window_jobs - row.window_placed,
            max_sim_s,
        )
    return row


def summarise(rows: list[SweepRow]) -> list[dict]:
    """Mean and sample std over seeds, per (policy, target load)."""
    groups: dict[tuple[str, float], list[SweepRow]] = defaultdict(list)
    for r in rows:
        groups[r.policy, r.target_load].append(r)
    metrics = [f.name for f in fields(SweepRow) if f.name not in ID_FIELDS]
    out = []
    for (policy, load), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row: dict = {"policy": policy, "target_load": load, "seeds": len(rs)}
        for m in metrics:
            vals = [float(getattr(r, m)) for r in rs]
            row[f"{m}_mean"] = statistics.fmean(vals)
            row[f"{m}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def summary_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}_summary.csv")


def read_rows(path: Path) -> list[SweepRow]:
    types = {f.name: f.type for f in fields(SweepRow)}
    cast = {"str": str, "int": int, "float": float}
    with path.open(newline="") as f:
        return [SweepRow(**{k: cast[types[k]](v) for k, v in r.items()}) for r in csv.DictReader(f)]


def append_row(row: SweepRow, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(SweepRow)])
        if new:
            w.writeheader()
        w.writerow(asdict(row))


def write_summary(summary: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


def main() -> None:
    import grpc

    from gen import isotherm_pb2 as pb
    from gen import isotherm_pb2_grpc as pb_grpc
    from policies import GREEDY_POLICIES
    from simulate import make_policy

    choices = [*GREEDY_POLICIES, "ortools"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sim", default="localhost:50051")
    p.add_argument("--index", type=Path, required=True, help="index.csv from loader.py sweep")
    p.add_argument("--policies", nargs="+", default=choices, choices=choices)
    p.add_argument("--tick-s", type=float, default=10.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="skip rows already in --out")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    index = read_index(args.index)
    try:
        check_rules(index)
    except ValueError as e:
        p.error(str(e))
    if args.out.exists() and not args.resume:
        p.error(f"{args.out} exists; pass --resume to continue it, or remove it")
    done = (
        {(r.policy, r.target_load, r.seed) for r in read_rows(args.out)}
        if args.out.exists()
        else set()
    )

    with grpc.insecure_channel(args.sim) as channel:
        sim = pb_grpc.ThermalSimStub(channel)
        sim_gpus = sum(len(n.gpus) for n in sim.GetState(pb.GetStateRequest()).nodes)
        mismatch = {r.capacity_gpus for r in index} - {sim_gpus}
        if mismatch:
            p.error(f"traces were sized for {sorted(mismatch)} GPUs but the sim has {sim_gpus}")
        for entry in index:
            jobs = read_trace(entry.trace)
            for name in args.policies:
                if (name, entry.target_load, entry.seed) in done:
                    continue
                row = run_steady_state(
                    sim,
                    jobs,
                    make_policy(name, seed=entry.seed),
                    entry.target_load,
                    entry.seed,
                    args.tick_s,
                )
                log.info(
                    "%-13s load=%.2f seed=%d offered=%.3f util=%.3f wait=%.0fs peak=%.1fC",
                    name,
                    entry.target_load,
                    entry.seed,
                    row.offered_load,
                    row.achieved_utilisation,
                    row.mean_wait_s,
                    row.peak_temp_c,
                )
                append_row(row, args.out)

    summary = summarise(read_rows(args.out))
    write_summary(summary, summary_path(args.out))
    print(f"wrote {args.out} and {summary_path(args.out)}")


if __name__ == "__main__":
    main()
