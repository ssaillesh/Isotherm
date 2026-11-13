"""Load job traces and normalize them to Isotherm's trace CSV.

Normalized trace CSV (the format experiments/ and replayer.py consume):

    job_id,submit_s,duration_s,gpu_count,gpu_util

Cluster capacity (total GPUs) is read from configs/ (--config-dir, default ../configs).

Usage:
    uv run python loader.py synthetic --jobs 500 --out ../data/synthetic.csv
    uv run python loader.py synthetic --jobs 500 --load 0.7 --out ../data/synthetic_70.csv
    uv run python loader.py sweep --loads 0.3 0.5 0.7 0.9 --seeds 5 --jobs 2000 \
        --out ../data/sweep
    uv run python loader.py csv path/to/raw.csv --out ../data/trace.csv --limit 1000
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import yaml

COLUMNS = ("job_id", "submit_s", "duration_s", "gpu_count", "gpu_util")
INDEX_COLUMNS = ("trace", "target_load", "seed", "jobs", "capacity_gpus", "offered_load")
DEFAULT_CONFIG_DIR = Path("../configs")

# Sweep rules: enough jobs per trace for a steady state, enough seeds for a spread.
MIN_SWEEP_JOBS = 2000
MIN_SWEEP_SEEDS = 5


@dataclass(frozen=True)
class Job:
    job_id: str
    submit_s: float
    duration_s: float
    gpu_count: int
    gpu_util: float

    def __post_init__(self) -> None:
        if self.gpu_count <= 0:
            raise ValueError(f"{self.job_id}: gpu_count must be positive")
        if self.duration_s <= 0:
            raise ValueError(f"{self.job_id}: duration_s must be positive")
        if not 0.0 <= self.gpu_util <= 1.0:
            raise ValueError(f"{self.job_id}: gpu_util must be in [0, 1]")


def load_csv(path: Path, limit: int | None = None) -> list[Job]:
    """Read a normalized trace CSV. Rows are sorted by submit time, rebased to t=0."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        jobs = [
            Job(
                job_id=row["job_id"],
                submit_s=float(row["submit_s"]),
                duration_s=float(row["duration_s"]),
                gpu_count=int(row["gpu_count"]),
                gpu_util=float(row["gpu_util"]),
            )
            for row in reader
        ]
    jobs.sort(key=lambda j: j.submit_s)
    jobs = jobs[:limit] if limit else jobs
    t0 = jobs[0].submit_s if jobs else 0.0
    return [Job(j.job_id, j.submit_s - t0, j.duration_s, j.gpu_count, j.gpu_util) for j in jobs]


def capacity_gpus(config_dir: Path = DEFAULT_CONFIG_DIR) -> int:
    """Total GPUs in the row: n_racks * slots_per_rack * gpus_per_node from configs/."""
    layout = yaml.safe_load((config_dir / "rack.yaml").read_text())["layout"]
    node = yaml.safe_load((config_dir / "gpu_specs.yaml").read_text())["node"]
    total = int(layout["n_racks"]) * int(layout["slots_per_rack"]) * int(node["gpus_per_node"])
    if total <= 0:
        raise ValueError(f"{config_dir}: cluster has no GPUs")
    return total


def mean_job_gpu_seconds(
    mean_duration_s: float,
    gpu_choices: tuple[int, ...],
    gpu_weights: tuple[float, ...],
    min_duration_s: float,
) -> float:
    """Expected GPU-seconds of one synthetic job: E[gpu_count] * E[duration].

    Durations are max(min_duration_s, X) with X ~ Exp(mean_duration_s), so
    E[duration] = min + mean * exp(-min / mean). GPU count is independent of duration.
    """
    e_duration = min_duration_s + mean_duration_s * math.exp(-min_duration_s / mean_duration_s)
    e_gpus = sum(c * w for c, w in zip(gpu_choices, gpu_weights, strict=True)) / sum(gpu_weights)
    return e_gpus * e_duration


def offered_load(jobs: list[Job], capacity_gpus: int) -> float:
    """Measured offered load: arrival rate * mean GPU-seconds per job / cluster GPUs.

    This is the average fraction of GPUs the trace asks for while jobs are arriving.
    It ignores the drain at the end, when arrivals stop and running jobs finish.
    """
    if len(jobs) < 2 or jobs[-1].submit_s <= jobs[0].submit_s:
        raise ValueError("need at least two jobs with distinct submit times")
    rate = (len(jobs) - 1) / (jobs[-1].submit_s - jobs[0].submit_s)
    mean_work = sum(j.gpu_count * j.duration_s for j in jobs) / len(jobs)
    return rate * mean_work / capacity_gpus


def synthetic(
    n_jobs: int,
    *,
    target_load: float | None = None,
    capacity_gpus: int | None = None,
    mean_interarrival_s: float = 30.0,
    mean_duration_s: float = 1800.0,
    min_duration_s: float = 60.0,
    gpu_choices: tuple[int, ...] = (1, 2, 4, 8),
    gpu_weights: tuple[float, ...] = (0.4, 0.25, 0.2, 0.15),
    seed: int = 0,
) -> list[Job]:
    """Poisson arrivals, exponential durations, skewed GPU counts. Deterministic per seed.

    With target_load (a fraction of capacity_gpus, e.g. 0.7), the arrival rate is chosen
    so the offered load matches it, and mean_interarrival_s is ignored. By Little's law
    the average number of busy GPUs is rate * E[GPU-seconds per job], so
    mean_interarrival_s = E[GPU-seconds per job] / (target_load * capacity_gpus).
    """
    if target_load is not None:
        if capacity_gpus is None or capacity_gpus <= 0:
            raise ValueError("target_load needs a positive capacity_gpus")
        if target_load <= 0:
            raise ValueError("target_load must be positive")
        work = mean_job_gpu_seconds(mean_duration_s, gpu_choices, gpu_weights, min_duration_s)
        mean_interarrival_s = work / (target_load * capacity_gpus)
    rng = random.Random(seed)
    t, jobs = 0.0, []
    for i in range(n_jobs):
        jobs.append(
            Job(
                job_id=f"job-{i:05d}",
                submit_s=round(t, 3),
                duration_s=round(max(min_duration_s, rng.expovariate(1 / mean_duration_s)), 3),
                gpu_count=rng.choices(gpu_choices, gpu_weights)[0],
                gpu_util=round(rng.uniform(0.5, 1.0), 3),
            )
        )
        t += rng.expovariate(1 / mean_interarrival_s)
    return jobs


def write_csv(jobs: list[Job], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f.name for f in fields(Job)])
        w.writerows(astuple(j) for j in jobs)


def sweep(
    out_dir: Path,
    loads: list[float],
    seeds: list[int],
    n_jobs: int,
    capacity: int,
) -> Path:
    """Write one synthetic trace per (target load, seed) and an index.csv describing them.

    The index is what experiments/sweep.py reads: trace path (relative to the index),
    target load, seed, job count, capacity and the trace's measured offered load.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for load in loads:
        for seed in seeds:
            jobs = synthetic(n_jobs, target_load=load, capacity_gpus=capacity, seed=seed)
            name = f"load{load:.2f}-seed{seed}.csv"
            write_csv(jobs, out_dir / name)
            rows.append(
                (name, load, seed, n_jobs, capacity, round(offered_load(jobs, capacity), 4))
            )
    index = out_dir / "index.csv"
    with index.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(INDEX_COLUMNS)
        w.writerows(rows)
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic", help="generate a synthetic trace")
    s.add_argument("--jobs", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--load", type=float, help="target offered load, a fraction such as 0.7")
    w = sub.add_parser("sweep", help="traces for every (load, seed) plus index.csv")
    w.add_argument("--loads", type=float, nargs="+", required=True)
    w.add_argument("--seeds", type=int, default=MIN_SWEEP_SEEDS, help="seeds 0..N-1")
    w.add_argument("--jobs", type=int, default=MIN_SWEEP_JOBS)
    for sp in (s, w):
        sp.add_argument(
            "--config-dir",
            type=Path,
            default=DEFAULT_CONFIG_DIR,
            help="directory with rack.yaml and gpu_specs.yaml (cluster capacity)",
        )
    c = sub.add_parser("csv", help="validate and normalize an existing trace CSV")
    c.add_argument("path", type=Path)
    c.add_argument("--limit", type=int)
    for sp in (s, c, w):
        sp.add_argument("--out", type=Path, required=True, help="output CSV (sweep: directory)")
    args = p.parse_args()

    if args.cmd == "sweep":
        if args.jobs < MIN_SWEEP_JOBS:
            p.error(f"--jobs must be at least {MIN_SWEEP_JOBS} for a sweep")
        if args.seeds < MIN_SWEEP_SEEDS:
            p.error(f"--seeds must be at least {MIN_SWEEP_SEEDS} for a sweep")
        if any(load <= 0 for load in args.loads):
            p.error("--loads must be positive")
        capacity = capacity_gpus(args.config_dir)
        index = sweep(args.out, args.loads, list(range(args.seeds)), args.jobs, capacity)
        print(
            f"wrote {len(args.loads) * args.seeds} traces of {args.jobs} jobs "
            f"({capacity} GPUs) and {index}"
        )
        return

    capacity = None
    if args.cmd == "synthetic":
        capacity = capacity_gpus(args.config_dir)
        jobs = synthetic(args.jobs, target_load=args.load, capacity_gpus=capacity, seed=args.seed)
    else:
        jobs = load_csv(args.path, args.limit)
    write_csv(jobs, args.out)
    print(f"wrote {len(jobs)} jobs to {args.out}")
    if capacity:
        print(f"offered load {offered_load(jobs, capacity):.3f} of {capacity} GPUs")


if __name__ == "__main__":
    main()
