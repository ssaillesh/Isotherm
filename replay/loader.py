"""Load job traces and normalize them to Isotherm's trace CSV.

Normalized trace CSV (the format experiments/ and replayer.py consume):

    job_id,submit_s,duration_s,gpu_count,gpu_util

Usage:
    uv run python loader.py synthetic --jobs 500 --out ../data/synthetic.csv
    uv run python loader.py synthetic --jobs 500 --load 0.7 --capacity-gpus 320 \
        --out ../data/synthetic_70.csv
    uv run python loader.py csv path/to/raw.csv --out ../data/trace.csv --limit 1000
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import astuple, dataclass, fields
from pathlib import Path

COLUMNS = ("job_id", "submit_s", "duration_s", "gpu_count", "gpu_util")


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic", help="generate a synthetic trace")
    s.add_argument("--jobs", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--load", type=float, help="target offered load, a fraction such as 0.7")
    s.add_argument(
        "--capacity-gpus",
        type=int,
        help="total GPUs in the cluster (n_racks * slots_per_rack * gpus_per_node in configs/)",
    )
    c = sub.add_parser("csv", help="validate and normalize an existing trace CSV")
    c.add_argument("path", type=Path)
    c.add_argument("--limit", type=int)
    for sp in (s, c):
        sp.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.cmd == "synthetic":
        if args.load is not None and args.capacity_gpus is None:
            p.error("--load needs --capacity-gpus")
        jobs = synthetic(
            args.jobs, target_load=args.load, capacity_gpus=args.capacity_gpus, seed=args.seed
        )
    else:
        jobs = load_csv(args.path, args.limit)
    write_csv(jobs, args.out)
    print(f"wrote {len(jobs)} jobs to {args.out}")
    if args.cmd == "synthetic" and args.capacity_gpus:
        load = offered_load(jobs, args.capacity_gpus)
        print(f"offered load {load:.3f} of {args.capacity_gpus} GPUs")


if __name__ == "__main__":
    main()
