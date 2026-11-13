"""Load job traces and normalize them to Isotherm's trace CSV.

Normalized trace CSV (the format experiments/ and replayer.py consume):

    job_id,submit_s,duration_s,gpu_count,gpu_util

Usage:
    uv run python loader.py synthetic --jobs 500 --out ../data/synthetic.csv
    uv run python loader.py csv path/to/raw.csv --out ../data/trace.csv --limit 1000
"""

from __future__ import annotations

import argparse
import csv
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


def synthetic(
    n_jobs: int,
    *,
    mean_interarrival_s: float = 30.0,
    mean_duration_s: float = 1800.0,
    gpu_choices: tuple[int, ...] = (1, 2, 4, 8),
    gpu_weights: tuple[float, ...] = (0.4, 0.25, 0.2, 0.15),
    seed: int = 0,
) -> list[Job]:
    """Poisson arrivals, exponential durations, skewed GPU counts. Deterministic per seed."""
    rng = random.Random(seed)
    t, jobs = 0.0, []
    for i in range(n_jobs):
        jobs.append(
            Job(
                job_id=f"job-{i:05d}",
                submit_s=round(t, 3),
                duration_s=round(max(60.0, rng.expovariate(1 / mean_duration_s)), 3),
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
    c = sub.add_parser("csv", help="validate and normalize an existing trace CSV")
    c.add_argument("path", type=Path)
    c.add_argument("--limit", type=int)
    for sp in (s, c):
        sp.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    jobs = (
        synthetic(args.jobs, seed=args.seed)
        if args.cmd == "synthetic"
        else load_csv(args.path, args.limit)
    )
    write_csv(jobs, args.out)
    print(f"wrote {len(jobs)} jobs to {args.out}")


if __name__ == "__main__":
    main()
