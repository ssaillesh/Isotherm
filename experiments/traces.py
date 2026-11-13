"""Read the normalized trace CSV produced by replay/loader.py.

Columns: job_id,submit_s,duration_s,gpu_count,gpu_util. The CSV schema is the interface
between replay/ and experiments/; this module intentionally doesn't import replay/.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceJob:
    job_id: str
    submit_s: float
    duration_s: float
    gpu_count: int
    gpu_util: float


def read_trace(path: Path, limit: int | None = None) -> list[TraceJob]:
    with path.open(newline="") as f:
        jobs = [
            TraceJob(
                r["job_id"],
                float(r["submit_s"]),
                float(r["duration_s"]),
                int(r["gpu_count"]),
                float(r["gpu_util"]),
            )
            for r in csv.DictReader(f)
        ]
    jobs.sort(key=lambda j: j.submit_s)
    return jobs[:limit] if limit else jobs
