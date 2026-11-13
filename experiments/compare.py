"""Paired comparison: one policy against every other, per target load, from a sweep.csv.

    uv run python compare.py ../docs/results/sweep.csv --policy isotherm \\
        --out ../docs/results/compare.csv

Rows are paired by seed: both policies ran the same trace (same target load and seed), so
trace-to-trace variation cancels. For every metric, d = chosen - other per seed, reported as
the mean and sample standard deviation of d over the seeds, and two counts:

- lower: seeds where the chosen policy's value was strictly lower;
- better: seeds where it was strictly better, using BETTER below (blank for metrics with
  no agreed direction, such as achieved utilisation).

A mean gap no larger than its standard deviation is labelled "no difference" (this includes
the case where every d is 0). Each scenario (supply temperature) has its own results
directory and sweep.csv, so run this once per directory.

Reads the CSV by column name, so sweeps written before a metric existed still compare on
the metrics they have.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from sweep import MIN_SEEDS  # the reporting rule, shared with the sweep

KEY = ("policy", "target_load", "seed")
# Columns that describe the trace or its window rather than the run: equal for every policy.
NOT_COMPARED = {
    "jobs",
    "capacity_gpus",
    "offered_load",
    "window_start_s",
    "window_end_s",
    "window_jobs",
}
# Direction of "better": -1 = lower is better, +1 = higher is better.
BETTER = {
    "mean_wait_s": -1,
    "p95_wait_s": -1,
    "peak_temp_c": -1,
    "mean_busy_temp_c": -1,
    "peak_inlet_c": -1,
    "min_headroom_c": +1,
    "throttled_gpu_s": -1,
    "inlet_exceedance_cs": -1,
    "time_over_inlet_s": -1,
    "gpu_exceedance_cs": -1,
}


@dataclass
class Paired:
    target_load: float
    policy: str
    other: str
    metric: str
    n: int
    mean_diff: float  # mean over seeds of (policy - other)
    std_diff: float  # sample standard deviation of the same
    lower: int  # seeds where policy < other
    better: int | None  # seeds where policy beat other; None without a direction
    verdict: str  # "difference" or "no difference"


def read_sweep(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Rows and the metric columns to compare, in file order."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    missing = [k for k in KEY if k not in header]
    if missing:
        raise ValueError(f"{path} has no {missing} column")
    metrics = [c for c in header if c not in KEY and c not in NOT_COMPARED]
    return rows, metrics


def compare(rows: list[dict[str, str]], metrics: list[str], policy: str) -> list[Paired]:
    """Per (target load, other policy, metric): the paired difference, policy - other."""
    by: dict[tuple[float, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        load, seed = float(r["target_load"]), int(r["seed"])
        if seed in by[load, r["policy"]]:
            raise ValueError(f"two rows for {r['policy']} load={load} seed={seed}")
        by[load, r["policy"]][seed] = r
    others = list(dict.fromkeys(r["policy"] for r in rows if r["policy"] != policy))
    loads = sorted({load for load, _ in by})
    if not any((load, policy) in by for load in loads):
        raise ValueError(f"no rows for policy {policy!r}")

    out = []
    for load in loads:
        mine = by.get((load, policy), {})
        for other in others:
            theirs = by.get((load, other), {})
            if not theirs:
                continue
            if set(mine) != set(theirs):
                raise ValueError(
                    f"load={load}: {policy} has seeds {sorted(mine)}, {other} has {sorted(theirs)}"
                )
            seeds = sorted(mine)
            if len(seeds) < 2:
                raise ValueError(f"load={load}: need at least 2 seeds for a spread")
            for m in metrics:
                d = [float(mine[s][m]) - float(theirs[s][m]) for s in seeds]
                mean, std = statistics.fmean(d), statistics.stdev(d)
                direction = BETTER.get(m)
                out.append(
                    Paired(
                        target_load=load,
                        policy=policy,
                        other=other,
                        metric=m,
                        n=len(d),
                        mean_diff=mean,
                        std_diff=std,
                        lower=sum(x < 0 for x in d),
                        better=None if direction is None else sum(x * direction > 0 for x in d),
                        verdict="no difference" if abs(mean) <= std else "difference",
                    )
                )
    return out


def write_compare(results: list[Paired], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(Paired)])
        w.writeheader()
        for p in results:
            row = asdict(p)
            row["better"] = "" if p.better is None else p.better
            w.writerow(row)


def _decimals(metric: str) -> int:
    if metric == "achieved_utilisation":
        return 4
    return 1 if metric.endswith(("_s", "_cs")) or metric == "window_placed" else 2


def format_table(results: list[Paired], metric: str) -> str:
    """One metric: a row per other policy, a column per load."""
    here = [p for p in results if p.metric == metric]
    if not here:
        return ""
    direction = BETTER.get(metric)
    count = {None: "lower", -1: "better (lower)", 1: "better (higher)"}[direction]
    loads = sorted({p.target_load for p in here})
    others = list(dict.fromkeys(p.other for p in here))
    cell = {(p.other, p.target_load): p for p in here}
    k = _decimals(metric)

    def fmt(p: Paired | None) -> str:
        if p is None:
            return "-"
        wins = p.lower if p.better is None else p.better
        tag = "  no diff" if p.verdict == "no difference" else ""
        j = k
        while j < k + 3 and f"{abs(p.mean_diff):.{j}f}" == f"{p.std_diff:.{j}f}" and tag == "":
            j += 1  # a real gap that rounds to mean = std: show the digit that separates them
        return f"{p.mean_diff:+.{j}f} ± {p.std_diff:.{j}f} ({wins}/{p.n}){tag}"

    lines = [f"{metric}: {here[0].policy} minus other; count = seeds {count}"]
    grid = [[o, *(fmt(cell.get((o, load))) for load in loads)] for o in others]
    head = ["vs", *(f"load {load:g}" for load in loads)]
    widths = [max(len(r[i]) for r in [head, *grid]) for i in range(len(head))]
    for r in [head, *grid]:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("sweep", type=Path, help="sweep.csv from sweep.py")
    p.add_argument("--policy", default="isotherm", help="the policy compared with every other")
    p.add_argument("--out", type=Path, help="default: compare.csv next to the sweep")
    args = p.parse_args()

    try:
        rows, metrics = read_sweep(args.sweep)
        results = compare(rows, metrics, args.policy)
    except ValueError as e:
        p.error(str(e))
    few = sorted({(r.target_load, r.n) for r in results if r.n < MIN_SEEDS})
    if few:
        p.error(f"(load, seeds) below the {MIN_SEEDS}-seed reporting rule: {few}")

    out = args.out or args.sweep.with_name("compare.csv")
    write_compare(results, out)
    print("Simulated results. Paired by seed; mean ± sample std of the per-seed difference.")
    print('"no diff": the mean gap is no larger than its standard deviation.\n')
    for m in metrics:
        print(format_table(results, m), end="\n\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
