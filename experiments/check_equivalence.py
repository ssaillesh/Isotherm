"""Prove a speed change leaves results unchanged: the same 12 runs on two versions, diffed.

The 12 runs are all six policies on the load 0.3 and load 0.9 seed 0 sweep traces (from
`make sweep-traces`), each through `sweep.run_steady_state`, the function `sweep.py` calls.
Every column of every row must match exactly. Wall-clock seconds per run are printed side by
side. Runs are sequential (all of A, then all of B) and each is timed once.

Two git refs. Each ref is exported with `git archive` into a temporary directory, and that
ref's own simulator and experiments code run against each other, with metrics off. A ref is
a commit, so uncommitted changes are not included. Put the older version first:

    uv run python check_equivalence.py refs 5d260c5 HEAD

Two simulators already running (start each with `--metrics-port 0`). This checkout's
experiments code drives both, so this mode only checks changes on the sim side:

    uv run python check_equivalence.py sims localhost:50061 localhost:50062

Rows, timings and simulator logs go to --out (default ../data/equivalence/). With --expect,
both sides are also compared with the matching rows of a committed sweep.csv. Exit status 0
means every comparison matched.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
POLICIES = ("random", "binpack", "spread", "coolest-inlet", "isotherm", "ortools")
TRACES = ((0.3, 0, "load0.30-seed0.csv"), (0.9, 0, "load0.90-seed0.csv"))
KEY = ("policy", "target_load", "seed")
SIM_START_TIMEOUT_S = 120.0

Rows = dict[tuple[str, ...], dict[str, str]]
Times = dict[tuple[str, ...], float]


# ------------------------------------------------------------------ the 12 runs


def run_cases(sim_addr: str, traces_dir: Path, rows_path: Path, times_path: Path) -> None:
    """Run the 12 cases against one sim with the experiments code that is on sys.path."""
    import grpc

    from gen import isotherm_pb2_grpc as pb_grpc
    from simulate import make_policy
    from sweep import append_row, run_steady_state
    from traces import read_trace

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.unlink(missing_ok=True)  # append_row appends
    with grpc.insecure_channel(sim_addr) as channel, times_path.open("w", newline="") as f:
        sim = pb_grpc.ThermalSimStub(channel)
        times = csv.writer(f)
        times.writerow([*KEY, "wall_s"])
        for load, seed, name in TRACES:
            jobs = read_trace(traces_dir / name)
            for policy in POLICIES:
                start = time.perf_counter()
                row = run_steady_state(sim, jobs, make_policy(policy, seed=seed), load, seed)
                wall = round(time.perf_counter() - start, 2)  # as stored, so prints agree
                append_row(row, rows_path)
                times.writerow([policy, load, seed, f"{wall:.2f}"])
                f.flush()
                print(f"  {policy:13s} load={load} seed={seed} {wall:7.1f} s", flush=True)


# ------------------------------------------------------------------ comparison


def read_rows(path: Path) -> Rows:
    with path.open(newline="") as f:
        return {tuple(r[k] for k in KEY): r for r in csv.DictReader(f)}


def read_times(path: Path) -> Times:
    with path.open(newline="") as f:
        return {tuple(r[k] for k in KEY): float(r["wall_s"]) for r in csv.DictReader(f)}


def compare_rows(a: Rows, b: Rows, names: tuple[str, str] = ("A", "B")) -> list[str]:
    """Every difference between two sets of rows, one line each; empty means identical.

    Values are compared as the text in the CSV, so a difference in the last digit counts.
    """
    problems = []
    for key in [*a, *(k for k in b if k not in a)]:
        ra, rb = a.get(key), b.get(key)
        if ra is None or rb is None:
            problems.append(f"{key}: only in {names[1] if ra is None else names[0]}")
            continue
        for col in [*ra, *(c for c in rb if c not in ra)]:
            if ra.get(col) != rb.get(col):
                problems.append(
                    f"{key} {col}: {ra.get(col)} ({names[0]}) != {rb.get(col)} ({names[1]})"
                )
    return problems


def speed_rows(a: Times, b: Times) -> list[tuple[str, str, float, float]]:
    """(policy, load, A seconds, B seconds) per run, then a total per load and overall."""
    out = [(k[0], k[1], a[k], b[k]) for k in a if k in b]
    for load in dict.fromkeys(r[1] for r in out):
        here = [r for r in out if r[1] == load]
        out.append(("all", load, sum(r[2] for r in here), sum(r[3] for r in here)))
    runs = [r for r in out if r[0] != "all"]
    out.append(("all", "all", sum(r[2] for r in runs), sum(r[3] for r in runs)))
    return out


def print_speed(a: Times, b: Times) -> None:
    print(f"\n{'policy':13s} {'load':>5s} {'A s':>8s} {'B s':>8s} {'A/B':>6s}")
    for policy, load, ta, tb in speed_rows(a, b):
        ratio = f"{ta / tb:5.1f}x" if tb > 0 else "   n/a"
        print(f"{policy:13s} {load:>5s} {ta:8.1f} {tb:8.1f} {ratio}")


# ------------------------------------------------------------------ running a side


def clean_env() -> dict[str, str]:
    """This process's environment without its venv, so `uv run` elsewhere picks its own."""
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wait_ready(addr: str, proc: subprocess.Popen, log_path: Path) -> None:
    import grpc

    deadline = time.monotonic() + SIM_START_TIMEOUT_S
    with grpc.insecure_channel(addr) as channel:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"simulator exited with status {proc.returncode}: {log_path}")
            try:
                grpc.channel_ready_future(channel).result(timeout=1.0)
                return
            except grpc.FutureTimeoutError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"simulator not ready after {SIM_START_TIMEOUT_S} s"
                    ) from None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def export_ref(sha: str, dest: Path) -> None:
    """The tree at `sha` in dest, with its gRPC stubs generated and this script copied in."""
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", sha], check=True, capture_output=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    # The copy runs from inside the export, so Python imports that tree's experiments code.
    shutil.copy2(Path(__file__).resolve(), dest / "experiments" / Path(__file__).name)
    subprocess.run(
        ["make", "--no-print-directory", "proto-sim", "proto-experiments"],
        cwd=dest,
        check=True,
        env=clean_env(),
    )


def run_ref(sha: str, side: str, traces_dir: Path, out: Path) -> None:
    """Export `sha`, start its simulator, drive it with its own experiments code, stop it."""
    with tempfile.TemporaryDirectory(prefix=f"isotherm-{sha[:7]}-") as tmp:
        tree = Path(tmp)
        export_ref(sha, tree)
        port = free_port()
        log_path = out / f"{side}.sim.log"
        with log_path.open("w") as log:
            sim = subprocess.Popen(
                ["uv", "run", "python", "server.py", "--port", str(port), "--metrics-port", "0"],
                cwd=tree / "sim",
                stdout=log,
                stderr=subprocess.STDOUT,
                env=clean_env(),
                start_new_session=True,  # its own process group, so stopping it stops uv's child
            )
            try:
                wait_ready(f"localhost:{port}", sim, log_path)
                subprocess.run(
                    [
                        "uv", "run", "python", Path(__file__).name, "worker",
                        "--sim", f"localhost:{port}",
                        "--traces-dir", str(traces_dir),
                        "--rows", str(out / f"{side}.rows.csv"),
                        "--times", str(out / f"{side}.times.csv"),
                    ],
                    cwd=tree / "experiments",
                    check=True,
                    env=clean_env(),
                )  # fmt: skip
            finally:
                os.killpg(sim.pid, signal.SIGTERM)
                try:
                    sim.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(sim.pid, signal.SIGKILL)


# ------------------------------------------------------------------ CLI


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode, what in (("refs", "git ref"), ("sims", "simulator address (host:port)")):
        s = sub.add_parser(mode, help=f"compare two of: {what}")
        s.add_argument("a", help=f"first {what} (the older version)")
        s.add_argument("b", help=f"second {what}")
        s.add_argument("--traces-dir", type=Path, default=REPO / "data" / "sweep")
        s.add_argument("--out", type=Path, default=REPO / "data" / "equivalence")
        s.add_argument("--expect", type=Path, help="a sweep.csv both sides must also match")
    w = sub.add_parser("worker", help="internal: the 12 runs against one simulator")
    w.add_argument("--sim", required=True)
    w.add_argument("--traces-dir", type=Path, required=True)
    w.add_argument("--rows", type=Path, required=True)
    w.add_argument("--times", type=Path, required=True)
    args = p.parse_args()

    if args.mode == "worker":
        run_cases(args.sim, args.traces_dir, args.rows, args.times)
        return 0

    traces_dir, out = args.traces_dir.resolve(), args.out.resolve()
    missing = [n for _, _, n in TRACES if not (traces_dir / n).exists()]
    if missing:
        p.error(f"{missing} not in {traces_dir}; run `make sweep-traces` first")
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "refs":
        shas = [git("rev-parse", "--verify", f"{ref}^{{commit}}") for ref in (args.a, args.b)]
        labels = [f"{ref} = {sha}" for ref, sha in zip((args.a, args.b), shas, strict=True)]
        for side, sha, label in zip("ab", shas, labels, strict=True):
            print(f"{side.upper()}: {label}", flush=True)
            run_ref(sha, side, traces_dir, out)
    else:
        labels = [args.a, args.b]
        for side, addr in zip("ab", labels, strict=True):
            print(f"{side.upper()}: simulator at {addr}", flush=True)
            run_cases(addr, traces_dir, out / f"{side}.rows.csv", out / f"{side}.times.csv")

    rows = {side: read_rows(out / f"{side}.rows.csv") for side in "ab"}
    problems = compare_rows(rows["a"], rows["b"])
    if args.expect:
        expected = read_rows(args.expect)
        for side in "ab":
            subset = {k: expected[k] for k in rows[side] if k in expected}
            problems += compare_rows(rows[side], subset, (side.upper(), args.expect.name))

    print_speed(read_times(out / "a.times.csv"), read_times(out / "b.times.csv"))
    print(f"\nA: {labels[0]}\nB: {labels[1]}\nfiles: {out}")
    n = len(rows["a"])
    if problems:
        print(f"\nMISMATCH ({len(problems)} differences):")
        print("\n".join(f"  {line}" for line in problems))
        return 1
    against = f" and {args.expect}" if args.expect else ""
    print(f"\nOK: all {n} rows of A and B match exactly in every column{against}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
