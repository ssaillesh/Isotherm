"""Check that warmer supply air shifts temperatures by exactly the supply change.

The stress scenarios (PLAN.md session 4) pick their supply temperatures from the v1
results by assuming every temperature moves one for one with the supply air. This runs one
trace with one policy against two simulators, each started with its own config directory,
and checks that peak inlet and peak GPU temperature rise by the supply difference.

    # terminal 1 and 2
    cd sim && uv run python server.py --port 50051 --metrics-port 0 --config-dir ../configs
    cd sim && uv run python server.py --port 50052 --metrics-port 0 \\
        --config-dir ../data/configs/supply24.5
    # terminal 3
    uv run python linearity_check.py --sim-a localhost:50051 --config-a ../configs \\
        --sim-b localhost:50052 --config-b ../data/configs/supply24.5 \\
        --trace ../data/sweep/load0.50-seed0.csv --load 0.5 --seed 0 --policy binpack

The supply temperatures are read from the two config directories; each simulator must have
been started with the matching one (the proto does not report it, so this is not checked).
Both rows go to --out. Exit status 0 means both checked columns moved by the shift to
within --tol.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, fields
from pathlib import Path

from sweep import KEY_FIELDS, SweepRow
from variant_config import read_supply

# Columns that must rise by exactly the supply shift (PLAN.md session 4, step 3).
CHECKED = ("peak_inlet_c", "peak_temp_c")
TOLERANCE_C = 0.05


def shift_report(
    a: SweepRow, b: SweepRow, shift_c: float, tol_c: float = TOLERANCE_C
) -> tuple[bool, list[tuple[str, float, float, float]]]:
    """(passed, [(column, a, b, b - a)]) for every numeric column of two runs.

    Passes when every CHECKED column satisfies |(b - a) - shift_c| <= tol_c.
    """
    rows = []
    for f in fields(SweepRow):
        if f.name in KEY_FIELDS:
            continue
        va, vb = getattr(a, f.name), getattr(b, f.name)
        rows.append((f.name, float(va), float(vb), float(vb) - float(va)))
    delta = {name: d for name, _, _, d in rows}
    passed = all(abs(delta[c] - shift_c) <= tol_c for c in CHECKED)
    return passed, rows


def main() -> int:
    import grpc

    from gen import isotherm_pb2_grpc as pb_grpc
    from simulate import make_policy
    from sweep import run_steady_state
    from traces import read_trace

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sim-a", required=True)
    p.add_argument("--config-a", type=Path, required=True)
    p.add_argument("--sim-b", required=True)
    p.add_argument("--config-b", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--load", type=float, required=True, help="target load of --trace")
    p.add_argument("--seed", type=int, required=True, help="seed of --trace")
    p.add_argument("--policy", default="binpack")
    p.add_argument("--tick-s", type=float, default=10.0)
    p.add_argument("--tol", type=float, default=TOLERANCE_C)
    p.add_argument("--out", type=Path, default=Path("../data/stress_check/linearity.csv"))
    args = p.parse_args()

    shift = read_supply(args.config_b) - read_supply(args.config_a)
    jobs = read_trace(args.trace)
    runs = []
    for addr in (args.sim_a, args.sim_b):
        with grpc.insecure_channel(addr) as channel:
            sim = pb_grpc.ThermalSimStub(channel)
            policy = make_policy(args.policy, seed=args.seed)
            runs.append(run_steady_state(sim, jobs, policy, args.load, args.seed, args.tick_s))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", *(x.name for x in fields(SweepRow))])
        w.writeheader()
        for cfg, row in zip((args.config_a, args.config_b), runs, strict=True):
            w.writerow({"config": str(cfg), **asdict(row)})

    passed, rows = shift_report(runs[0], runs[1], shift, args.tol)
    print(f"{args.policy}, {args.trace.name}: supply shift {shift:+.2f} C (b minus a)")
    print(f"{'column':24} {'a':>14} {'b':>14} {'b - a':>12}")
    for name, va, vb, d in rows:
        mark = "  <- checked" if name in CHECKED else ""
        print(f"{name:24} {va:14.4f} {vb:14.4f} {d:+12.6f}{mark}")
    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict}: {', '.join(CHECKED)} within {args.tol} C of {shift:+.2f} C; rows in {args.out}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
