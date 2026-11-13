"""Run a trace through the simulator under each placement policy (no Kubernetes needed).

Requires a running simulator (`make sim`). Results go to a CSV for plots.py.

    uv run python simulate.py --trace ../data/synthetic.csv --out ../docs/results/baselines.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from gen import isotherm_pb2 as pb
from policies import GREEDY_POLICIES, to_workload
from traces import TraceJob, read_trace

log = logging.getLogger("isotherm.experiments")


@dataclass
class RunMetrics:
    policy: str
    jobs: int
    placed: int
    mean_wait_s: float
    makespan_s: float
    peak_temp_c: float
    mean_busy_temp_c: float
    throttled_gpu_s: float


def _bind(sim, job: TraceJob, node: str) -> bool:
    resp = sim.ApplyEvent(
        pb.PodEvent(kind=pb.PodEvent.KIND_BIND, workload=to_workload(job), node=node)
    )
    if not resp.ok:
        log.warning("bind %s -> %s rejected: %s", job.job_id, node, resp.error)
    return resp.ok


def run_policy(
    sim, jobs: list[TraceJob], policy, tick_s: float = 10.0, max_sim_s: float = 30 * 86400
) -> RunMetrics:
    """Event loop: arrivals join a FIFO queue (with backfill); the sim expires finished jobs."""
    sim.Reset(pb.ResetRequest())
    arrivals = deque(sorted(jobs, key=lambda j: j.submit_s))
    queue: list[TraceJob] = []
    waits: list[float] = []
    peak = busy_sum = throttled_gpu_s = 0.0
    busy_n = 0
    t = 0.0

    while t <= max_sim_s:
        while arrivals and arrivals[0].submit_s <= t:
            queue.append(arrivals.popleft())

        state = sim.GetState(pb.GetStateRequest())
        if hasattr(policy, "assign"):  # batch policy
            placement = policy.assign(state, queue, sim)
            placed = [
                j for j in queue if j.job_id in placement and _bind(sim, j, placement[j.job_id])
            ]
        else:
            placed = []
            for job in queue:
                node = policy.choose(state, job, sim)
                if node and _bind(sim, job, node):
                    placed.append(job)
                    state = sim.GetState(pb.GetStateRequest())
        for job in placed:
            waits.append(t - job.submit_s)
            queue.remove(job)

        state = sim.Step(pb.StepRequest(dt_s=tick_s))
        t += tick_s
        gpus = [g for n in state.nodes for g in n.gpus]
        busy = [g.temp_c for g in gpus if g.workload_id]
        peak = max(peak, max((g.temp_c for g in gpus), default=0.0))
        busy_sum += sum(busy)
        busy_n += len(busy)
        throttled_gpu_s += sum(g.throttled for g in gpus) * tick_s
        if not arrivals and not queue and not busy:
            break

    return RunMetrics(
        policy=policy.name,
        jobs=len(jobs),
        placed=len(waits),
        mean_wait_s=sum(waits) / len(waits) if waits else 0.0,
        makespan_s=t,
        peak_temp_c=peak,
        mean_busy_temp_c=busy_sum / busy_n if busy_n else 0.0,
        throttled_gpu_s=throttled_gpu_s,
    )


def write_metrics(rows: list[RunMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(RunMetrics)])
        w.writeheader()
        w.writerows(asdict(r) for r in rows)


def make_policy(name: str):
    if name == "ortools":
        from ortools_opt import OrToolsBatchPolicy

        return OrToolsBatchPolicy()
    return GREEDY_POLICIES[name]()


def main() -> None:
    import grpc

    from gen import isotherm_pb2_grpc as pb_grpc

    choices = [*GREEDY_POLICIES, "ortools"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sim", default="localhost:50051")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--policies", nargs="+", default=choices, choices=choices)
    p.add_argument("--tick-s", type=float, default=10.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    jobs = read_trace(args.trace, args.limit)
    rows = []
    with grpc.insecure_channel(args.sim) as channel:
        sim = pb_grpc.ThermalSimStub(channel)
        for name in args.policies:
            m = run_policy(sim, jobs, make_policy(name), args.tick_s)
            log.info("%s", m)
            rows.append(m)
    write_metrics(rows, args.out)


if __name__ == "__main__":
    main()
