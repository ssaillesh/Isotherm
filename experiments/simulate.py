"""Run a trace through the simulator under each placement policy (no Kubernetes needed).

Requires a running simulator (`make sim`). Results go to a CSV for plots.py.

    uv run python simulate.py --trace ../data/synthetic.csv --out ../docs/results/baselines.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
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
    inlet_exceedance_cs: float
    time_over_inlet_s: float
    gpu_exceedance_cs: float


@dataclass(frozen=True)
class Tick:
    """One tick, the interval (t_s - tick_s, t_s].

    allocated_gpus is what was bound while the tick ran (after that tick's placements).
    Everything else is what the sim reported at the end of the tick. Totals over time
    count each tick's end-of-tick value for the whole tick (value x tick_s).
    """

    t_s: float
    allocated_gpus: int
    busy_gpus: int
    peak_temp_c: float
    busy_temp_sum_c: float
    peak_inlet_c: float
    # Smallest margin in the row: min(inlet_limit_c - inlet_c, throttle_c - temp_c).
    min_headroom_c: float
    throttled_gpus: int
    # How far limits are broken, in °C summed over everything above its limit:
    # sum over servers of max(0, inlet_c - inlet_limit_c), and over GPUs of
    # max(0, temp_c - throttle_c). Limits are the sim's, which it reads from configs/.
    inlet_excess_c: float
    inlets_over: int  # servers whose inlet is strictly above inlet_limit_c
    gpu_excess_c: float


@dataclass
class Trajectory:
    tick_s: float
    capacity_gpus: int
    ticks: list[Tick] = field(default_factory=list)
    placed_at: dict[str, float] = field(default_factory=dict)  # job_id -> bind time


# (sim time, jobs not yet placed, busy GPUs, placements so far) -> stop?
StopRule = Callable[[float, int, int, dict[str, float]], bool]


def drained(t: float, waiting: int, busy: int, placed_at: dict[str, float]) -> bool:
    return waiting == 0 and busy == 0


def _tick(t: float, allocated: int, state: pb.ClusterState) -> Tick:
    gpus = [g for n in state.nodes for g in n.gpus]
    busy = [g.temp_c for g in gpus if g.workload_id]
    headroom = [n.inlet_limit_c - n.inlet_c for n in state.nodes]
    headroom += [n.throttle_c - g.temp_c for n in state.nodes for g in n.gpus]
    inlet_excess = [max(0.0, n.inlet_c - n.inlet_limit_c) for n in state.nodes]
    gpu_excess = [max(0.0, g.temp_c - n.throttle_c) for n in state.nodes for g in n.gpus]
    return Tick(
        t_s=t,
        allocated_gpus=allocated,
        busy_gpus=len(busy),
        peak_temp_c=max((g.temp_c for g in gpus), default=0.0),
        busy_temp_sum_c=sum(busy),
        peak_inlet_c=max((n.inlet_c for n in state.nodes), default=0.0),
        min_headroom_c=min(headroom, default=0.0),
        throttled_gpus=sum(g.throttled for g in gpus),
        inlet_excess_c=sum(inlet_excess),
        inlets_over=sum(e > 0.0 for e in inlet_excess),
        gpu_excess_c=sum(gpu_excess),
    )


def _bind(sim, job: TraceJob, node: str) -> bool:
    resp = sim.ApplyEvent(
        pb.PodEvent(kind=pb.PodEvent.KIND_BIND, workload=to_workload(job), node=node)
    )
    if not resp.ok:
        log.warning("bind %s -> %s rejected: %s", job.job_id, node, resp.error)
    return resp.ok


def simulate(
    sim,
    jobs: list[TraceJob],
    policy,
    tick_s: float = 10.0,
    max_sim_s: float = 30 * 86400,
    stop: StopRule = drained,
) -> Trajectory:
    """Event loop: arrivals join a FIFO queue (with backfill); the sim expires finished jobs.

    Records what the sim reports after every tick, and when each job was bound.

    This loop is the sim's only client, so the state that Reset or Step returned is still
    current at the top of the next tick, and asking again with GetState would return the
    same thing. GetState is only needed after a bind.
    """
    state = sim.Reset(pb.ResetRequest())
    traj = Trajectory(tick_s, capacity_gpus=sum(len(n.gpus) for n in state.nodes))
    arrivals = deque(sorted(jobs, key=lambda j: j.submit_s))
    queue: list[TraceJob] = []
    t = 0.0

    while t <= max_sim_s:
        while arrivals and arrivals[0].submit_s <= t:
            queue.append(arrivals.popleft())

        free_before = sum(n.free_gpus for n in state.nodes)
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
            traj.placed_at[job.job_id] = t
            queue.remove(job)
        # Sampling busy GPUs after Step would miss each job's last tick (it has expired).
        allocated = traj.capacity_gpus - free_before + sum(j.gpu_count for j in placed)

        state = sim.Step(pb.StepRequest(dt_s=tick_s))
        t += tick_s
        tick = _tick(t, allocated, state)
        traj.ticks.append(tick)
        if stop(t, len(arrivals) + len(queue), tick.busy_gpus, traj.placed_at):
            break
    return traj


def run_policy(
    sim, jobs: list[TraceJob], policy, tick_s: float = 10.0, max_sim_s: float = 30 * 86400
) -> RunMetrics:
    """Whole-run metrics, from t = 0 until every job has arrived, run and finished."""
    traj = simulate(sim, jobs, policy, tick_s, max_sim_s)
    submit = {j.job_id: j.submit_s for j in jobs}
    waits = [t - submit[jid] for jid, t in traj.placed_at.items()]
    busy_n = sum(k.busy_gpus for k in traj.ticks)
    exceed = limit_metrics(traj.ticks, tick_s)
    return RunMetrics(
        policy=policy.name,
        jobs=len(jobs),
        placed=len(waits),
        mean_wait_s=sum(waits) / len(waits) if waits else 0.0,
        makespan_s=traj.ticks[-1].t_s if traj.ticks else 0.0,
        peak_temp_c=max((k.peak_temp_c for k in traj.ticks), default=0.0),
        mean_busy_temp_c=sum(k.busy_temp_sum_c for k in traj.ticks) / busy_n if busy_n else 0.0,
        throttled_gpu_s=sum(k.throttled_gpus for k in traj.ticks) * tick_s,
        **exceed,
    )


def limit_metrics(ticks: list[Tick], tick_s: float) -> dict[str, float]:
    """How far and how long limits were broken over these ticks, in °C·s and s."""
    return {
        "inlet_exceedance_cs": sum(k.inlet_excess_c for k in ticks) * tick_s,
        "time_over_inlet_s": sum(k.inlets_over > 0 for k in ticks) * tick_s,
        "gpu_exceedance_cs": sum(k.gpu_excess_c for k in ticks) * tick_s,
    }


def write_metrics(rows: list[RunMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(RunMetrics)])
        w.writeheader()
        w.writerows(asdict(r) for r in rows)


def make_policy(name: str, seed: int = 0):
    """Build a policy by name; `seed` only matters for the random policy."""
    if name == "ortools":
        from ortools_opt import OrToolsBatchPolicy

        return OrToolsBatchPolicy()
    if name == "random":
        return GREEDY_POLICIES[name](seed=seed)
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
