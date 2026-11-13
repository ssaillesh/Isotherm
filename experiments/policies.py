"""Placement policies compared in the experiments.

Every thermal quantity a policy uses comes from the simulator (ClusterState or
ScoreCandidates). Policies rank and pick; they never compute temperatures.
"""

from __future__ import annotations

import random
from typing import Protocol

from gen import isotherm_pb2 as pb
from traces import TraceJob


def to_workload(job: TraceJob) -> pb.Workload:
    return pb.Workload(
        workload_id=job.job_id,
        gpu_count=job.gpu_count,
        gpu_util=job.gpu_util,
        duration_s=job.duration_s,
    )


def fitting(state: pb.ClusterState, job: TraceJob) -> list[pb.NodeState]:
    return [n for n in state.nodes if n.free_gpus >= job.gpu_count]


class Policy(Protocol):
    name: str

    def choose(self, state: pb.ClusterState, job: TraceJob, sim) -> str | None: ...


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, state, job, sim):
        nodes = fitting(state, job)
        return self.rng.choice(nodes).name if nodes else None


class BinPackPolicy:
    """Most-allocated first: consolidate onto as few nodes as possible."""

    name = "binpack"

    def choose(self, state, job, sim):
        nodes = fitting(state, job)
        return min(nodes, key=lambda n: (n.free_gpus, n.name)).name if nodes else None


class SpreadPolicy:
    """Least-allocated first, like kube-scheduler's default NodeResourcesFit scoring."""

    name = "spread"

    def choose(self, state, job, sim):
        nodes = fitting(state, job)
        return max(nodes, key=lambda n: (n.free_gpus, n.name)).name if nodes else None


class CoolestInletPolicy:
    """Pick the node whose inlet air the simulator reports as coolest right now."""

    name = "coolest-inlet"

    def choose(self, state, job, sim):
        nodes = fitting(state, job)
        return min(nodes, key=lambda n: (n.inlet_c, n.name)).name if nodes else None


class IsothermPolicy:
    """What the Go plugin does: ask the sim to score candidates, take the most headroom."""

    name = "isotherm"

    def __init__(self, predictor: str = "", horizon_s: float = 0.0):
        self.predictor = predictor
        self.horizon_s = horizon_s

    def choose(self, state, job, sim):
        nodes = fitting(state, job)
        if not nodes:
            return None
        resp = sim.ScoreCandidates(
            pb.ScoreRequest(
                workload=to_workload(job),
                candidate_nodes=[n.name for n in nodes],
                horizon_s=self.horizon_s,
                predictor=self.predictor,
            )
        )
        feasible = [s for s in resp.scores if s.feasible] or list(resp.scores)
        return max(feasible, key=lambda s: (s.headroom_c, s.node)).node if feasible else None


GREEDY_POLICIES = {
    p.name: p
    for p in (RandomPolicy, BinPackPolicy, SpreadPolicy, CoolestInletPolicy, IsothermPolicy)
}
