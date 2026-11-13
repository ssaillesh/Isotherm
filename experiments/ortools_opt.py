"""Batch placement with CP-SAT, as an offline upper reference for greedy policies.

Each tick, all queued jobs are assigned jointly: maximize the number placed, then the
total thermal headroom. Headroom per (job, node) comes from the simulator's
ScoreCandidates. Scores are taken independently per job, so interactions between jobs
landing in the same rack during one batch are not captured (a known approximation).
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from gen import isotherm_pb2 as pb
from policies import fitting, to_workload
from traces import TraceJob

PLACE_WEIGHT = 1_000_000  # placing a job always beats any headroom gain


class OrToolsBatchPolicy:
    name = "ortools"

    def __init__(self, time_limit_s: float = 2.0, predictor: str = ""):
        self.time_limit_s = time_limit_s
        self.predictor = predictor

    def assign(self, state: pb.ClusterState, jobs: list[TraceJob], sim) -> dict[str, str]:
        headroom: dict[tuple[int, str], int] = {}
        for j, job in enumerate(jobs):
            nodes = [n.name for n in fitting(state, job)]
            if not nodes:
                continue
            resp = sim.ScoreCandidates(
                pb.ScoreRequest(
                    workload=to_workload(job), candidate_nodes=nodes, predictor=self.predictor
                )
            )
            for s in resp.scores:
                if s.feasible:
                    headroom[j, s.node] = round(s.headroom_c * 100)
        if not headroom:
            return {}

        model = cp_model.CpModel()
        x = {key: model.new_bool_var(f"x_{key[0]}_{key[1]}") for key in headroom}
        for j in range(len(jobs)):
            model.add_at_most_one(v for (jj, _), v in x.items() if jj == j)
        free = {n.name: n.free_gpus for n in state.nodes}
        for node, cap in free.items():
            model.add(sum(jobs[j].gpu_count * v for (j, n), v in x.items() if n == node) <= cap)
        model.maximize(sum((PLACE_WEIGHT + headroom[k]) * v for k, v in x.items()))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {}
        return {jobs[j].job_id: n for (j, n), v in x.items() if solver.value(v)}
