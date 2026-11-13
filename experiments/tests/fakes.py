"""A contract-level test double for the ThermalSim service.

Temperatures here are canned numbers chosen per node so tests can assert which node a
policy picks. They are not physics; the real model lives only in sim/.
"""

from __future__ import annotations

from gen import isotherm_pb2 as pb

# node -> (inlet_c reported, headroom_c reported)
NODES = {"n-cool": (25.0, 20.0), "n-mid": (30.0, 10.0), "n-hot": (35.0, 2.0)}
GPUS_PER_NODE = 8
INLET_LIMIT_C, THROTTLE_C = 40.0, 90.0


class FakeSim:
    def __init__(self):
        self.Reset(None)
        self.score_calls = 0
        self.get_state_calls = 0

    def Reset(self, req):
        self.t = 0.0
        self.alloc: dict[str, tuple[str, int, float]] = {}  # id -> (node, gpus, end)
        return self._state()

    def _used(self, node: str) -> int:
        return sum(g for n, g, _ in self.alloc.values() if n == node)

    def _state(self) -> pb.ClusterState:
        nodes = []
        for name, (inlet, _) in NODES.items():
            owners = [wid for wid, (n, g, _) in self.alloc.items() if n == name for _ in range(g)]
            gpus = [
                pb.GpuState(
                    index=i, temp_c=inlet + 30, workload_id=owners[i] if i < len(owners) else ""
                )
                for i in range(GPUS_PER_NODE)
            ]
            nodes.append(
                pb.NodeState(
                    name=name,
                    inlet_c=inlet,
                    free_gpus=GPUS_PER_NODE - self._used(name),
                    gpus=gpus,
                    inlet_limit_c=INLET_LIMIT_C,
                    throttle_c=THROTTLE_C,
                )
            )
        return pb.ClusterState(sim_time_s=self.t, nodes=nodes)

    def GetState(self, req):
        self.get_state_calls += 1
        return self._state()

    def ScoreCandidates(self, req):
        self.score_calls += 1
        return pb.ScoreResponse(
            scores=[
                pb.CandidateScore(node=n, headroom_c=NODES[n][1], feasible=NODES[n][1] > 5)
                for n in req.candidate_nodes
            ],
            predictor="fake",
        )

    def ApplyEvent(self, req):
        w = req.workload
        if req.kind == pb.PodEvent.KIND_BIND:
            if self._used(req.node) + w.gpu_count > GPUS_PER_NODE:
                return pb.ApplyEventResponse(ok=False, error="full")
            self.alloc[w.workload_id] = (req.node, w.gpu_count, self.t + w.duration_s)
        else:
            self.alloc.pop(w.workload_id, None)
        return pb.ApplyEventResponse(ok=True)

    def Step(self, req):
        self.t += req.dt_s
        self.alloc = {k: v for k, v in self.alloc.items() if v[2] > self.t}
        return self._state()
