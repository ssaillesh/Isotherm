"""Ground-truth predictor: simulate each candidate placement forward with the physics model.

Accurate but O(candidates x horizon). This is the oracle the learned surrogate imitates.
Recirculation couples every server in the row, so margins are taken row-wide.
"""

from __future__ import annotations

import numpy as np

from physics import Cluster, PlacementError, Workload
from surrogate.base import CandidateScore
from surrogate.features import features


class RolloutPredictor:
    name = "rollout"

    def __init__(self, dt_s: float = 10.0):
        self.dt_s = dt_s

    def score(
        self, cluster: Cluster, workload: Workload, candidates: list[str], horizon_s: float
    ) -> list[CandidateScore]:
        # The probe workload stays for the whole horizon regardless of its real duration.
        probe = Workload(workload.workload_id, workload.gpu_count, workload.gpu_util, 0.0)
        return [self._score_one(cluster, probe, name, horizon_s) for name in candidates]

    def _score_one(
        self, cluster: Cluster, workload: Workload, name: str, horizon_s: float
    ) -> CandidateScore:
        node = cluster.node_index.get(name)
        if node is None:
            return CandidateScore(name, feasible=False)
        feats = features(cluster, workload, node)

        sim = cluster.copy()
        try:
            sim.bind(workload, name)
        except PlacementError:
            return CandidateScore(name, feasible=False, features=feats)

        peak_gpu, peak_inlet = sim.temp_c.max(), sim.inlet_c.max()
        gpu_headroom = sim.gpu_headroom_c()
        throttled = False
        for _ in range(max(1, int(np.ceil(horizon_s / self.dt_s)))):
            sim.step(self.dt_s)
            peak_gpu = max(peak_gpu, sim.temp_c.max())
            peak_inlet = max(peak_inlet, sim.inlet_c.max())
            gpu_headroom = min(gpu_headroom, sim.gpu_headroom_c())
            throttled = throttled or bool(sim.throttled.any())

        inlet_headroom = sim.inlet_limit_c - peak_inlet
        # Throttling caps power so temperatures sit at the limit; report it as no headroom.
        gpu_headroom = 0.0 if throttled else float(gpu_headroom)
        return CandidateScore(
            name,
            feasible=inlet_headroom >= 0 and not throttled,
            predicted_peak_c=float(peak_gpu),
            predicted_peak_inlet_c=float(peak_inlet),
            inlet_headroom_c=float(inlet_headroom),
            gpu_headroom_c=gpu_headroom,
            features=feats,
        )
