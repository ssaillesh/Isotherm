"""Ground-truth predictor: simulate each candidate placement forward with the physics model.

All candidates are rolled out together, one scenario each (see Cluster.branch), so the cost
is O(horizon) array steps rather than O(candidates x horizon). This is the oracle the
learned surrogate imitates. Recirculation couples every server in the row, so margins are
taken row-wide.
"""

from __future__ import annotations

import numpy as np

from physics import Cluster, Workload
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
        scores: list[CandidateScore | None] = []
        placeable: list[int] = []  # positions in `candidates` that get a rollout
        for name in candidates:
            node = cluster.node_index.get(name)
            if node is None:
                scores.append(CandidateScore(name, feasible=False))
            elif not cluster.can_bind(probe, name):
                feats = features(cluster, probe, node)
                scores.append(CandidateScore(name, feasible=False, features=feats))
            else:
                placeable.append(len(scores))
                scores.append(None)
        if placeable:
            names = [candidates[k] for k in placeable]
            rolled = self._roll_out(cluster, probe, names, horizon_s)
            for k, score in zip(placeable, rolled, strict=True):
                scores[k] = score
        return scores

    def _roll_out(
        self, cluster: Cluster, probe: Workload, names: list[str], horizon_s: float
    ) -> list[CandidateScore]:
        """Step one scenario per name over the horizon; keep each scenario's worst margins."""
        sim = cluster.branch(probe, names)
        peak_gpu = sim.temp_c.max(axis=1)
        peak_inlet = sim.inlet_c.max(axis=1)
        gpu_headroom = sim.gpu_headroom_c()
        throttled = np.zeros(len(names), dtype=bool)
        for _ in range(max(1, int(np.ceil(horizon_s / self.dt_s)))):
            sim.step(self.dt_s)
            peak_gpu = np.maximum(peak_gpu, sim.temp_c.max(axis=1))
            peak_inlet = np.maximum(peak_inlet, sim.inlet_c.max(axis=1))
            gpu_headroom = np.minimum(gpu_headroom, sim.gpu_headroom_c())
            throttled |= sim.throttled.any(axis=1)

        inlet_headroom = sim.inlet_limit_c - peak_inlet
        # Throttling caps power so temperatures sit at the limit; report it as no headroom.
        gpu_headroom = np.where(throttled, 0.0, gpu_headroom)
        return [
            CandidateScore(
                name,
                feasible=bool(inlet_headroom[k] >= 0 and not throttled[k]),
                predicted_peak_c=float(peak_gpu[k]),
                predicted_peak_inlet_c=float(peak_inlet[k]),
                inlet_headroom_c=float(inlet_headroom[k]),
                gpu_headroom_c=float(gpu_headroom[k]),
                features=features(cluster, probe, cluster.node_index[name]),
            )
            for k, name in enumerate(names)
        ]
