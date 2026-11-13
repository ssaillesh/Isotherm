"""Fast predictor backed by a regressor trained in ml/ on rollout labels.

The model file is a joblib dict: {"model": <regressor>, "feature_schema": str, "targets":
["inlet_headroom_c", "gpu_headroom_c"]}. model.predict(X) returns one column per target.
Requires the `learned` extra (scikit-learn, joblib).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physics import Cluster, Workload
from surrogate.base import CandidateScore
from surrogate.features import FEATURE_SCHEMA, features

TARGETS = ["inlet_headroom_c", "gpu_headroom_c"]
# Predicted headroom below this margin is treated as a risk of crossing a limit.
FEASIBLE_MARGIN_C = 0.5


class LearnedPredictor:
    name = "learned"

    def __init__(self, model_path: Path):
        import joblib  # optional dependency

        bundle = joblib.load(model_path)
        if bundle.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError(
                f"model at {model_path} was trained on feature schema "
                f"{bundle.get('feature_schema')!r}, simulator uses {FEATURE_SCHEMA!r}"
            )
        if bundle.get("targets") != TARGETS:
            raise ValueError(f"model targets {bundle.get('targets')!r}, expected {TARGETS!r}")
        self.model = bundle["model"]

    def score(
        self, cluster: Cluster, workload: Workload, candidates: list[str], horizon_s: float
    ) -> list[CandidateScore]:
        out = {n: CandidateScore(n, feasible=False) for n in candidates}
        known = [(n, cluster.node_index[n]) for n in candidates if n in cluster.node_index]
        if not known:
            return list(out.values())
        x = np.array([features(cluster, workload, i) for _, i in known])
        pred = np.asarray(self.model.predict(x)).reshape(len(known), len(TARGETS))
        for (name, i), (inlet_h, gpu_h), f in zip(known, pred, x, strict=True):
            fits = len(cluster.free_gpus(i)) >= workload.gpu_count
            out[name] = CandidateScore(
                name,
                feasible=fits and min(inlet_h, gpu_h) > FEASIBLE_MARGIN_C,
                predicted_peak_c=float(cluster.throttle_c.min() - gpu_h),
                predicted_peak_inlet_c=float(cluster.inlet_limit_c - inlet_h),
                inlet_headroom_c=float(inlet_h),
                gpu_headroom_c=float(gpu_h),
                features=f.tolist(),
            )
        return [out[n] for n in candidates]
