"""Surrogate input features for a (cluster state, workload, candidate node) triple.

The simulator owns this definition. It returns the vectors in CandidateScore.features so
ml/ can train on them without re-deriving anything. Bump FEATURE_SCHEMA whenever
FEATURE_NAMES changes; the learned predictor refuses models trained on another schema.
"""

from __future__ import annotations

import numpy as np

from physics import Cluster, Workload

FEATURE_NAMES = (
    "inlet_c",
    "slot_frac",
    "is_end_rack",
    "node_power_w",
    "row_max_inlet_c",
    "row_inlet_headroom_c",
    "node_max_gpu_c",
    "node_mean_gpu_c",
    "row_gpu_headroom_c",
    "self_heat_c_per_kw",  # D[i, i]: how much this server heats its own inlet
    "max_heat_c_per_kw",  # max_k D[k, i]: worst inlet this server heats
    "node_free_gpus",
    "workload_gpus",
    "workload_util",
    "gpu_tdp_w",
    "gpu_r_c_per_w",
    "gpu_throttle_c",
)
FEATURE_SCHEMA = "v2:" + ",".join(FEATURE_NAMES)


def features(cluster: Cluster, workload: Workload, node: int) -> list[float]:
    spec = cluster.spec
    g = cluster.node_gpus(node)
    first = g.start
    slots = max(spec.slots_per_rack - 1, 1)
    rack = cluster.node_rack[node]
    values = (
        cluster.inlet_c[node],
        cluster.node_slot[node] / slots,
        float(rack in (0, spec.n_racks - 1)),
        cluster.node_power_w[node],
        cluster.inlet_c.max(),
        cluster.inlet_headroom_c(),
        cluster.temp_c[g].max(),
        cluster.temp_c[g].mean(),
        cluster.gpu_headroom_c(),
        cluster.heat[node, node] * 1000,
        cluster.heat[:, node].max() * 1000,
        len(cluster.free_gpus(node)),
        workload.gpu_count,
        workload.gpu_util,
        cluster.tdp_w[first],
        cluster.r[first],
        cluster.throttle_c[first],
    )
    return [float(v) for v in np.asarray(values, dtype=float)]
