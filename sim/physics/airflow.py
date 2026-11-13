"""Row airflow: where each server's exhaust goes, and the resulting inlet temperatures.

Model (docs/design.md, "Recirculation"). For servers i, j with rack r and slot s:

    h_i  = s_i / (S - 1)                         height fraction
    e_i  = 1 if r_i is the first or last rack    end-of-row flag
    d_ij = sqrt((r_i - r_j)^2 + (s_i - s_j)^2)   distance, 1 rack pitch = 1 slot
    w_j  = 1 + height_bias * h_j + end_bias * e_j
    G_ij = w_j * exp(-d_ij / length_scale), each row normalised to sum to 1
    f_i  = max_fraction * w_i / max(w)
    A_ij = f_i * G_ij                            share of i's exhaust entering j's inlet

Energy balance at every inlet, with K = diag(airflow * cp):

    D    = (K - A^T K)^-1 A^T                    degC per W
    T_in = T_supply + D @ P

Valid only if every inlet draws less recirculated air than its total flow (column sums
of A below 1 for equal airflow); `recirculation_matrix` asserts that.
"""

from __future__ import annotations

import numpy as np

from config import ClusterSpec


def recirculation_matrix(spec: ClusterSpec) -> np.ndarray:
    rc = spec.recirculation
    r = np.array([n.rack for n in spec.nodes], dtype=float)
    s = np.array([n.slot for n in spec.nodes], dtype=float)

    h = s / (spec.slots_per_rack - 1) if spec.slots_per_rack > 1 else np.zeros_like(s)
    e = ((r == 0) | (r == spec.n_racks - 1)).astype(float)
    d = np.hypot(r[:, None] - r[None, :], s[:, None] - s[None, :])
    w = 1.0 + rc.height_bias * h + rc.end_bias * e

    g = w[None, :] * np.exp(-d / rc.length_scale)
    g /= g.sum(axis=1, keepdims=True)
    f = rc.max_fraction * w / w.max()
    a = f[:, None] * g

    k = np.full(len(spec.nodes), spec.airflow_kg_s * spec.cp_j_kgk)
    # Recirculated heat-capacity flow into each inlet must stay below that inlet's own flow.
    intake = (a * k[:, None]).sum(axis=0)
    if not np.all(intake < k):
        worst = int(np.argmax(intake / k))
        raise ValueError(
            f"recirculation infeasible: inlet of {spec.nodes[worst].name} would take "
            f"{intake[worst] / k[worst]:.2f}x its own airflow; lower max_fraction"
        )
    return a


def heat_matrix(spec: ClusterSpec, a: np.ndarray | None = None) -> np.ndarray:
    """D such that T_in = T_supply + D @ P (P in W per server)."""
    a = recirculation_matrix(spec) if a is None else a
    k = np.diag(np.full(len(spec.nodes), spec.airflow_kg_s * spec.cp_j_kgk))
    return np.linalg.solve(k - a.T @ k, a.T)
