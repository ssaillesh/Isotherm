"""Per-GPU power and junction temperature, vectorized over every GPU in the row.

Junction model:  C * dT/dt = P - (T - T_inlet) / R

For inputs held constant over a step the exact solution is

    T(t + dt) = T_ss + (T(t) - T_ss) * exp(-dt / (R * C)),   T_ss = T_inlet + P * R

which is unconditionally stable, so callers can take large steps.

Every function works element by element, so arrays may carry extra leading axes (for
example one row per what-if scenario).
"""

from __future__ import annotations

import numpy as np


def gpu_power(util: np.ndarray, tdp_w: np.ndarray, idle_fraction: float) -> np.ndarray:
    """Idle draw (idle_fraction * TDP) plus utilization-proportional dynamic power."""
    idle = idle_fraction * tdp_w
    return idle + np.clip(util, 0.0, 1.0) * (tdp_w - idle)


def throttle(
    demand_w: np.ndarray,
    inlet_c: np.ndarray,
    r_c_per_w: np.ndarray,
    throttle_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap power so the steady-state temperature never exceeds the throttle point.

    Returns (power_w, throttled_mask).
    """
    cap_w = np.maximum((throttle_c - inlet_c) / r_c_per_w, 0.0)
    throttled = demand_w > cap_w
    return np.where(throttled, cap_w, demand_w), throttled


def steady_state(inlet_c: np.ndarray, power_w: np.ndarray, r_c_per_w: np.ndarray) -> np.ndarray:
    """T_ss: the temperature a junction settles at if power and inlet stay as they are."""
    return inlet_c + power_w * r_c_per_w


def decay(r_c_per_w: np.ndarray, c_j_per_c: np.ndarray, dt_s: float) -> np.ndarray:
    """exp(-dt / (R * C)): the share of the gap to T_ss that is left after dt_s.

    It depends only on the GPU and the step length, so callers can compute it once.
    """
    return np.exp(-dt_s / (r_c_per_w * c_j_per_c))


def relax(temp_c: np.ndarray, steady_c: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """Move each junction toward its steady state: the exact solution over one step."""
    return steady_c + (temp_c - steady_c) * decay


def step_temperature(
    temp_c: np.ndarray,
    power_w: np.ndarray,
    inlet_c: np.ndarray,
    r_c_per_w: np.ndarray,
    c_j_per_c: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """One exact step from scratch. Cluster.step uses the three pieces above directly."""
    steady = steady_state(inlet_c, power_w, r_c_per_w)
    return relax(temp_c, steady, decay(r_c_per_w, c_j_per_c, dt_s))
