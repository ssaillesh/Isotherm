"""Row-level simulation: GPU allocation, server power, airflow coupling, and time stepping.

Per substep:
    1. GPU demand from utilization; throttle against the current inlet temperature.
    2. Server power P = base_w + sum(GPU power).
    3. Inlet temperatures T_in = T_supply + D @ P   (physics/airflow.py; air is quasi-static).
    4. Exhaust T_out = T_in + P / (airflow * cp).
    5. GPU junction temperatures advance with the RC model (physics/thermal.py).

Steps 1-3 are coupled (throttling depends on inlet, inlet on power). Each substep uses
the previous inlet for throttling, which is stable for substeps << the GPU time constant.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from config import ClusterSpec
from physics.airflow import heat_matrix
from physics.thermal import gpu_power, step_temperature, throttle

MAX_SUBSTEP_S = 5.0


@dataclass(frozen=True)
class Workload:
    workload_id: str
    gpu_count: int
    gpu_util: float
    duration_s: float = 0.0  # 0 = runs until unbound


@dataclass(frozen=True)
class Allocation:
    workload: Workload
    node: int
    gpus: tuple[int, ...]  # global GPU indices
    end_s: float


class PlacementError(ValueError):
    pass


class Cluster:
    def __init__(self, spec: ClusterSpec):
        self.spec = spec
        nodes = spec.nodes
        self.node_names = [n.name for n in nodes]
        self.node_index = {n.name: i for i, n in enumerate(nodes)}
        self.node_rack = np.array([n.rack for n in nodes])
        self.node_slot = np.array([n.slot for n in nodes])
        self.heat = heat_matrix(spec)  # degC at inlet i per W at server j
        self.k_w_per_c = spec.airflow_kg_s * spec.cp_j_kgk

        g = spec.gpus_per_node
        self.gpu_node = np.repeat(np.arange(len(nodes)), g)
        self.tdp_w = np.array([n.gpu.tdp_w for n in nodes for _ in range(g)])
        self.r = np.array([n.gpu.r_c_per_w for n in nodes for _ in range(g)])
        self.c = np.array([n.gpu.c_j_per_c for n in nodes for _ in range(g)])
        self.throttle_c = np.array([n.gpu.throttle_c for n in nodes for _ in range(g)])
        self.reset()

    # ----------------------------------------------------------------- state

    @property
    def n_gpus(self) -> int:
        return len(self.gpu_node)

    @property
    def inlet_limit_c(self) -> float:
        return self.spec.inlet_limit_c

    def reset(self) -> None:
        self.time_s = 0.0
        self.util = np.zeros(self.n_gpus)
        self.owner = np.full(self.n_gpus, "", dtype=object)
        self.allocations: dict[str, Allocation] = {}
        self.inlet_c = np.full(len(self.node_names), self.spec.supply_temp_c)
        # Settle the idle steady state (fixed point of throttle <-> inlet coupling).
        for _ in range(5):
            self._update_power()
        self.temp_c = self.inlet_c[self.gpu_node] + self.power_w * self.r

    def copy(self) -> Cluster:
        """Independent copy for what-if rollouts. Spec and static arrays are shared."""
        new = copy.copy(self)
        for attr in ("util", "owner", "power_w", "throttled", "inlet_c", "temp_c", "node_power_w"):
            setattr(new, attr, getattr(self, attr).copy())
        new.allocations = dict(self.allocations)
        return new

    def node_gpus(self, node: int) -> slice:
        g = self.spec.gpus_per_node
        return slice(node * g, (node + 1) * g)

    def free_gpus(self, node: int) -> np.ndarray:
        s = self.node_gpus(node)
        return np.flatnonzero(self.owner[s] == "") + s.start

    @property
    def exhaust_c(self) -> np.ndarray:
        return self.inlet_c + self.node_power_w / self.k_w_per_c

    def inlet_headroom_c(self) -> float:
        return float(self.inlet_limit_c - self.inlet_c.max())

    def gpu_headroom_c(self) -> float:
        return float((self.throttle_c - self.temp_c).min())

    # ------------------------------------------------------------ placement

    def bind(self, workload: Workload, node_name: str) -> Allocation:
        if workload.workload_id in self.allocations:
            raise PlacementError(f"workload {workload.workload_id!r} is already bound")
        if node_name not in self.node_index:
            raise PlacementError(f"unknown node {node_name!r}")
        if workload.gpu_count <= 0:
            raise PlacementError("gpu_count must be positive")
        node = self.node_index[node_name]
        free = self.free_gpus(node)
        if len(free) < workload.gpu_count:
            raise PlacementError(
                f"node {node_name!r} has {len(free)} free GPUs, need {workload.gpu_count}"
            )
        gpus = free[: workload.gpu_count]
        self.owner[gpus] = workload.workload_id
        self.util[gpus] = np.clip(workload.gpu_util, 0.0, 1.0)
        end = self.time_s + workload.duration_s if workload.duration_s > 0 else math.inf
        alloc = Allocation(workload, node, tuple(int(g) for g in gpus), end)
        self.allocations[workload.workload_id] = alloc
        return alloc

    def unbind(self, workload_id: str) -> bool:
        alloc = self.allocations.pop(workload_id, None)
        if alloc is None:
            return False
        idx = list(alloc.gpus)
        self.owner[idx] = ""
        self.util[idx] = 0.0
        return True

    # ---------------------------------------------------------------- time

    def step(self, dt_s: float) -> list[str]:
        """Advance time by dt_s. Returns ids of workloads that finished during the step."""
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        finished: list[str] = []
        remaining = dt_s
        while remaining > 1e-9:
            h = min(MAX_SUBSTEP_S, remaining)
            self._update_power()
            self.temp_c = step_temperature(
                self.temp_c, self.power_w, self.inlet_c[self.gpu_node], self.r, self.c, h
            )
            self.time_s += h
            remaining -= h
            finished += self._expire()
        return finished

    def _update_power(self) -> None:
        demand = gpu_power(self.util, self.tdp_w, self.spec.idle_gpu_fraction)
        self.power_w, self.throttled = throttle(
            demand, self.inlet_c[self.gpu_node], self.r, self.throttle_c
        )
        self.node_power_w = self.spec.base_w + np.bincount(
            self.gpu_node, weights=self.power_w, minlength=len(self.node_names)
        )
        self.inlet_c = self.spec.supply_temp_c + self.heat @ self.node_power_w

    def _expire(self) -> list[str]:
        done = [wid for wid, a in self.allocations.items() if a.end_s <= self.time_s + 1e-9]
        for wid in done:
            self.unbind(wid)
        return done
