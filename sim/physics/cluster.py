"""Row-level simulation: GPU allocation, server power, airflow coupling, and time stepping.

Per substep:
    1. GPU demand from utilization; throttle against the current inlet temperature.
    2. Server power P = base_w + sum(GPU power).
    3. Inlet temperatures T_in = T_supply + D @ P   (physics/airflow.py; air is quasi-static).
    4. Exhaust T_out = T_in + P / (airflow * cp).
    5. GPU junction temperatures advance with the RC model (physics/thermal.py).

Steps 1-3 are coupled (throttling depends on inlet, inlet on power). Each substep uses
the previous inlet for throttling, which is stable for substeps << the GPU time constant.

Scenarios. Every state array has a leading scenario axis: (scenarios, GPUs) or
(scenarios, nodes). The live row is one scenario. A what-if rollout calls `branch` to get
one scenario per candidate node and steps them all at once; `step` is the same code for
one scenario or forty. Scenarios share the clock and the workload schedule (who is bound
where, until when) and differ only in the trial workload each one adds.

Speed, with unchanged arithmetic:
    - Power and inlets are only recomputed when something that feeds them changed.
    - Finished workloads come off a heap ordered by end time, not from a scan of them all.
    - The decay factor exp(-h / RC) is computed once per substep length.
"""

from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass

import numpy as np

from config import ClusterSpec
from physics.airflow import heat_matrix
from physics.thermal import decay, gpu_power, relax, steady_state, throttle

MAX_SUBSTEP_S = 5.0

# Arrays with a leading scenario axis. Everything else on a Cluster is shared by its scenarios.
PER_SCENARIO = ("util", "temp_c", "power_w", "throttled", "inlet_c", "node_power_w")


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
    order: int  # bind sequence number: 0 for the first bind since reset, then 1, ...


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
        self._decay: dict[float, np.ndarray] = {}  # substep length -> exp(-h / RC) per GPU
        self.reset()

    # ----------------------------------------------------------------- state

    @property
    def n_gpus(self) -> int:
        return len(self.gpu_node)

    @property
    def n_scenarios(self) -> int:
        return len(self.util)

    @property
    def inlet_limit_c(self) -> float:
        return self.spec.inlet_limit_c

    def reset(self) -> None:
        self.time_s = 0.0
        self.util = np.zeros((1, self.n_gpus))
        self.owner = np.full(self.n_gpus, "", dtype=object)
        self.allocations: dict[str, Allocation] = {}
        self._expiries: list[tuple[float, int, str]] = []  # heap of (end_s, order, workload_id)
        self._binds = 0
        self.inlet_c = np.full((1, len(self.node_names)), self.spec.supply_temp_c)
        self._load_changed()
        # Settle the idle steady state (fixed point of throttle <-> inlet coupling).
        for _ in range(5):
            self._update_power()
        self.temp_c = steady_state(self.inlet_c[:, self.gpu_node], self.power_w, self.r)

    def copy(self) -> Cluster:
        """Independent copy. Spec and static arrays are shared."""
        new = copy.copy(self)
        for attr in PER_SCENARIO:
            setattr(new, attr, getattr(self, attr).copy())
        new.owner = self.owner.copy()
        new.allocations = dict(self.allocations)
        new._expiries = list(self._expiries)
        return new

    def branch(self, trial: Workload, node_names: list[str]) -> Cluster:
        """What-if copies: one scenario per name, each with `trial` added on that node.

        The trial workload has no end time in the branch, so it stays for as long as the
        branch is stepped. Workloads already bound expire on schedule in every scenario.
        Raises PlacementError if `trial` does not fit on one of the nodes. A branch is for
        stepping and reading; bind on the live cluster.
        """
        if self.n_scenarios != 1:
            raise ValueError("branch from a single-scenario cluster")
        picks = [self._pick_gpus(trial, name) for name in node_names]
        new = self.copy()
        for attr in PER_SCENARIO:
            setattr(new, attr, np.repeat(getattr(new, attr), len(node_names), axis=0))
        for scenario, gpus in enumerate(picks):
            new.util[scenario, gpus] = np.clip(trial.gpu_util, 0.0, 1.0)
        new._load_changed()
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

    def inlet_headroom_c(self) -> np.ndarray:
        """Per scenario: inlet limit minus the hottest inlet in the row."""
        return self.inlet_limit_c - self.inlet_c.max(axis=1)

    def gpu_headroom_c(self) -> np.ndarray:
        """Per scenario: the smallest gap between a GPU and its throttle point."""
        return (self.throttle_c - self.temp_c).min(axis=1)

    # ------------------------------------------------------------ placement

    def _pick_gpus(self, workload: Workload, node_name: str) -> np.ndarray:
        """The GPUs `workload` would take on `node_name`, or PlacementError."""
        if workload.workload_id in self.allocations:
            raise PlacementError(f"workload {workload.workload_id!r} is already bound")
        if node_name not in self.node_index:
            raise PlacementError(f"unknown node {node_name!r}")
        if workload.gpu_count <= 0:
            raise PlacementError("gpu_count must be positive")
        free = self.free_gpus(self.node_index[node_name])
        if len(free) < workload.gpu_count:
            raise PlacementError(
                f"node {node_name!r} has {len(free)} free GPUs, need {workload.gpu_count}"
            )
        return free[: workload.gpu_count]

    def can_bind(self, workload: Workload, node_name: str) -> bool:
        try:
            self._pick_gpus(workload, node_name)
        except PlacementError:
            return False
        return True

    def bind(self, workload: Workload, node_name: str) -> Allocation:
        if self.n_scenarios != 1:
            raise PlacementError("bind on the live cluster, not on a branch")
        gpus = self._pick_gpus(workload, node_name)
        self.owner[gpus] = workload.workload_id
        self.util[:, gpus] = np.clip(workload.gpu_util, 0.0, 1.0)
        end = self.time_s + workload.duration_s if workload.duration_s > 0 else math.inf
        alloc = Allocation(
            workload, self.node_index[node_name], tuple(int(g) for g in gpus), end, self._binds
        )
        self._binds += 1
        self.allocations[workload.workload_id] = alloc
        if end < math.inf:
            heapq.heappush(self._expiries, (end, alloc.order, workload.workload_id))
        self._load_changed()
        return alloc

    def unbind(self, workload_id: str) -> bool:
        alloc = self.allocations.pop(workload_id, None)
        if alloc is None:
            return False
        idx = list(alloc.gpus)
        self.owner[idx] = ""
        self.util[:, idx] = 0.0  # in every scenario: the schedule is shared
        self._load_changed()
        return True

    # ---------------------------------------------------------------- time

    def step(self, dt_s: float) -> list[str]:
        """Advance every scenario by dt_s. Returns ids of workloads that finished."""
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        finished: list[str] = []
        remaining = dt_s
        while remaining > 1e-9:
            h = min(MAX_SUBSTEP_S, remaining)
            self._update_power()
            if self._steady_c is None:
                self._steady_c = steady_state(self.inlet_c[:, self.gpu_node], self.power_w, self.r)
            if h not in self._decay:
                self._decay[h] = decay(self.r, self.c, h)
            self.temp_c = relax(self.temp_c, self._steady_c, self._decay[h])
            self.time_s += h
            remaining -= h
            finished += self._expire()
        return finished

    def _load_changed(self) -> None:
        """Utilization changed somewhere: demand, power and inlets must be recomputed."""
        self._demand_w = None
        self._settled = False

    def _update_power(self) -> None:
        """Steps 1-3: power from demand and inlet, then inlets from power.

        Skipped once settled. If an update hands back exactly the inlets it was given,
        the next one would run the same arithmetic on the same inputs and change nothing,
        so it can be skipped until utilization changes. Rows that are not throttling
        settle on the second update after a change; a row that keeps moving never skips.
        """
        if self._settled:
            return
        if self._demand_w is None:
            self._demand_w = gpu_power(self.util, self.tdp_w, self.spec.idle_gpu_fraction)
        self.power_w, self.throttled = throttle(
            self._demand_w, self.inlet_c[:, self.gpu_node], self.r, self.throttle_c
        )
        self.node_power_w = self.spec.base_w + self._per_node(self.power_w)
        # One matrix-vector product per scenario. A single matrix-matrix product would be
        # faster but rounds differently in the last bit, and results must reproduce exactly.
        recirculated = np.stack([self.heat @ p for p in self.node_power_w])
        inlet_c = self.spec.supply_temp_c + recirculated
        self._settled = np.array_equal(inlet_c, self.inlet_c)
        self.inlet_c = inlet_c
        self._steady_c = None  # depends on power and inlet

    def _per_node(self, per_gpu: np.ndarray) -> np.ndarray:
        """(scenarios, GPUs) -> (scenarios, nodes): each server's GPUs added up.

        A running total in GPU order. `sum(axis=2)` would pair terms up and round
        differently in the last bit.
        """
        by_node = per_gpu.reshape(len(per_gpu), len(self.node_names), self.spec.gpus_per_node)
        total = by_node[:, :, 0]
        for k in range(1, self.spec.gpus_per_node):
            total = total + by_node[:, :, k]
        return total

    def _expire(self) -> list[str]:
        """Unbind workloads whose end time has passed. Returns their ids in bind order."""
        done: list[tuple[int, str]] = []
        while self._expiries and self._expiries[0][0] <= self.time_s + 1e-9:
            _, order, wid = heapq.heappop(self._expiries)
            alloc = self.allocations.get(wid)
            # An entry is stale if its workload was unbound early (and maybe bound again).
            if alloc is not None and alloc.order == order:
                done.append((order, wid))
        done.sort()
        for _, wid in done:
            self.unbind(wid)
        return [wid for _, wid in done]
