"""Predictor interface: given the current cluster and a workload, score candidate nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from physics import Cluster, Workload

DEFAULT_HORIZON_S = 600.0


@dataclass(frozen=True)
class CandidateScore:
    node: str
    feasible: bool
    predicted_peak_c: float = 0.0  # hottest GPU in the row
    predicted_peak_inlet_c: float = 0.0  # hottest inlet in the row
    inlet_headroom_c: float = 0.0
    gpu_headroom_c: float = 0.0
    features: list[float] = field(default_factory=list)

    @property
    def headroom_c(self) -> float:
        """The binding margin: whichever limit is closer."""
        return min(self.inlet_headroom_c, self.gpu_headroom_c)


class Predictor(Protocol):
    name: str

    def score(
        self, cluster: Cluster, workload: Workload, candidates: list[str], horizon_s: float
    ) -> list[CandidateScore]: ...
