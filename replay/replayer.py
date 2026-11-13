"""Replay a trace into the kwok cluster while driving the simulator's clock.

The replayer is the clock in a kwok run: each tick it submits due pods, notices which
ones the scheduler has bound, deletes finished ones (sending UNBIND to the sim), and
calls Step so the simulator advances. It records the per-tick cluster state the sim
reports; it never computes a temperature itself.

    uv run python replayer.py ../data/synthetic.csv --scheduler isotherm-scheduler \
        --out ../docs/results/replay-isotherm.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from gen import isotherm_pb2 as pb
from loader import Job, load_csv
from pods import LABEL_RUN, pod_manifest, pod_name

log = logging.getLogger("isotherm.replay")


class Kube(Protocol):
    def create(self, manifest: dict) -> None: ...
    def bound(self) -> dict[str, str]:
        """Pod name -> node name for every bound pod in this run."""
        ...

    def delete(self, name: str) -> None: ...


class SimClient(Protocol):
    def Reset(self, request: pb.ResetRequest) -> pb.ClusterState: ...
    def Step(self, request: pb.StepRequest) -> pb.ClusterState: ...
    def ApplyEvent(self, request: pb.PodEvent) -> pb.ApplyEventResponse: ...


@dataclass
class TickRecord:
    sim_time_s: float
    pending: int
    running: int
    max_temp_c: float
    throttled_gpus: int


@dataclass
class Replayer:
    jobs: list[Job]
    sim: SimClient
    kube: Kube
    namespace: str = "default"
    scheduler: str = "isotherm-scheduler"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    tick_s: float = 10.0
    speedup: float = 60.0  # simulated seconds per wall-clock second
    sleep: Callable[[float], None] = time.sleep
    records: list[TickRecord] = field(default_factory=list)

    def run(self, max_sim_s: float = float("inf")) -> list[TickRecord]:
        self.sim.Reset(pb.ResetRequest())
        queue = deque(sorted(self.jobs, key=lambda j: j.submit_s))
        submitted: dict[str, Job] = {}
        running: dict[str, tuple[Job, float]] = {}  # pod -> (job, end time)
        t = 0.0

        while (queue or submitted or running) and t <= max_sim_s:
            while queue and queue[0].submit_s <= t:
                job = queue.popleft()
                self.kube.create(
                    pod_manifest(
                        job,
                        namespace=self.namespace,
                        scheduler_name=self.scheduler,
                        run_id=self.run_id,
                    )
                )
                submitted[pod_name(job)] = job

            for name in self.kube.bound():
                if name in submitted:
                    job = submitted.pop(name)
                    running[name] = (job, t + job.duration_s)

            for name, (_job, end) in list(running.items()):
                if end <= t:
                    self.kube.delete(name)
                    self.sim.ApplyEvent(
                        pb.PodEvent(
                            kind=pb.PodEvent.KIND_UNBIND,
                            workload=pb.Workload(workload_id=f"{self.namespace}/{name}"),
                        )
                    )
                    del running[name]

            state = self.sim.Step(pb.StepRequest(dt_s=self.tick_s))
            self.records.append(self._record(state, len(submitted), len(running)))
            t += self.tick_s
            self.sleep(self.tick_s / self.speedup)
        return self.records

    @staticmethod
    def _record(state: pb.ClusterState, pending: int, running: int) -> TickRecord:
        gpus = [g for n in state.nodes for g in n.gpus]
        return TickRecord(
            sim_time_s=state.sim_time_s,
            pending=pending,
            running=running,
            max_temp_c=max((g.temp_c for g in gpus), default=0.0),
            throttled_gpus=sum(g.throttled for g in gpus),
        )


class KubernetesKube:
    """Kube backed by the official client, scoped to one replay run's label."""

    def __init__(self, namespace: str, run_id: str, kubeconfig: str | None = None):
        from kubernetes import client, config

        config.load_kube_config(config_file=kubeconfig)
        self.api = client.CoreV1Api()
        self.namespace = namespace
        self.selector = f"{LABEL_RUN}={run_id}"

    def create(self, manifest: dict) -> None:
        self.api.create_namespaced_pod(self.namespace, manifest)

    def bound(self) -> dict[str, str]:
        pods = self.api.list_namespaced_pod(self.namespace, label_selector=self.selector)
        return {p.metadata.name: p.spec.node_name for p in pods.items if p.spec.node_name}

    def delete(self, name: str) -> None:
        self.api.delete_namespaced_pod(name, self.namespace, grace_period_seconds=0)


def write_records(records: Iterable[TickRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_time_s", "pending", "running", "max_temp_c", "throttled_gpus"])
        w.writerows(
            (r.sim_time_s, r.pending, r.running, r.max_temp_c, r.throttled_gpus) for r in records
        )


def main() -> None:
    import grpc

    from gen import isotherm_pb2_grpc as pb_grpc

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("trace", type=Path, help="normalized trace CSV (see loader.py)")
    p.add_argument("--sim", default="localhost:50051")
    p.add_argument(
        "--scheduler",
        default="isotherm-scheduler",
        help="schedulerName for pods; use default-scheduler for the baseline",
    )
    p.add_argument("--namespace", default="default")
    p.add_argument("--kubeconfig")
    p.add_argument("--tick-s", type=float, default=10.0)
    p.add_argument("--speedup", type=float, default=60.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    run_id = uuid.uuid4().hex[:8]
    with grpc.insecure_channel(args.sim) as channel:
        replayer = Replayer(
            jobs=load_csv(args.trace, args.limit),
            sim=pb_grpc.ThermalSimStub(channel),
            kube=KubernetesKube(args.namespace, run_id, args.kubeconfig),
            namespace=args.namespace,
            tick_s=args.tick_s,
            speedup=args.speedup,
            scheduler=args.scheduler,
            run_id=run_id,
        )
        records = replayer.run()
    write_records(records, args.out)
    log.info("run %s: %d ticks written to %s", run_id, len(records), args.out)


if __name__ == "__main__":
    main()
