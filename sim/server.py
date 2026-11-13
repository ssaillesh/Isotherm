"""gRPC server exposing the simulator through proto/isotherm.proto.

uv run python server.py --port 50051 --metrics-port 9100
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent import futures
from pathlib import Path

import grpc
from prometheus_client import Gauge, start_http_server

from config import load_cluster
from gen import isotherm_pb2 as pb
from gen import isotherm_pb2_grpc as pb_grpc
from physics import Cluster, PlacementError, Workload
from surrogate import DEFAULT_HORIZON_S, FEATURE_SCHEMA, Predictor, make_predictor

log = logging.getLogger("isotherm.sim")

GPU_TEMP = Gauge("isotherm_gpu_temp_c", "GPU temperature", ["node", "rack", "gpu"])
GPU_POWER = Gauge("isotherm_gpu_power_w", "GPU power draw", ["node", "rack", "gpu"])
GPU_THROTTLED = Gauge(
    "isotherm_gpu_throttled", "1 if the GPU is throttling", ["node", "rack", "gpu"]
)
NODE_INLET = Gauge("isotherm_node_inlet_c", "Node inlet air temperature", ["node", "rack"])
NODE_POWER = Gauge("isotherm_node_power_w", "Whole-server power", ["node", "rack"])
SIM_TIME = Gauge("isotherm_sim_time_s", "Simulated time")


def _workload(msg: pb.Workload) -> Workload:
    return Workload(msg.workload_id, msg.gpu_count, msg.gpu_util, msg.duration_s)


def cluster_state(cluster: Cluster) -> pb.ClusterState:
    nodes = []
    for i, name in enumerate(cluster.node_names):
        spec = cluster.spec.nodes[i]
        g = cluster.node_gpus(i)
        gpus = [
            pb.GpuState(
                index=k,
                temp_c=float(cluster.temp_c[j]),
                power_w=float(cluster.power_w[j]),
                util=float(cluster.util[j]),
                throttled=bool(cluster.throttled[j]),
                workload_id=str(cluster.owner[j]),
            )
            for k, j in enumerate(range(g.start, g.stop))
        ]
        nodes.append(
            pb.NodeState(
                name=name,
                rack=str(spec.rack),
                slot=spec.slot,
                gpu_model=spec.gpu.model,
                inlet_c=float(cluster.inlet_c[i]),
                free_gpus=len(cluster.free_gpus(i)),
                gpus=gpus,
                throttle_c=spec.gpu.throttle_c,
                power_w=float(cluster.node_power_w[i]),
                exhaust_c=float(cluster.exhaust_c[i]),
                inlet_limit_c=cluster.inlet_limit_c,
            )
        )
    return pb.ClusterState(sim_time_s=cluster.time_s, nodes=nodes)


def export_metrics(cluster: Cluster) -> None:
    SIM_TIME.set(cluster.time_s)
    for i, name in enumerate(cluster.node_names):
        rack = str(cluster.spec.nodes[i].rack)
        NODE_INLET.labels(name, rack).set(cluster.inlet_c[i])
        NODE_POWER.labels(name, rack).set(cluster.node_power_w[i])
        g = cluster.node_gpus(i)
        for k, j in enumerate(range(g.start, g.stop)):
            GPU_TEMP.labels(name, rack, str(k)).set(cluster.temp_c[j])
            GPU_POWER.labels(name, rack, str(k)).set(cluster.power_w[j])
            GPU_THROTTLED.labels(name, rack, str(k)).set(float(cluster.throttled[j]))


class ThermalSimServicer(pb_grpc.ThermalSimServicer):
    def __init__(self, cluster: Cluster, predictors: dict[str, Predictor], default_predictor: str):
        if default_predictor not in predictors:
            raise ValueError(f"default predictor {default_predictor!r} is not loaded")
        self.cluster = cluster
        self.predictors = predictors
        self.default_predictor = default_predictor
        self.lock = threading.Lock()

    def GetState(self, request, context):
        with self.lock:
            return cluster_state(self.cluster)

    def ScoreCandidates(self, request, context):
        name = request.predictor or self.default_predictor
        predictor = self.predictors.get(name)
        if predictor is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"predictor {name!r} is not loaded")
        horizon = request.horizon_s or DEFAULT_HORIZON_S
        with self.lock:
            scores = predictor.score(
                self.cluster, _workload(request.workload), list(request.candidate_nodes), horizon
            )
        return pb.ScoreResponse(
            scores=[
                pb.CandidateScore(
                    node=s.node,
                    predicted_peak_c=s.predicted_peak_c,
                    headroom_c=s.headroom_c,
                    predicted_peak_inlet_c=s.predicted_peak_inlet_c,
                    inlet_headroom_c=s.inlet_headroom_c,
                    gpu_headroom_c=s.gpu_headroom_c,
                    feasible=s.feasible,
                    features=s.features,
                )
                for s in scores
            ],
            predictor=predictor.name,
            feature_schema=FEATURE_SCHEMA,
        )

    def ApplyEvent(self, request, context):
        with self.lock:
            try:
                if request.kind == pb.PodEvent.KIND_BIND:
                    self.cluster.bind(_workload(request.workload), request.node)
                elif request.kind == pb.PodEvent.KIND_UNBIND:
                    if not self.cluster.unbind(request.workload.workload_id):
                        return pb.ApplyEventResponse(ok=False, error="workload not bound")
                else:
                    return pb.ApplyEventResponse(ok=False, error="event kind is unspecified")
            except PlacementError as e:
                return pb.ApplyEventResponse(ok=False, error=str(e))
        return pb.ApplyEventResponse(ok=True)

    def Step(self, request, context):
        if request.dt_s < 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "dt_s must be non-negative")
        with self.lock:
            self.cluster.step(request.dt_s)
            export_metrics(self.cluster)
            return cluster_state(self.cluster)

    def Reset(self, request, context):
        with self.lock:
            self.cluster.reset()
            export_metrics(self.cluster)
            return cluster_state(self.cluster)


def build_server(servicer: ThermalSimServicer, address: str) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_ThermalSimServicer_to_server(servicer, server)
    port = server.add_insecure_port(address)
    return server, port


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--metrics-port", type=int, default=9100, help="0 disables metrics")
    p.add_argument("--predictor", default="rollout", choices=["rollout", "learned"])
    p.add_argument("--model-path", type=Path, help="joblib model for the learned predictor")
    p.add_argument("--rack", type=Path, help="override configs/rack.yaml")
    p.add_argument("--gpu-specs", type=Path, help="override configs/gpu_specs.yaml")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cluster = Cluster(load_cluster(args.rack, args.gpu_specs))
    predictors = {"rollout": make_predictor("rollout")}
    if args.model_path:
        predictors["learned"] = make_predictor("learned", args.model_path)
    servicer = ThermalSimServicer(cluster, predictors, args.predictor)

    if args.metrics_port:
        start_http_server(args.metrics_port)
        export_metrics(cluster)
    server, port = build_server(servicer, f"[::]:{args.port}")
    server.start()
    log.info(
        "thermal sim listening on :%d (%d nodes, %d GPUs, predictor=%s)",
        port,
        len(cluster.node_names),
        cluster.n_gpus,
        args.predictor,
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()
