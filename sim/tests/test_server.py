"""End-to-end over a real gRPC channel: exercises the contract, not just the Python objects."""

import grpc
import pytest

from gen import isotherm_pb2 as pb
from gen import isotherm_pb2_grpc as pb_grpc
from server import ThermalSimServicer, build_server
from surrogate import FEATURE_SCHEMA, make_predictor


@pytest.fixture
def stub(cluster):
    servicer = ThermalSimServicer(cluster, {"rollout": make_predictor("rollout")}, "rollout")
    server, port = build_server(servicer, "127.0.0.1:0")
    server.start()
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        yield pb_grpc.ThermalSimStub(channel)
    server.stop(None)


def test_get_state(stub, cluster):
    state = stub.GetState(pb.GetStateRequest())
    assert [n.name for n in state.nodes] == cluster.node_names
    assert all(n.free_gpus == len(n.gpus) for n in state.nodes)


def test_score_bind_step_reset(stub, cluster):
    node = cluster.node_names[0]
    work = pb.Workload(workload_id="default/job-1", gpu_count=4, gpu_util=1.0, duration_s=120)

    resp = stub.ScoreCandidates(
        pb.ScoreRequest(workload=work, candidate_nodes=[node], horizon_s=60)
    )
    assert resp.predictor == "rollout"
    assert resp.feature_schema == FEATURE_SCHEMA
    assert resp.scores[0].feasible

    assert stub.ApplyEvent(pb.PodEvent(kind=pb.PodEvent.KIND_BIND, workload=work, node=node)).ok
    state = stub.Step(pb.StepRequest(dt_s=60))
    assert state.sim_time_s == pytest.approx(60)
    assert state.nodes[0].free_gpus == len(state.nodes[0].gpus) - 4

    state = stub.Step(pb.StepRequest(dt_s=90))  # duration elapsed -> auto-released
    assert state.nodes[0].free_gpus == len(state.nodes[0].gpus)

    state = stub.Reset(pb.ResetRequest())
    assert state.sim_time_s == 0


def test_apply_event_reports_errors(stub):
    bad = pb.PodEvent(
        kind=pb.PodEvent.KIND_BIND, workload=pb.Workload(workload_id="x", gpu_count=1), node="nope"
    )
    resp = stub.ApplyEvent(bad)
    assert not resp.ok and "unknown node" in resp.error
    assert not stub.ApplyEvent(pb.PodEvent(kind=pb.PodEvent.KIND_UNSPECIFIED)).ok


def test_unknown_predictor_is_invalid_argument(stub):
    with pytest.raises(grpc.RpcError) as e:
        stub.ScoreCandidates(pb.ScoreRequest(predictor="learned"))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
