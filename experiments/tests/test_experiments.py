import pytest
from fakes import NODES, FakeSim

from ortools_opt import OrToolsBatchPolicy
from plots import plot
from policies import (
    GREEDY_POLICIES,
    BinPackPolicy,
    CoolestInletPolicy,
    IsothermPolicy,
    SpreadPolicy,
)
from simulate import run_policy, write_metrics
from traces import TraceJob, read_trace


def job(i=0, gpus=2, submit=0.0, duration=60.0):
    return TraceJob(f"j{i}", submit, duration, gpus, 0.8)


@pytest.fixture
def sim():
    return FakeSim()


def test_coolest_and_isotherm_pick_cool_node(sim):
    state = sim.GetState(None)
    assert CoolestInletPolicy().choose(state, job(), sim) == "n-cool"
    assert IsothermPolicy().choose(state, job(), sim) == "n-cool"


def test_isotherm_skips_infeasible(sim):
    sim.ApplyEvent(_bind("fill-cool", "n-cool", 8))
    sim.ApplyEvent(_bind("fill-mid", "n-mid", 8))
    # Only n-hot fits; it's infeasible but still the sole option.
    assert IsothermPolicy().choose(sim.GetState(None), job(), sim) == "n-hot"


def test_binpack_vs_spread(sim):
    sim.ApplyEvent(_bind("x", "n-mid", 4))
    state = sim.GetState(None)
    assert BinPackPolicy().choose(state, job(), sim) == "n-mid"
    assert SpreadPolicy().choose(state, job(), sim) != "n-mid"


def test_every_policy_places_every_job(sim):
    jobs = [job(i, gpus=(i % 4) + 1, submit=i * 5.0) for i in range(12)]
    for name, cls in GREEDY_POLICIES.items():
        m = run_policy(sim, jobs, cls())
        assert m.placed == len(jobs), name
        assert m.peak_temp_c > 0


def test_ortools_respects_capacity_and_prefers_headroom(sim):
    jobs = [job(i, gpus=8) for i in range(3)]
    placement = OrToolsBatchPolicy().assign(sim.GetState(None), jobs, sim)
    # n-hot is infeasible, so only two 8-GPU jobs can be placed, on cool and mid.
    assert sorted(placement.values()) == ["n-cool", "n-mid"]


def test_ortools_in_simulation(sim):
    jobs = [job(i, gpus=4, submit=i * 10.0) for i in range(6)]
    m = run_policy(sim, jobs, OrToolsBatchPolicy())
    assert m.placed == len(jobs)


def test_results_roundtrip_and_plots(sim, tmp_path):
    rows = [run_policy(sim, [job()], cls()) for cls in (SpreadPolicy, IsothermPolicy)]
    out = tmp_path / "results.csv"
    write_metrics(rows, out)
    assert len(plot(out, tmp_path / "plots")) == 4


def test_read_trace(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("job_id,submit_s,duration_s,gpu_count,gpu_util\nb,5,60,2,0.5\na,0,30,1,0.9\n")
    assert [j.job_id for j in read_trace(p)] == ["a", "b"]


def _bind(wid, node, gpus):
    from gen import isotherm_pb2 as pb

    return pb.PodEvent(
        kind=pb.PodEvent.KIND_BIND,
        node=node,
        workload=pb.Workload(workload_id=wid, gpu_count=gpus, duration_s=1e9),
    )


assert set(NODES) == {"n-cool", "n-mid", "n-hot"}
