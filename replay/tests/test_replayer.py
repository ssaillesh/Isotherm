from gen import isotherm_pb2 as pb
from loader import Job
from replayer import Replayer


class FakeKube:
    """Binds every pod to 'node-0' as soon as it is created."""

    def __init__(self):
        self.pods: dict[str, str] = {}
        self.created: list[str] = []
        self.deleted: list[str] = []

    def create(self, manifest):
        name = manifest["metadata"]["name"]
        self.created.append(name)
        self.pods[name] = "node-0"

    def bound(self):
        return dict(self.pods)

    def delete(self, name):
        self.deleted.append(name)
        self.pods.pop(name)


class FakeSim:
    def __init__(self):
        self.t = 0.0
        self.unbinds: list[str] = []

    def Reset(self, req):
        self.t = 0.0
        return pb.ClusterState()

    def Step(self, req):
        self.t += req.dt_s
        gpu = pb.GpuState(temp_c=60.0 + self.t / 100, throttled=False)
        return pb.ClusterState(sim_time_s=self.t, nodes=[pb.NodeState(name="node-0", gpus=[gpu])])

    def ApplyEvent(self, req):
        assert req.kind == pb.PodEvent.KIND_UNBIND
        self.unbinds.append(req.workload.workload_id)
        return pb.ApplyEventResponse(ok=True)


def test_replay_lifecycle():
    jobs = [Job("a", 0.0, 30.0, 1, 0.5), Job("b", 20.0, 10.0, 2, 0.9)]
    kube, sim = FakeKube(), FakeSim()
    records = Replayer(jobs, sim, kube, namespace="ns", tick_s=10.0, sleep=lambda _: None).run()

    assert kube.created == ["a", "b"]
    assert sorted(kube.deleted) == ["a", "b"]
    assert sorted(sim.unbinds) == ["ns/a", "ns/b"]
    assert records[-1].running == 0 and records[-1].pending == 0
    # Temperatures come straight from the simulator's reported state.
    assert records[0].max_temp_c == 60.0 + 10.0 / 100
