import joblib
import numpy as np

from dataset import TARGETS, generate, save
from gen import isotherm_pb2 as pb
from train import load, train

N_NODES, GPUS, N_FEATURES = 4, 8, 5


class FakeSim:
    """Contract-level double. Features and labels are canned numbers, not physics."""

    def __init__(self):
        self.used = {}
        self.calls = []

    def _state(self):
        return pb.ClusterState(
            nodes=[
                pb.NodeState(name=f"n{i}", free_gpus=GPUS - self.used.get(f"n{i}", 0))
                for i in range(N_NODES)
            ]
        )

    def Reset(self, req):
        self.used = {}
        return self._state()

    def Step(self, req):
        return self._state()

    def ApplyEvent(self, req):
        self.used[req.node] = self.used.get(req.node, 0) + req.workload.gpu_count
        return pb.ApplyEventResponse(ok=True)

    def ScoreCandidates(self, req):
        self.calls.append(req)
        return pb.ScoreResponse(
            feature_schema="test:v1",
            scores=[
                pb.CandidateScore(
                    node=n,
                    features=[float(i)] * N_FEATURES,
                    inlet_headroom_c=8.0 - i,
                    gpu_headroom_c=20.0 - 2 * i,
                )
                for i, n in enumerate(req.candidate_nodes)
            ],
        )


def test_generate_uses_rollout_oracle_and_only_fitting_nodes():
    sim = FakeSim()
    x, y, schema = generate(sim, n_states=20, seed=1)
    assert schema == "test:v1"
    assert x.shape[1] == N_FEATURES and y.shape[1] == len(TARGETS)
    assert len(x) == len(y) > 0
    assert all(c.predictor == "rollout" for c in sim.calls)


def test_dataset_roundtrip_and_training(tmp_path):
    x, y, schema = generate(FakeSim(), n_states=60, seed=2)
    path = tmp_path / "d.npz"
    save(path, x, y, schema)
    x2, y2, schema2, targets = load(path)
    assert schema2 == schema and targets == TARGETS
    np.testing.assert_array_equal(x, x2)

    model, metrics = train(x2, y2, targets)
    assert set(metrics) >= {f"mae_{t}" for t in TARGETS}
    bundle_path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "feature_schema": schema, "targets": targets}, bundle_path)
    pred = joblib.load(bundle_path)["model"].predict(x2[:3])
    assert pred.shape == (3, len(TARGETS))
