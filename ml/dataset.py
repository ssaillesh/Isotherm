"""Build a surrogate training set by querying the simulator's rollout oracle over gRPC.

For each sample: reset the sim, load it with random workloads, let it warm up, then ask
ScoreCandidates(predictor="rollout") about a random new workload. The sim returns both
the inputs (CandidateScore.features) and the labels (inlet/gpu headroom). This module
never derives a thermal quantity itself.

    uv run python dataset.py --states 300 --out ../data/surrogate.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from gen import isotherm_pb2 as pb

TARGETS = ["inlet_headroom_c", "gpu_headroom_c"]
GPU_CHOICES = (1, 2, 4, 8)

log = logging.getLogger("isotherm.ml")


def _random_load(sim, rng: np.random.Generator, state: pb.ClusterState) -> None:
    fill = rng.uniform(0.0, 1.0)
    for node in state.nodes:
        free = node.free_gpus
        i = 0
        while free > 0 and rng.uniform() < fill:
            gpus = int(rng.choice([g for g in GPU_CHOICES if g <= free]))
            sim.ApplyEvent(
                pb.PodEvent(
                    kind=pb.PodEvent.KIND_BIND,
                    node=node.name,
                    workload=pb.Workload(
                        workload_id=f"bg/{node.name}/{i}",
                        gpu_count=gpus,
                        gpu_util=float(rng.uniform(0.2, 1.0)),
                    ),
                )
            )
            free -= gpus
            i += 1


def generate(
    sim, n_states: int, seed: int = 0, horizon_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    xs, ys, schema = [], [], ""
    for k in range(n_states):
        state = sim.Reset(pb.ResetRequest())
        _random_load(sim, rng, state)
        state = sim.Step(pb.StepRequest(dt_s=float(rng.uniform(0, 1200))))

        gpus = int(rng.choice(GPU_CHOICES))
        candidates = [n.name for n in state.nodes if n.free_gpus >= gpus]
        if not candidates:
            continue
        resp = sim.ScoreCandidates(
            pb.ScoreRequest(
                workload=pb.Workload(
                    workload_id=f"probe/{k}", gpu_count=gpus, gpu_util=float(rng.uniform(0.3, 1.0))
                ),
                candidate_nodes=candidates,
                horizon_s=horizon_s,
                predictor="rollout",
            )
        )
        if schema and resp.feature_schema != schema:
            raise RuntimeError("feature schema changed mid-run")
        schema = resp.feature_schema
        for s in resp.scores:
            if s.features:
                xs.append(list(s.features))
                ys.append([getattr(s, t) for t in TARGETS])
        if (k + 1) % 50 == 0:
            log.info("%d/%d states, %d samples", k + 1, n_states, len(xs))
    return np.array(xs), np.array(ys), schema


def save(path: Path, x: np.ndarray, y: np.ndarray, schema: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, feature_schema=schema, targets=np.array(TARGETS))


def main() -> None:
    import grpc

    from gen import isotherm_pb2_grpc as pb_grpc

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sim", default="localhost:50051")
    p.add_argument("--states", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    with grpc.insecure_channel(args.sim) as channel:
        x, y, schema = generate(pb_grpc.ThermalSimStub(channel), args.states, args.seed)
    save(args.out, x, y, schema)
    log.info("wrote %d samples (%s) to %s", len(x), schema.split(":")[0], args.out)


if __name__ == "__main__":
    main()
