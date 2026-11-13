import csv
from pathlib import Path

import pytest

from loader import (
    Job,
    capacity_gpus,
    load_csv,
    offered_load,
    sweep,
    synthetic,
    write_csv,
)
from pods import GPU_RESOURCE, pod_manifest, pod_name


def test_synthetic_is_deterministic_and_sorted():
    a, b = synthetic(50, seed=7), synthetic(50, seed=7)
    assert a == b
    assert [j.submit_s for j in a] == sorted(j.submit_s for j in a)
    assert synthetic(50, seed=8) != a


REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
CAPACITY_GPUS = capacity_gpus(REPO_CONFIGS)


def test_capacity_from_repo_configs():
    # 10 racks x 4 slots x 8 GPUs today; this only checks the config is read, not frozen.
    assert CAPACITY_GPUS > 0
    assert CAPACITY_GPUS % 8 == 0


def test_capacity_from_config_dir(tmp_path):
    (tmp_path / "rack.yaml").write_text("layout:\n  n_racks: 3\n  slots_per_rack: 2\n")
    (tmp_path / "gpu_specs.yaml").write_text("node:\n  gpus_per_node: 4\n")
    assert capacity_gpus(tmp_path) == 24


def test_capacity_missing_config_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        capacity_gpus(tmp_path / "nope")


def test_sweep_writes_traces_and_index(tmp_path):
    index = sweep(tmp_path, loads=[0.4, 0.8], seeds=[0, 1], n_jobs=300, capacity=CAPACITY_GPUS)
    with index.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(float(r["target_load"]), int(r["seed"])) for r in rows] == [
        (0.4, 0),
        (0.4, 1),
        (0.8, 0),
        (0.8, 1),
    ]
    for r in rows:
        jobs = load_csv(tmp_path / r["trace"])
        assert len(jobs) == int(r["jobs"]) == 300
        assert int(r["capacity_gpus"]) == CAPACITY_GPUS
        assert float(r["offered_load"]) == pytest.approx(offered_load(jobs, CAPACITY_GPUS), 1e-3)
    # Same seed, different load: same jobs, different spacing.
    a, b = (load_csv(tmp_path / rows[i]["trace"]) for i in (0, 2))
    assert [j.gpu_count for j in a] == [j.gpu_count for j in b]
    assert a[-1].submit_s > b[-1].submit_s


@pytest.mark.parametrize("target", [0.3, 0.5, 0.7, 0.9])
def test_synthetic_hits_target_load(target):
    # 20k jobs: the measured load has about 1.3% relative standard error, so 5% is ~4 sigma.
    jobs = synthetic(20_000, target_load=target, capacity_gpus=CAPACITY_GPUS, seed=1)
    assert offered_load(jobs, CAPACITY_GPUS) == pytest.approx(target, rel=0.05)


def test_synthetic_load_scales_arrival_rate_only():
    lo = synthetic(200, target_load=0.3, capacity_gpus=CAPACITY_GPUS, seed=3)
    hi = synthetic(200, target_load=0.9, capacity_gpus=CAPACITY_GPUS, seed=3)
    assert [(j.duration_s, j.gpu_count, j.gpu_util) for j in lo] == [
        (j.duration_s, j.gpu_count, j.gpu_util) for j in hi
    ]
    assert hi[-1].submit_s < lo[-1].submit_s


@pytest.mark.parametrize("kwargs", [dict(target_load=0.5), dict(target_load=0.0, capacity_gpus=8)])
def test_synthetic_load_validation(kwargs):
    with pytest.raises(ValueError):
        synthetic(10, **kwargs)


def test_csv_roundtrip_rebases_time(tmp_path):
    jobs = [Job("b", 110.0, 60.0, 2, 0.5), Job("a", 100.0, 30.0, 1, 0.9)]
    path = tmp_path / "t.csv"
    write_csv(jobs, path)
    loaded = load_csv(path)
    assert [j.job_id for j in loaded] == ["a", "b"]
    assert [j.submit_s for j in loaded] == [0.0, 10.0]


def test_csv_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("job_id,submit_s\nx,0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_csv(path)


@pytest.mark.parametrize("bad", [dict(gpu_count=0), dict(duration_s=0), dict(gpu_util=1.5)])
def test_job_validation(bad):
    args = dict(job_id="j", submit_s=0.0, duration_s=10.0, gpu_count=1, gpu_util=0.5) | bad
    with pytest.raises(ValueError):
        Job(**args)


def test_pod_manifest():
    job = Job("Train_Job/42", 0.0, 60.0, 4, 0.75)
    m = pod_manifest(job, namespace="ns", scheduler_name="isotherm-scheduler", run_id="r1")
    assert m["metadata"]["name"] == pod_name(job) == "train-job-42"
    assert m["spec"]["schedulerName"] == "isotherm-scheduler"
    limits = m["spec"]["containers"][0]["resources"]["limits"]
    assert limits[GPU_RESOURCE] == "4"
    assert "isotherm.io/duration-s" not in m["metadata"]["annotations"]
