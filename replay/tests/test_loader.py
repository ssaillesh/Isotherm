import pytest

from loader import Job, load_csv, offered_load, synthetic, write_csv
from pods import GPU_RESOURCE, pod_manifest, pod_name


def test_synthetic_is_deterministic_and_sorted():
    a, b = synthetic(50, seed=7), synthetic(50, seed=7)
    assert a == b
    assert [j.submit_s for j in a] == sorted(j.submit_s for j in a)
    assert synthetic(50, seed=8) != a


CAPACITY_GPUS = 320  # test value; matches the current configs/ (10 racks x 4 nodes x 8 GPUs)


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
