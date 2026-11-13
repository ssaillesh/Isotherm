import csv
from pathlib import Path

import pytest
from fakes import GPUS_PER_NODE, NODES, FakeSim

from plots import plot_sweep
from policies import BinPackPolicy, IsothermPolicy, SpreadPolicy
from simulate import Tick, Trajectory, run_policy, simulate
from sweep import (
    IndexRow,
    append_row,
    check_rules,
    percentile,
    read_index,
    read_rows,
    run_steady_state,
    steady_state_metrics,
    steady_state_window,
    summarise,
    summary_path,
    window_offered_load,
    write_summary,
)
from traces import TraceJob

CAPACITY = len(NODES) * GPUS_PER_NODE


def jobs_every(gap_s, n, gpus=2, duration=100.0):
    return [TraceJob(f"j{i}", i * gap_s, duration, gpus, 0.8) for i in range(n)]


def test_window_is_10_to_90_percent_of_arrivals():
    jobs = [TraceJob(f"j{i}", 50.0 + i * 10, 60.0, 1, 0.5) for i in range(101)]  # 50..1050
    assert steady_state_window(jobs) == pytest.approx((150.0, 950.0))
    assert steady_state_window(jobs, (0.0, 1.0)) == pytest.approx((50.0, 1050.0))
    with pytest.raises(ValueError):
        steady_state_window(jobs, (0.5, 0.5))
    with pytest.raises(ValueError):
        steady_state_window(jobs[:1])


def test_window_offered_load_counts_only_window_arrivals():
    jobs = [TraceJob("early", 0, 1e6, 8, 1.0)] + jobs_every(10.0, 100, gpus=2, duration=60.0)
    # In [100, 200): 10 arrivals x 2 GPUs x 60 s = 1200 GPU-s over 100 s x 24 GPUs.
    assert window_offered_load(jobs, 100.0, 200.0, CAPACITY) == pytest.approx(1200 / 2400)


def test_steady_state_metrics_ignore_ticks_outside_window():
    def tick(t, busy, peak):
        return Tick(t, busy, busy, peak, busy * 50.0, 20.0, 5.0, 0)

    traj = Trajectory(tick_s=10.0, capacity_gpus=10)
    traj.ticks = [tick(10, 10, 99.0), tick(20, 4, 60.0), tick(30, 6, 70.0), tick(40, 10, 99.0)]
    jobs = [TraceJob("a", 12.0, 5.0, 1, 0.5), TraceJob("b", 35.0, 5.0, 1, 0.5)]
    traj.placed_at = {"a": 20.0, "b": 40.0}
    row = steady_state_metrics(traj, jobs, 10.0, 30.0, "p", 0.5, 3)
    assert row.achieved_utilisation == pytest.approx(0.5)  # ticks at 20 and 30 only
    assert row.peak_temp_c == 70.0
    assert row.mean_busy_temp_c == pytest.approx(50.0)
    assert (row.window_jobs, row.window_placed, row.mean_wait_s) == (1, 1, 8.0)
    assert (row.target_load, row.seed) == (0.5, 3)


def test_steady_state_run_measures_offered_and_achieved_load():
    # 2 GPUs x 100 s every 20 s = 10 busy GPUs on average, under capacity: no queueing.
    jobs = jobs_every(20.0, 200)
    row = run_steady_state(FakeSim(), jobs, SpreadPolicy(), target_load=0.4, seed=0)
    assert row.offered_load == pytest.approx(10 / CAPACITY, rel=0.02)
    assert row.achieved_utilisation == pytest.approx(row.offered_load, rel=0.05)
    assert row.window_placed == row.window_jobs > 0
    assert row.mean_wait_s == 0.0
    assert row.min_headroom_c < row.peak_inlet_c  # read from the sim's limits, not invented
    # Stops at the window's end, not after the drain.
    assert row.sim_end_s <= row.window_end_s + 10.0


def test_steady_state_run_waits_for_window_jobs_under_overload():
    # 8-GPU jobs of 200 s every 20 s: 80 GPUs asked for on 24, so the queue grows.
    jobs = jobs_every(20.0, 60, gpus=8, duration=200.0)
    row = run_steady_state(FakeSim(), jobs, BinPackPolicy(), target_load=3.0, seed=0)
    assert row.offered_load > 1.0
    assert row.achieved_utilisation == pytest.approx(1.0)
    assert row.window_placed == row.window_jobs
    assert row.mean_wait_s > 0
    assert row.sim_end_s > row.window_end_s


def test_run_policy_unchanged_by_refactor():
    jobs = jobs_every(5.0, 12, gpus=3)
    m = run_policy(FakeSim(), jobs, IsothermPolicy())
    assert (m.jobs, m.placed) == (12, 12)
    traj = simulate(FakeSim(), jobs, IsothermPolicy())
    assert traj.ticks[-1].busy_gpus == 0
    assert m.makespan_s == traj.ticks[-1].t_s


def _index(loads, seeds, jobs=2000):
    return [IndexRow(Path(f"l{lo}s{s}.csv"), lo, s, jobs, CAPACITY) for lo in loads for s in seeds]


def test_rules_need_2000_jobs_and_5_seeds():
    check_rules(_index([0.5, 0.9], range(5)))
    with pytest.raises(ValueError, match="fewer than 2000 jobs"):
        check_rules(_index([0.5], range(5), jobs=1999))
    with pytest.raises(ValueError, match="fewer than 5 seeds"):
        check_rules(_index([0.5], range(5)) + _index([0.9], range(4)))
    with pytest.raises(ValueError):
        check_rules([])


def test_read_index_resolves_trace_paths(tmp_path):
    (tmp_path / "index.csv").write_text(
        "trace,target_load,seed,jobs,capacity_gpus,offered_load\n"
        "load0.50-seed0.csv,0.5,0,2000,320,0.49\n"
    )
    [row] = read_index(tmp_path / "index.csv")
    assert row.trace == tmp_path / "load0.50-seed0.csv"
    assert (row.target_load, row.seed, row.jobs, row.capacity_gpus) == (0.5, 0, 2000, 320)


def test_percentile_nearest_rank():
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile(list(map(float, range(1, 101))), 95) == 95.0


def test_rows_roundtrip_summary_and_plot(tmp_path):
    out = tmp_path / "sweep.csv"
    jobs = jobs_every(20.0, 100)
    for load, gap in ((0.3, 30.0), (0.6, 15.0)):
        for seed in range(3):
            for cls in (SpreadPolicy, BinPackPolicy):
                trace = jobs_every(gap, 100) if seed else jobs
                append_row(run_steady_state(FakeSim(), trace, cls(), load, seed), out)
    rows = read_rows(out)
    assert len(rows) == 12
    assert {(r.policy, r.target_load, r.seed) for r in rows} >= {("spread", 0.3, 0)}

    summary = summarise(rows)
    assert [(s["policy"], s["target_load"], s["seeds"]) for s in summary] == [
        ("binpack", 0.3, 3),
        ("spread", 0.3, 3),
        ("binpack", 0.6, 3),
        ("spread", 0.6, 3),
    ]
    assert {"offered_load_mean", "achieved_utilisation_std", "peak_temp_c_mean"} <= set(summary[0])
    assert "seed_mean" not in summary[0]

    path = summary_path(out)
    write_summary(summary, path)
    with path.open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 4
    assert plot_sweep(path, tmp_path / "plots").stat().st_size > 0
