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
    record_config,
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
        return Tick(t, busy, busy, peak, busy * 50.0, 20.0, 5.0, 0, 0.0, 0, 0.0)

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


def test_simulate_asks_for_state_only_after_a_bind():
    # Step already returns the state the next tick needs; one GetState per placement.
    sim = FakeSim()
    jobs = jobs_every(50.0, 6)
    traj = simulate(sim, jobs, SpreadPolicy())
    assert len(traj.ticks) > len(jobs)
    assert sim.get_state_calls == len(traj.placed_at) == len(jobs)

    sim = FakeSim()
    simulate(sim, [], SpreadPolicy())
    assert sim.get_state_calls == 0


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


class HotSim(FakeSim):
    """FakeSim, but the state at t = 20 s (the tick (10, 20]) breaks the chosen limits by 1°C."""

    def __init__(self, hot_inlets=(), hot_gpu=None):
        self.hot_inlets, self.hot_gpu = set(hot_inlets), hot_gpu
        super().__init__()

    def _state(self):
        state = super()._state()
        if self.t == 20.0:
            for n in state.nodes:
                if n.name in self.hot_inlets:
                    n.inlet_c = n.inlet_limit_c + 1.0
                if n.name == self.hot_gpu:
                    n.gpus[0].temp_c = n.throttle_c + 1.0
        return state


LIMIT_FIELDS = ("inlet_exceedance_cs", "time_over_inlet_s", "gpu_exceedance_cs")


def _limits(m):
    return tuple(getattr(m, f) for f in LIMIT_FIELDS)


def test_one_server_1c_over_inlet_limit_for_10s_is_10_cs():
    m = run_policy(HotSim(hot_inlets=["n-hot"]), jobs_every(20.0, 5), SpreadPolicy())
    assert _limits(m) == (10.0, 10.0, 0.0)


def test_inlet_time_counts_seconds_not_servers():
    m = run_policy(HotSim(hot_inlets=["n-hot", "n-cool"]), jobs_every(20.0, 5), SpreadPolicy())
    assert _limits(m) == (20.0, 10.0, 0.0)  # two servers 1°C over, in the same 10 s


def test_one_gpu_1c_over_throttle_for_10s_is_10_cs():
    m = run_policy(HotSim(hot_gpu="n-mid"), jobs_every(20.0, 5), SpreadPolicy())
    assert _limits(m) == (0.0, 0.0, 10.0)


def test_cool_runs_break_no_limits():
    assert _limits(run_policy(FakeSim(), jobs_every(20.0, 5), SpreadPolicy())) == (0.0, 0.0, 0.0)
    row = run_steady_state(FakeSim(), jobs_every(10.0, 11), SpreadPolicy(), 0.5, 0)
    assert _limits(row) == (0.0, 0.0, 0.0)


def test_steady_state_counts_limits_inside_the_window_only():
    jobs = jobs_every(10.0, 11)  # arrivals 0..100 s, window [10, 90)
    hot = HotSim(hot_inlets=["n-hot"], hot_gpu="n-hot")
    row = run_steady_state(hot, jobs, SpreadPolicy(), 0.5, 0)
    assert _limits(row) == (10.0, 10.0, 10.0)  # the tick ending at 20 s is in the window
    narrow = run_steady_state(hot, jobs, SpreadPolicy(), 0.5, 0, window=(0.3, 0.9))
    assert _limits(narrow) == (0.0, 0.0, 0.0)  # window [30, 90) leaves it out


def test_resume_refuses_a_csv_with_other_columns(tmp_path):
    old = tmp_path / "sweep.csv"
    old.write_text("policy,target_load,seed,peak_temp_c\nbinpack,0.3,0,72.5\n")
    with pytest.raises(ValueError, match="missing"):
        read_rows(old)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert read_rows(empty) == []


def _config(dir_, supply="18.0"):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "rack.yaml").write_text(f"cooling:\n  supply_temp_c: {supply}\n")
    (dir_ / "gpu_specs.yaml").write_text("node:\n  gpus_per_node: 8\n")
    return dir_


def test_sweep_records_the_config_it_ran_with(tmp_path):
    cfg = _config(tmp_path / "cfg")
    dest = record_config(cfg, tmp_path / "results")
    assert dest == tmp_path / "results" / "configs"
    for name in ("rack.yaml", "gpu_specs.yaml"):
        assert (dest / name).read_bytes() == (cfg / name).read_bytes()
    assert record_config(cfg, tmp_path / "results") == dest  # a resume with the same config


def test_resume_refuses_a_different_config(tmp_path):
    record_config(_config(tmp_path / "cool"), tmp_path / "results")
    with pytest.raises(ValueError, match="another config"):
        record_config(_config(tmp_path / "warm", supply="24.5"), tmp_path / "results")


def test_record_config_needs_both_files(tmp_path):
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "rack.yaml").write_text("x: 1\n")
    with pytest.raises(ValueError, match="gpu_specs.yaml"):
        record_config(tmp_path / "cfg", tmp_path / "results")
    assert not (tmp_path / "results").exists()  # nothing half-copied
