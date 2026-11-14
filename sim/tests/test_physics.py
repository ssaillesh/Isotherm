from dataclasses import replace

import numpy as np
import pytest

from physics import Cluster, PlacementError, Workload
from physics.cluster import MAX_SUBSTEP_S
from physics.thermal import gpu_power, step_temperature, throttle


def test_rc_step_converges_to_steady_state():
    t = step_temperature(
        temp_c=np.array([30.0]),
        power_w=np.array([500.0]),
        inlet_c=np.array([25.0]),
        r_c_per_w=np.array([0.1]),
        c_j_per_c=np.array([500.0]),
        dt_s=1e6,
    )
    assert t[0] == pytest.approx(25.0 + 500.0 * 0.1)


def test_gpu_power_idle_and_full():
    tdp = np.array([700.0, 700.0])
    np.testing.assert_allclose(gpu_power(np.array([0.0, 1.0]), tdp, 0.1), [70.0, 700.0])


def test_throttle_caps_steady_state_at_limit():
    power, throttled = throttle(
        demand_w=np.array([700.0, 100.0]),
        inlet_c=np.array([40.0, 40.0]),
        r_c_per_w=np.array([0.075, 0.075]),
        throttle_c=np.array([87.0, 87.0]),
    )
    assert throttled.tolist() == [True, False]
    assert 40.0 + power[0] * 0.075 == pytest.approx(87.0)
    assert power[1] == 100.0


def test_idle_row_sits_just_above_supply(cluster, spec):
    assert cluster.inlet_c.min() >= spec.supply_temp_c
    assert cluster.inlet_c.max() < spec.inlet_limit_c
    np.testing.assert_allclose(
        cluster.node_power_w[0],
        spec.base_w
        + spec.gpus_per_node * spec.idle_gpu_fraction * cluster.tdp_w[0 :: spec.gpus_per_node],
    )


def test_exhaust_is_inlet_plus_power_over_airflow(cluster, spec):
    k = spec.airflow_kg_s * spec.cp_j_kgk
    np.testing.assert_allclose(cluster.exhaust_c, cluster.inlet_c + cluster.node_power_w / k)


def test_bind_and_unbind(cluster):
    node = cluster.node_names[0]
    cluster.bind(Workload("w1", 3, 0.9), node)
    assert len(cluster.free_gpus(0)) == cluster.spec.gpus_per_node - 3
    with pytest.raises(PlacementError):
        cluster.bind(Workload("w1", 1, 0.5), node)  # duplicate id
    assert cluster.unbind("w1")
    assert not cluster.unbind("w1")
    assert len(cluster.free_gpus(0)) == cluster.spec.gpus_per_node


def test_bind_rejects_oversubscription(cluster):
    with pytest.raises(PlacementError, match="free GPUs"):
        cluster.bind(Workload("big", cluster.spec.gpus_per_node + 1, 1.0), cluster.node_names[0])


def test_load_heats_own_gpus_and_other_inlets(cluster):
    hot = 0
    before = cluster.inlet_c.copy()
    cluster.bind(Workload("w", cluster.spec.gpus_per_node, 1.0), cluster.node_names[hot])
    cluster.step(1200)
    assert cluster.temp_c[0, cluster.node_gpus(hot)].min() > 60
    assert (cluster.inlet_c >= before - 1e-12).all()
    assert (cluster.inlet_c > before).sum() > 1  # recirculation couples servers


def test_workloads_expire_after_duration(cluster):
    cluster.bind(Workload("short", 2, 1.0, duration_s=60), cluster.node_names[0])
    assert cluster.step(30) == []
    assert cluster.step(40) == ["short"]
    assert "short" not in cluster.allocations


def test_copy_is_independent(cluster):
    clone = cluster.copy()
    clone.bind(Workload("w", 1, 1.0), cluster.node_names[0])
    clone.step(100)
    assert "w" not in cluster.allocations
    assert cluster.time_s == 0.0


def test_reset_restores_initial_state(cluster):
    initial = cluster.temp_c.copy()
    cluster.bind(Workload("w", 8, 1.0), cluster.node_names[0])
    cluster.step(600)
    cluster.reset()
    assert cluster.allocations == {}
    np.testing.assert_allclose(cluster.temp_c, initial)


def test_state_arrays_have_one_scenario_on_the_live_cluster(cluster):
    assert cluster.n_scenarios == 1
    assert cluster.temp_c.shape == (1, cluster.n_gpus)
    assert cluster.inlet_c.shape == (1, len(cluster.node_names))


def plain_step(cluster, temp, inlet, dt_s):
    """The substep written out longhand: nothing cached, nothing skipped, one scenario."""
    spec = cluster.spec
    remaining = dt_s
    while remaining > 1e-9:
        h = min(MAX_SUBSTEP_S, remaining)
        demand = gpu_power(cluster.util[0], cluster.tdp_w, spec.idle_gpu_fraction)
        power, _ = throttle(demand, inlet[cluster.gpu_node], cluster.r, cluster.throttle_c)
        node_power = spec.base_w + np.bincount(cluster.gpu_node, weights=power)
        inlet = spec.supply_temp_c + cluster.heat @ node_power
        temp = step_temperature(temp, power, inlet[cluster.gpu_node], cluster.r, cluster.c, h)
        remaining -= h
    return temp, inlet


@pytest.mark.parametrize("supply_c", [18.0, 41.0])  # 41 C makes the H100 racks throttle
def test_step_equals_plain_substeps_exactly(spec, supply_c):
    # Guards the shortcuts in Cluster.step (skipped power updates, cached decay, running
    # per-node totals): the result must be the same floats, not just close.
    cluster = Cluster(replace(spec, supply_temp_c=supply_c))
    temp, inlet = cluster.temp_c[0], cluster.inlet_c[0]
    names = cluster.node_names
    for k, dt in enumerate([10.0, 10.0, 7.0, 60.0, 10.0, 600.0, 3.0]):
        cluster.bind(Workload(f"w{k}", 4 + k % 5, 0.5 + 0.07 * k), names[(k * 11) % len(names)])
        if k == 4:
            cluster.unbind("w1")
        temp, inlet = plain_step(cluster, temp, inlet, dt)
        cluster.step(dt)
        assert np.array_equal(cluster.temp_c[0], temp)
        assert np.array_equal(cluster.inlet_c[0], inlet)
    assert bool(cluster.throttled.any()) == (supply_c > 40)


def test_expiry_is_by_end_time_and_reported_in_bind_order(cluster):
    a, b = cluster.node_names[:2]
    cluster.bind(Workload("late", 1, 1.0, duration_s=60), a)
    cluster.bind(Workload("early", 1, 1.0, duration_s=20), a)
    cluster.bind(Workload("forever", 1, 1.0), b)
    assert cluster.step(30) == ["early"]
    cluster.bind(Workload("early", 1, 1.0, duration_s=30), a)  # same id again, ends at 60
    assert cluster.step(30) == ["late", "early"]
    assert list(cluster.allocations) == ["forever"]
    assert cluster.step(600) == []


def test_unbinding_early_cancels_the_expiry(cluster):
    node = cluster.node_names[0]
    cluster.bind(Workload("w", 2, 1.0, duration_s=50), node)
    assert cluster.unbind("w")
    cluster.bind(Workload("w", 2, 1.0, duration_s=200), node)  # the old end time is stale
    assert cluster.step(100) == []
    assert "w" in cluster.allocations
    assert cluster.step(100) == ["w"]


def test_branch_steps_like_separate_copies(cluster):
    names = cluster.node_names
    cluster.bind(Workload("bg", 8, 0.9, duration_s=200), names[5])  # expires mid-horizon
    cluster.bind(Workload("bg2", 3, 0.6), names[17])
    cluster.step(120)
    trial = Workload("trial", 4, 0.8)
    picks = [names[0], names[17], names[39]]

    batch = cluster.branch(trial, picks)
    assert batch.n_scenarios == 3
    singles = []
    for name in picks:
        one = cluster.copy()
        one.bind(trial, name)
        singles.append(one)
    for _ in range(30):
        batch.step(10)
        for k, one in enumerate(singles):
            one.step(10)
            assert np.array_equal(batch.temp_c[k], one.temp_c[0])
            assert np.array_equal(batch.inlet_c[k], one.inlet_c[0])
    assert "bg" not in batch.allocations


def test_branch_leaves_the_live_cluster_alone(cluster):
    node = cluster.node_names[0]
    before = cluster.temp_c.copy()
    batch = cluster.branch(Workload("trial", 8, 1.0), [node, cluster.node_names[1]])
    batch.step(600)
    assert np.array_equal(cluster.temp_c, before)
    assert cluster.time_s == 0.0 and cluster.allocations == {}
    assert len(cluster.free_gpus(0)) == cluster.spec.gpus_per_node
    with pytest.raises(PlacementError):
        batch.bind(Workload("x", 1, 1.0), node)


def test_branch_rejects_a_node_that_cannot_take_the_trial(cluster):
    node = cluster.node_names[0]
    cluster.bind(Workload("fill", cluster.spec.gpus_per_node, 0.5), node)
    with pytest.raises(PlacementError):
        cluster.branch(Workload("trial", 1, 1.0), [node])
    assert not cluster.can_bind(Workload("trial", 1, 1.0), node)
    assert cluster.can_bind(Workload("trial", 1, 1.0), cluster.node_names[1])


def test_supply_shift_moves_every_temperature_by_the_shift(spec):
    # The stress scenarios rely on this: with nothing throttling, warmer supply air shifts
    # every inlet and GPU temperature by exactly the supply change (T_in = T_supply + D @ P,
    # and power does not depend on temperature below throttle).
    shift = 6.5
    cool = Cluster(spec)
    warm = Cluster(replace(spec, supply_temp_c=spec.supply_temp_c + shift))
    names = cool.node_names
    for k, dt in enumerate([10.0, 60.0, 7.0, 600.0]):
        for c in (cool, warm):
            c.bind(Workload(f"w{k}", 4 + k % 5, 0.6 + 0.1 * k), names[(k * 7) % len(names)])
            c.step(dt)
        assert not warm.throttled.any()
        np.testing.assert_allclose(warm.inlet_c - cool.inlet_c, shift, rtol=0, atol=1e-9)
        np.testing.assert_allclose(warm.temp_c - cool.temp_c, shift, rtol=0, atol=1e-9)
