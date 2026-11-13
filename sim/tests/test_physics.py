import numpy as np
import pytest

from physics import PlacementError, Workload
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
        cluster.node_power_w,
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
    assert cluster.temp_c[cluster.node_gpus(hot)].min() > 60
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
