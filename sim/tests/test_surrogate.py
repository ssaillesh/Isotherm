from dataclasses import replace

from physics import Cluster, Workload
from surrogate import FEATURE_NAMES, RolloutPredictor


def full(cluster):
    return Workload("w", cluster.spec.gpus_per_node, 1.0)


def test_rollout_does_not_mutate_cluster(cluster):
    before = cluster.temp_c.copy()
    RolloutPredictor().score(cluster, full(cluster), cluster.node_names, 300)
    assert (cluster.temp_c == before).all()
    assert cluster.allocations == {}


def test_rollout_prefers_bottom_slot_over_top(cluster):
    by_name = {n.name: n for n in cluster.spec.nodes}
    middle = cluster.spec.n_racks // 2
    bottom, top = (
        n.name
        for n in cluster.spec.nodes
        if n.rack == middle and n.slot in (0, cluster.spec.slots_per_rack - 1)
    )
    assert by_name[bottom].slot == 0
    b, t = RolloutPredictor().score(cluster, full(cluster), [bottom, top], 600)
    assert b.inlet_headroom_c > t.inlet_headroom_c


def test_headroom_is_binding_margin(cluster):
    (s,) = RolloutPredictor().score(cluster, full(cluster), cluster.node_names[:1], 300)
    assert s.headroom_c == min(s.inlet_headroom_c, s.gpu_headroom_c)
    assert s.feasible


def test_inlet_limit_makes_placement_infeasible(spec):
    tight = Cluster(replace(spec, inlet_limit_c=spec.supply_temp_c + 0.5))
    (s,) = RolloutPredictor().score(tight, full(tight), tight.node_names[:1], 600)
    assert not s.feasible and s.inlet_headroom_c < 0


def test_rollout_marks_full_and_unknown_nodes_infeasible(cluster):
    node = cluster.node_names[0]
    cluster.bind(Workload("fill", cluster.spec.gpus_per_node, 0.5), node)
    a, b = RolloutPredictor().score(cluster, Workload("w", 1, 1.0), [node, "nope"], 60)
    assert not a.feasible and not b.feasible


def test_rollout_returns_features(cluster):
    (s,) = RolloutPredictor().score(cluster, Workload("w", 2, 0.5), cluster.node_names[:1], 60)
    assert len(s.features) == len(FEATURE_NAMES)
