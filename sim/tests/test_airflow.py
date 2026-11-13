from dataclasses import replace

import numpy as np
import pytest

from physics.airflow import heat_matrix, recirculation_matrix


def with_recirc(spec, **kw):
    return replace(spec, recirculation=replace(spec.recirculation, **kw))


def test_containment_means_supply_air_everywhere(spec):
    spec = with_recirc(spec, max_fraction=0.0)
    assert not recirculation_matrix(spec).any()
    assert not heat_matrix(spec).any()


def test_recirculated_share_is_bounded_by_max_fraction(spec):
    a = recirculation_matrix(spec)
    assert a.min() >= 0
    assert a.sum(axis=1).max() == pytest.approx(spec.recirculation.max_fraction)


def test_energy_balance_at_every_inlet(spec):
    """k*T_in_j = (k - sum_i A_ij k) T_sup + sum_i A_ij k T_out_i  for every inlet j."""
    a = recirculation_matrix(spec)
    d = heat_matrix(spec, a)
    k = spec.airflow_kg_s * spec.cp_j_kgk
    rng = np.random.default_rng(0)
    p = rng.uniform(1500, 7100, len(spec.nodes))
    t_sup = spec.supply_temp_c
    t_in = t_sup + d @ p
    t_out = t_in + p / k
    lhs = k * t_in
    rhs = (k - k * a.sum(axis=0)) * t_sup + k * (a.T @ t_out)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_infeasible_recirculation_is_rejected(spec):
    with pytest.raises(ValueError, match="recirculation infeasible"):
        recirculation_matrix(with_recirc(spec, max_fraction=5.0))


def test_top_slots_and_end_racks_run_hotter(spec):
    d = heat_matrix(spec)
    t_in = d @ np.full(len(spec.nodes), 5000.0)  # uniform load, rise above supply
    grid = t_in.reshape(spec.n_racks, spec.slots_per_rack)
    assert (grid[:, -1] > grid[:, 0]).all()  # top > bottom in every rack
    middle = spec.n_racks // 2
    assert grid[0].mean() > grid[middle].mean()
    assert grid[-1].mean() > grid[middle].mean()
