"""System hosting-capacity search tests."""

import sys

import pytest

sys.path.insert(0, "src")

from pvhostcap.feeders import build_lv_radial, build_mv_radial
from pvhostcap.sweep import evaluate, locational_hosting_capacity, system_hosting_capacity


def _small_lv():
    return build_lv_radial(n_buses=3, seg_km=0.15, load_kw=3.0)


def test_system_hc_brackets_true_limit():
    net, load_buses = _small_lv()
    res = system_hosting_capacity(net, load_buses, tol_kw=0.1)
    assert 0.0 < res.hc_kw_per_bus < 60.0
    assert res.n_pv_buses == 3
    assert abs(res.hc_kw_total - 3 * res.hc_kw_per_bus) < 1e-9
    # just above HC -> violation; just below -> clean
    assert evaluate(net, {b: res.hc_kw_per_bus + 1.0 for b in load_buses})
    assert not evaluate(net, {b: res.hc_kw_per_bus - 1.0 for b in load_buses})


def test_system_hc_is_deterministic():
    net, load_buses = _small_lv()
    r1 = system_hosting_capacity(net, load_buses, tol_kw=0.1)
    r2 = system_hosting_capacity(net, load_buses, tol_kw=0.1)
    assert r1.hc_kw_per_bus == r2.hc_kw_per_bus
    assert r1.evaluations == r2.evaluations


def test_lv_feeder_is_voltage_bound():
    net, load_buses = _small_lv()
    res = system_hosting_capacity(net, load_buses, tol_kw=0.1)
    assert res.binding is not None
    assert res.binding.kind == "overvoltage"
    assert "LV3" in res.binding.element  # far-end bus


def test_mv_feeder_is_thermal_bound():
    net, load_buses = build_mv_radial()
    res = system_hosting_capacity(net, load_buses, tol_kw=5.0)
    assert res.binding is not None
    assert res.binding.kind == "line_overload"
    assert "MV0-MV1" in res.binding.element  # head-end line


def test_tighter_vmax_lowers_hc():
    net, load_buses = _small_lv()
    loose = system_hosting_capacity(net, load_buses, tol_kw=0.1, vmax=1.10)
    tight = system_hosting_capacity(net, load_buses, tol_kw=0.1, vmax=1.03)
    assert tight.hc_kw_per_bus < loose.hc_kw_per_bus


def test_system_hc_rejects_non_positive_tolerance():
    net, load_buses = _small_lv()
    with pytest.raises(ValueError, match="tol_kw.*greater than zero"):
        system_hosting_capacity(net, load_buses, tol_kw=0.0)


def test_system_hc_rejects_duplicate_or_unknown_buses():
    net, load_buses = _small_lv()
    with pytest.raises(ValueError, match="duplicate"):
        system_hosting_capacity(net, [load_buses[0], load_buses[0]], tol_kw=0.1)
    with pytest.raises(ValueError, match="not present"):
        system_hosting_capacity(net, [load_buses[0], 999], tol_kw=0.1)


def test_locational_hc_requires_at_least_one_bus():
    net, _ = _small_lv()
    with pytest.raises(ValueError, match="at least one"):
        locational_hosting_capacity(net, [], tol_kw=0.1)


@pytest.mark.parametrize(
    ("allocation", "message"),
    [("negative", "non-negative"), ("unknown", "not present")],
)
def test_evaluate_rejects_invalid_pv_allocations(allocation, message):
    net, load_buses = _small_lv()
    bus = load_buses[0]
    allocation = {bus: -1.0} if allocation == "negative" else {999: 1.0}
    with pytest.raises(ValueError, match=message):
        evaluate(net, allocation)
