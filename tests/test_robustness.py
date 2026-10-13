"""Robustness tests: input validation and helpful errors."""

import sys

sys.path.insert(0, "src")

import pytest

from pvhostcap.feeders import build_lv_radial, build_mv_radial
from pvhostcap.mitigations import compare_strategies
from pvhostcap.powerflow import check_violations, run_case
from pvhostcap.report import summarize_comparison
from pvhostcap.sweep import (
    locational_hosting_capacity,
    system_hosting_capacity,
)


def _small_lv():
    return build_lv_radial(n_buses=3, seg_km=0.15, load_kw=3.0)


# --- feeder input validation ---------------------------------------------

def test_feeder_rejects_bad_n_buses():
    with pytest.raises(ValueError, match="n_buses"):
        build_lv_radial(n_buses=0)
    with pytest.raises(ValueError, match="n_buses"):
        build_mv_radial(n_buses=-2)


def test_feeder_rejects_non_positive_geometry():
    with pytest.raises(ValueError, match="seg_km"):
        build_lv_radial(seg_km=0.0)
    with pytest.raises(ValueError, match="load_kw"):
        build_lv_radial(load_kw=-1.0)


def test_feeder_rejects_bad_pf():
    with pytest.raises(ValueError, match="pf"):
        build_lv_radial(pf=0.0)
    with pytest.raises(ValueError, match="pf"):
        build_lv_radial(pf=1.5)


# --- helpful errors instead of bare KeyError ------------------------------

def test_compare_strategies_unknown_name():
    net, load_buses = _small_lv()
    with pytest.raises(ValueError, match="unknown mitigation strategies"):
        compare_strategies(net, load_buses, strategies=("none", "magic"))


def test_check_violations_unknown_kind():
    net, _ = _small_lv()
    assert run_case(net)
    with pytest.raises(ValueError, match="unknown violation kinds"):
        check_violations(net, kinds=("overvoltage", "magic"))


def test_empty_load_buses_rejected():
    net, _ = _small_lv()
    with pytest.raises(ValueError, match="load_buses"):
        system_hosting_capacity(net, [])
    with pytest.raises(ValueError, match="load_buses"):
        locational_hosting_capacity(net, [])


# --- summarize_comparison without a "none" baseline ------------------------

def test_summarize_comparison_custom_baseline():
    net, load_buses = _small_lv()
    comp = compare_strategies(net, load_buses,
                              strategies=("voltvar", "oltc"), tol_kw=0.5)
    text = summarize_comparison(comp)
    assert "voltvar" in text and "oltc" in text
    # falls back to the first strategy as baseline, no crash
    assert "baseline: voltvar" in text


def test_summarize_comparison_explicit_baseline():
    net, load_buses = _small_lv()
    comp = compare_strategies(net, load_buses,
                              strategies=("voltvar", "oltc"), tol_kw=0.5)
    text = summarize_comparison(comp, baseline="oltc")
    assert "baseline: oltc" in text


# --- connection_check: the DSO's everyday question ------------------------

def test_connection_check_by_name():
    from pvhostcap.sweep import connection_check

    net, load_buses = _small_lv()
    approved, violations = connection_check(net, "LV1", 5.0)
    assert approved and violations == []
    # far-end bus is weak: 60 kW must fail on the 3-bus feeder
    approved, violations = connection_check(net, "LV3", 60.0)
    assert not approved and len(violations) > 0


def test_connection_check_by_index_matches_name():
    from pvhostcap.sweep import connection_check

    net, load_buses = _small_lv()
    by_name = connection_check(net, "LV2", 10.0)
    by_idx = connection_check(net, load_buses[1], 10.0)
    assert by_name[0] == by_idx[0]
    assert [str(v) for v in by_name[1]] == [str(v) for v in by_idx[1]]


def test_resolve_bus_unknown_name():
    from pvhostcap.sweep import resolve_bus

    net, _ = _small_lv()
    with pytest.raises(ValueError, match="no bus named"):
        resolve_bus(net, "LV99")
