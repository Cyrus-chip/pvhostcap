"""Hosting-capacity sweeps: system-wide and per-bus (locational).

Method: deterministic bisection on PV size. Feasibility (no violation) is
monotone decreasing in PV size — more PV can only raise voltages and push
more reverse power through the same branches — so bisection on a single
scalar per case is exact up to the tolerance.
"""

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import pandapower as pp

from pvhostcap.powerflow import binding_violation, check_violations, run_case

__all__ = [
    "HostingCapacityResult",
    "evaluate",
    "system_hosting_capacity",
    "locational_hosting_capacity",
]


@dataclass
class HostingCapacityResult:
    """Result of one hosting-capacity search."""

    hc_kw_per_bus: float          # bisection result (per-bus kW)
    hc_kw_total: float            # hc_kw_per_bus * n_pv_buses
    n_pv_buses: int
    binding: object = None        # Violation that binds at the HC point
    evaluations: int = 0         # load-flow evaluations used
    converged: bool = True

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        b = f", binding: {self.binding}" if self.binding else ", no binding violation"
        return (f"HC = {self.hc_kw_total:.2f} kW total "
                f"({self.hc_kw_per_bus:.2f} kW x {self.n_pv_buses} buses){b}")


def _add_pv(net, pv_kw: dict) -> list:
    """Attach one sgen per bus in pv_kw (kW, positive = generation)."""
    sgens = []
    for bus, kw in pv_kw.items():
        if kw > 0:
            sgens.append(pp.create_sgen(net, bus=bus, p_mw=kw / 1000.0,
                                        q_mvar=0.0, name=f"PV bus {bus}"))
    return sgens


def _bus_index(net_template):
    """Return the bus index, with a useful error for malformed networks."""
    try:
        return net_template.bus.index
    except AttributeError as exc:
        raise TypeError(
            "net_template must be a pandapower network with a bus table"
        ) from exc


def _finite_real(value, name: str) -> float:
    """Coerce a real scalar to float and reject non-finite values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _positive_finite(value, name: str) -> float:
    value = _finite_real(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _validate_load_buses(net_template, load_buses) -> list:
    """Validate and materialize the buses participating in a sweep."""
    if isinstance(load_buses, (str, bytes)):
        raise TypeError("load_buses must be an iterable of bus indices")
    try:
        buses = list(load_buses)
    except TypeError as exc:
        raise TypeError("load_buses must be an iterable of bus indices") from exc
    if not buses:
        raise ValueError("load_buses must contain at least one bus index")
    try:
        has_duplicates = len(set(buses)) != len(buses)
    except TypeError as exc:
        raise TypeError("load_buses must contain hashable bus indices") from exc
    if has_duplicates:
        raise ValueError("load_buses must not contain duplicate bus indices")

    bus_index = _bus_index(net_template)
    missing = [bus for bus in buses if bus not in bus_index]
    if missing:
        raise ValueError(
            f"load_buses contains bus index(es) not present in net_template: {missing}"
        )
    return buses


def _validate_pv_allocation(net_template, pv_kw) -> None:
    """Reject allocations that would otherwise be silently ignored."""
    if not isinstance(pv_kw, Mapping):
        raise TypeError("pv_kw must be a mapping of bus index to PV kW")
    bus_index = _bus_index(net_template)
    for bus, kw in pv_kw.items():
        if bus not in bus_index:
            raise ValueError(
                f"pv_kw contains bus index not present in net_template: {bus!r}"
            )
        kw = _finite_real(kw, f"pv_kw[{bus!r}]")
        if kw < 0.0:
            raise ValueError(f"pv_kw[{bus!r}] must be non-negative")


def evaluate(net_template, pv_kw: dict, mitigation=None, **limits) -> list:
    """Evaluate one PV allocation against the limits.

    ``pv_kw`` maps bus index -> PV kW. ``mitigation`` is an optional
    callable ``(net, sgen_indices) -> None`` applied after the PV is added
    and before the load flow (see :mod:`pvhostcap.mitigations`).
    Returns the list of violations (empty = feasible). A non-converged
    load flow counts as infeasible and is reported as such.
    """
    _validate_pv_allocation(net_template, pv_kw)
    if mitigation is not None and not callable(mitigation):
        raise TypeError("mitigation must be callable or None")
    net = copy.deepcopy(net_template)
    sgens = _add_pv(net, pv_kw)
    if mitigation is not None:
        mitigation(net, sgens)
    if not run_case(net):
        from pvhostcap.powerflow import Violation
        return [Violation("non_converged", "load flow", 1.0, 1.0)]
    return check_violations(net, **limits)


def _bisect(feasible, hi_start: float = 1.0, tol: float = 0.05,
            hi_cap: float = 1e6) -> tuple:
    """Bisection on a monotone feasible(size)->bool predicate.

    Returns (size, evaluations). ``feasible(0)`` is assumed True; if it is
    not, (0.0, n) is returned.
    """
    hi_start = _positive_finite(hi_start, "hi_start")
    tol = _positive_finite(tol, "tol")
    hi_cap = _positive_finite(hi_cap, "hi_cap")
    if hi_start > hi_cap:
        raise ValueError("hi_start must not exceed hi_cap")

    n = 0
    if not feasible(0.0):
        return 0.0, 1
    n += 1
    hi = hi_start
    while True:
        n += 1
        if not feasible(hi):
            break
        hi *= 2.0
        if hi > hi_cap:
            return hi_cap, n
    lo = hi / 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        n += 1
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo, n


def system_hosting_capacity(
    net_template,
    load_buses: list,
    mitigation=None,
    tol_kw: float = 0.05,
    **limits,
) -> HostingCapacityResult:
    """System HC: max *uniform* per-bus PV before the first violation.

    Every bus in ``load_buses`` gets the same PV size; bisection finds the
    largest size with zero violations. Returns the HC point plus the
    binding violation evaluated just above it.
    """
    buses = _validate_load_buses(net_template, load_buses)
    tol_kw = _positive_finite(tol_kw, "tol_kw")

    def feasible(kw: float) -> bool:
        return not evaluate(net_template, {b: kw for b in buses},
                            mitigation=mitigation, **limits)

    hc_per_bus, n_eval = _bisect(feasible, tol=tol_kw)
    # binding violation just above the HC point
    above = evaluate(net_template, {b: hc_per_bus + max(tol_kw, 0.01)
                                    for b in buses},
                     mitigation=mitigation, **limits)
    return HostingCapacityResult(
        hc_kw_per_bus=hc_per_bus,
        hc_kw_total=hc_per_bus * len(buses),
        n_pv_buses=len(buses),
        binding=binding_violation(above),
        evaluations=n_eval + 1,
        converged=True,
    )


def locational_hosting_capacity(
    net_template,
    load_buses: list,
    mitigation=None,
    tol_kw: float = 0.05,
    **limits,
) -> dict:
    """Locational HC: max PV each bus can host *on its own*.

    For every bus in ``load_buses``, bisection finds the largest PV size at
    that bus alone (all other buses at zero PV) with zero violations.
    Returns ``{bus: HostingCapacityResult}``. This is the "weak-bus map" of
    the feeder — the quantity a DSO actually needs for connection offers.
    """
    buses = _validate_load_buses(net_template, load_buses)
    tol_kw = _positive_finite(tol_kw, "tol_kw")
    out = {}
    for b in buses:
        def feasible(kw: float, _b=b) -> bool:
            return not evaluate(net_template, {_b: kw},
                                mitigation=mitigation, **limits)

        hc, n_eval = _bisect(feasible, tol=tol_kw)
        above = evaluate(net_template, {b: hc + max(tol_kw, 0.01)},
                         mitigation=mitigation, **limits)
        out[b] = HostingCapacityResult(
            hc_kw_per_bus=hc,
            hc_kw_total=hc,
            n_pv_buses=1,
            binding=binding_violation(above),
            evaluations=n_eval + 1,
            converged=True,
        )
    return out
