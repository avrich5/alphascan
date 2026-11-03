"""Forward-curve gate: entry candidates x forward curves -> verdict table.

The statistic is the peak of the median excess curve over horizons
1..kmax. Taking a maximum over horizons is biased upward, and events of
one condition cluster in time, so the peak of a meaningless condition is
positive too. The reference is therefore a placebo: the same condition
mask shifted circularly by random offsets (same count, same clustering,
no link to the future), scored by the same statistic.

    edge    = peak - median(placebo peaks)
    edge_lo = peak - 95th percentile(placebo peaks)

A candidate enters further work only if edge_lo exceeds the round-trip
cost.
"""
import numpy as np
import pandas as pd

from . import registry as CAND
from . import conditions as COND
from . import curves as CV
from .funding import fund_adjusted_peak

DEFAULT_SEED = 20260830


def run_gate(d, round_trip_cost, kmax=48, fund_per_period=None, bars_per_funding=16,
             nnull=200, seed=DEFAULT_SEED, side="long", min_events=100, extra=None,
             conditions=None, include_registry=True):
    """d: OHLCV frame. round_trip_cost: fee + slippage for entry and exit,
    in quote currency per 10 000 notional (taker 0.06% per side = 12.0).
    extra: optional dict name -> boolean mask, gated alongside the registry.
    conditions: optional dict name -> expression (see alphascan.conditions).
    include_registry: gate the built-in registry too.
    Returns one row per candidate; `enters` is the verdict."""
    rng = np.random.default_rng(seed)
    base_stat = ((lambda m: float(np.nanmax(m))) if fund_per_period is None
                 else (lambda m: fund_adjusted_peak(m, bars_per_funding, fund_per_period)[0]))
    stat = base_stat if side == "long" else (lambda m: base_stat(-m))
    fwd = CV.fwd_matrix(d["close"].to_numpy(float), kmax)
    base = CV.drift_median(fwd, rng)
    n_bars = len(fwd)
    extra = dict(extra or {})
    extra.update(COND.evaluate_many(conditions or {}, d))
    masks = dict(CAND.candidates(d)) if include_registry else {}
    masks.update(extra)
    rows = []
    for name, mask in masks.items():
        if side == "short" and name not in extra and not CAND.is_short_candidate(name):
            continue
        idx = CV.event_indices(mask, kmax)
        if len(idx) < min_events:
            rows.append(dict(candidate=name, side=side, n=len(idx), verdict="too_rare"))
            continue
        point = stat(CV.median_excess(fwd[idx], base))
        null = np.empty(nnull)
        for b in range(nnull):
            sh = rng.integers(kmax + 100, n_bars - kmax - 100)
            nidx = CV.event_indices(np.roll(mask, sh), kmax)
            null[b] = stat(CV.median_excess(fwd[nidx], base))
        q50, q95 = np.percentile(null, [50, 95])
        curve = CV.median_excess(fwd[idx], base)
        curve = -curve if side == "short" else curve
        bar = (int(np.nanargmax(curve)) + 1 if fund_per_period is None
               else fund_adjusted_peak(curve, bars_per_funding, fund_per_period)[1])
        edge_lo = point - q95
        rows.append(dict(candidate=name, side=side, n=len(idx), peak=round(point, 2), bar=bar,
                         placebo_median=round(q50, 2), placebo_q95=round(q95, 2),
                         edge=round(point - q50, 2), edge_lo=round(edge_lo, 2),
                         breakeven_fee_per_side=round(edge_lo / 2, 2),
                         verdict="enters" if edge_lo > round_trip_cost else "below_cost"))
    out = pd.DataFrame(rows)
    out.attrs.update(registry=CAND.REGISTRY_VERSION, seed=seed, kmax=kmax,
                     round_trip_cost=round_trip_cost, nnull=nnull)
    return out
