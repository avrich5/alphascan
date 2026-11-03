"""Funding-adjusted optimum of a forward-move curve.

Holding a perpetual contract pays funding every `bars_per_funding`
bars. The value of exiting at bar k is
    curve[k] - ceil(k / bars_per_funding) * fund_per_period.
The optimum is the maximum over bars; it always exists because the
penalty grows in steps.

Defaults: 30m bars, funding every 8 hours (16 bars), 0.01% of a
10 000 notional = 1.0 per period. Conservative for longs.
"""
import numpy as np

__all__ = ["fund_adjusted_peak", "funding_schedule"]


def funding_schedule(n_bars, bars_per_funding=16, fund_per_period=1.0):
    """Funding penalty per holding bar (bar 1 = index 0)."""
    k = np.arange(1, n_bars + 1)
    return np.ceil(k / float(bars_per_funding)) * float(fund_per_period)


def fund_adjusted_peak(median_curve, bars_per_funding=16, fund_per_period=1.0):
    """(peak, bar, adjusted_curve); bar is the 1-based optimal exit."""
    mc = np.asarray(median_curve, dtype=float)
    adj = mc - funding_schedule(len(mc), bars_per_funding, fund_per_period)
    bar = int(np.nanargmax(adj)) + 1
    return float(adj[bar - 1]), bar, adj
