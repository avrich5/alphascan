"""Selection transfer test and breakeven cost.

Choosing the best of many combinations in-sample guarantees a good
in-sample number. The only evidence of a signal is transfer: the
combinations selected on the training windows must beat random
combinations of the same size on windows they have not seen.
"""
import numpy as np
import pandas as pd


def profit_factor(gross, fee_per_side=0.0):
    """PF of per-trade gross results (per 10 000 notional) after cost."""
    net = np.asarray(gross, dtype=float) - 2 * fee_per_side
    loss = -net[net < 0].sum()
    return np.inf if loss == 0 else net[net > 0].sum() / loss


def breakeven_fee(gross, target_pf=1.0, lo=-50.0, hi=100.0, iters=60):
    """Cost per side (per 10 000 notional) at which PF falls to target_pf.
    PF decreases monotonically in cost, so bisection is exact."""
    if profit_factor(gross, lo) < target_pf:
        return np.nan
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if profit_factor(gross, mid) >= target_pf else (lo, mid)
    return lo


def transfer_test(table, train, test, metric="pf", select=lambda r: r["pf"] > 1.5,
                  min_trades=30, n_random=1000, seed=0):
    """table: one row per (combo, window) with columns combo, window, pf,
    trades, pnl. train/test: lists of window labels.
    Selection uses train windows only; returns medians of `metric` on test
    windows for the selected set and for random sets of the same size,
    plus p = share of random sets at least as good."""
    tr = table[table.window.isin(train)].groupby("combo").agg(
        pf=("pf", "median"), trades=("trades", "sum"))
    pool = tr[tr.trades >= min_trades]
    chosen = pool[pool.apply(select, axis=1)].index
    rng = np.random.default_rng(seed)
    out = {"pool": len(pool), "selected": len(chosen)}
    for w in test:
        te = table[table.window == w].set_index("combo")[metric]
        te = te.reindex(pool.index)
        sel = float(te.loc[chosen].median()) if len(chosen) else np.nan
        k = max(len(chosen), 1)
        rand = np.array([te.iloc[rng.choice(len(te), k, replace=False)].median()
                         for _ in range(n_random)])
        out[w] = {"selected_median": sel, "random_median": float(np.median(rand)),
                  "p": float((rand >= sel).mean()) if len(chosen) else np.nan}
    return out
