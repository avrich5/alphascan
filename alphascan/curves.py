"""Forward-move curves: forward matrix, market drift, median excess,
percentile-bootstrap confidence interval of a peak statistic.

Gate rules: trust only the excess over drift and the MEDIAN of the
event; CI by percentile bootstrap over events; entry = close of the
event bar.
"""
import numpy as np

DEFAULT_NBOOT = 300
CI = (5, 95)


def fwd_matrix(close, kmax, notional=10_000.0):
    """[n, kmax] float32: price move in quote currency on `notional`
    after k = 1..kmax bars, entry at close of bar i."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    fwd = np.full((n, kmax), np.nan, dtype=np.float32)
    for k in range(1, kmax + 1):
        fwd[:n - k, k - 1] = ((close[k:] / close[:-k] - 1) * notional).astype(np.float32)
    return fwd


def drift_median(fwd, rng, n_base=20000, lo=60):
    """Median curve of random bars: the market drift baseline."""
    hi = fwd.shape[0] - fwd.shape[1]
    idx = rng.choice(np.arange(lo, hi), size=min(n_base, hi - lo), replace=False)
    return np.nanmedian(fwd[idx], axis=0)


def event_indices(mask, kmax, lo=60):
    """Bars where the event fires and the full forward window exists."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask[:len(mask) - kmax])[0]
    return idx[idx > lo]


def median_excess(F, base_med):
    """Median curve of event rows minus drift."""
    return np.nanmedian(F, axis=0) - base_med


def bootstrap_peak_ci(F, base_med, stat, nboot=DEFAULT_NBOOT, rng=None, ci=CI):
    """CI of a peak statistic. stat(median_excess_curve) -> float.
    Raw peak: stat = lambda m: float(np.nanmax(m)).
    Returns (point, lo, hi). Deterministic for a given rng seed."""
    rng = rng if rng is not None else np.random.default_rng(0)
    point = stat(median_excess(F, base_med))
    boots = np.empty(nboot)
    for b in range(nboot):
        samp = rng.integers(0, len(F), len(F))
        boots[b] = stat(median_excess(F[samp], base_med))
    lo, hi = np.percentile(boots, list(ci))
    return float(point), float(lo), float(hi)
