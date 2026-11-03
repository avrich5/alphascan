"""OHLCV input: load a local file or fetch from a public exchange API.

No data ships with the package. Every series passes a quality gate
and is fingerprinted (sha256), so a verdict is tied to the exact data.
"""
import hashlib
import numpy as np
import pandas as pd

COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def load_ohlcv(path):
    """CSV or parquet with columns ts, open, high, low, close, volume."""
    d = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in d.columns]
    if missing:
        raise ValueError("missing columns: %s" % missing)
    d = d[COLUMNS].copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    return d.sort_values("ts").reset_index(drop=True)


def fetch_ohlcv(exchange, symbol, timeframe, since, until=None, limit=1000):
    """Public candles via ccxt (pip install ccxt). since/until: ISO dates."""
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    t = ex.parse8601(pd.Timestamp(since, tz="UTC").isoformat())
    end = ex.parse8601(pd.Timestamp(until, tz="UTC").isoformat()) if until else ex.milliseconds()
    step = ex.parse_timeframe(timeframe) * 1000
    rows = []
    while t < end:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=t, limit=limit)
        if not batch:
            break
        rows += [r for r in batch if r[0] < end]
        t = batch[-1][0] + step
    d = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("ts")
    d["ts"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    return d.sort_values("ts").reset_index(drop=True)


def quality(d, timeframe):
    """Quality gate: gaps, duplicates, broken bars. Returns a dict;
    `ok` is False if anything that would bias the gate is found."""
    step = pd.Timedelta(timeframe)
    dt = d["ts"].diff().dropna()
    bad_bar = ((d["high"] < d[["open", "close"]].max(axis=1)) |
               (d["low"] > d[["open", "close"]].min(axis=1)) |
               (d[["open", "high", "low", "close"]] <= 0).any(axis=1))
    rep = {
        "bars": int(len(d)),
        "first": str(d["ts"].iloc[0]) if len(d) else None,
        "last": str(d["ts"].iloc[-1]) if len(d) else None,
        "duplicates": int(d["ts"].duplicated().sum()),
        "gaps": int((dt > step).sum()),
        "missing_bars": int(((dt / step).round() - 1).clip(lower=0).sum()),
        "broken_bars": int(bad_bar.sum()),
        "sha256": fingerprint(d),
    }
    rep["ok"] = rep["duplicates"] == 0 and rep["broken_bars"] == 0
    return rep


def fingerprint(d):
    """sha256 of the canonical OHLCV bytes."""
    a = np.column_stack([d["ts"].astype("int64").to_numpy(),
                         d[["open", "high", "low", "close", "volume"]].to_numpy(float).view("int64")])
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
