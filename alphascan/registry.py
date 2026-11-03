"""Entry candidate registry: named boolean conditions on OHLCV bars.

Each entry: name -> boolean array over the bars of an OHLCV frame.
The registry only grows; the meaning of an existing name never
changes. Results carry REGISTRY_VERSION.
"""
import numpy as np
import pandas as pd

REGISTRY_VERSION = "1"

DESCRIPTIONS = {
    "vol_spike_up": "volume > 3x 50-bar median, green candle",
    "vol_spike_dn": "volume > 3x 50-bar median, red candle",
    "big_up_2std": "bar return above +2 sigma of last 50 bars",
    "big_dn_2std": "bar return below -2 sigma of last 50 bars",
    "hammer": "hammer candle",
    "shooting_star": "shooting star candle",
    "engulf_up": "bullish engulfing",
    "engulf_dn": "bearish engulfing",
    "break_hh20": "close above previous 20-bar high",
    "break_ll20": "close below previous 20-bar low",
    "dryup_then_up": "previous bar volume < half median, then green candle",
    "vol3_and_break": "volume spike and close above 20-bar high",
    "squeeze_break_up": "ATR in lowest 20% of 100 bars and 20-bar breakout",
    "vol_confirm_dn": "-2 sigma drop on volume > 2x median",
    "engulf_confirm": "bullish engulfing, next bar closes above its high",
    "hammer_confirm": "hammer, next bar closes above its high",
    "tod_00_06": "-2 sigma drop between 00 and 06 UTC",
    "tod_06_12": "-2 sigma drop between 06 and 12 UTC",
    "tod_12_18": "-2 sigma drop between 12 and 18 UTC",
    "tod_18_24": "-2 sigma drop between 18 and 24 UTC",
}


def _rolling_pctl(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=window).apply(
        lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1), raw=True).to_numpy()


def _atr14(h, l, c):
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(14).mean().to_numpy()


def candidates(d):
    """d: DataFrame with ts, open, high, low, close, volume."""
    o, h, l, c, v = (d[k].to_numpy(float) for k in
                     ("open", "high", "low", "close", "volume"))
    ret = np.zeros_like(c); ret[1:] = c[1:] / c[:-1] - 1
    body = np.abs(c - o); rng = np.maximum(h - l, 1e-12)
    vmed = pd.Series(v).rolling(50).median().to_numpy()
    rstd = pd.Series(ret).rolling(50).std().to_numpy()
    hh20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    ll20 = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    prev_body = np.roll(body, 1); prev_up = np.roll(c > o, 1)
    prev_high = np.roll(h, 1)
    big_dn_2std = ret < -2 * rstd
    hammer = ((np.minimum(o, c) - l) > 2 * body) & ((h - np.maximum(o, c)) < 0.3 * rng)
    engulf_up = (c > o) & (~prev_up) & (body > prev_body * 1.2)
    break_hh20 = c > hh20
    atr_pctl100 = _rolling_pctl(_atr14(h, l, c), 100)
    engulf_confirm = np.roll(engulf_up, 1) & (c > prev_high); engulf_confirm[0] = False
    hammer_confirm = np.roll(hammer, 1) & (c > prev_high); hammer_confirm[0] = False
    hour = pd.to_datetime(d["ts"]).dt.hour.to_numpy()
    return {
        "vol_spike_up": (v > 3 * vmed) & (c > o),
        "vol_spike_dn": (v > 3 * vmed) & (c < o),
        "big_up_2std": ret > 2 * rstd,
        "big_dn_2std": big_dn_2std,
        "hammer": hammer,
        "shooting_star": ((h - np.maximum(o, c)) > 2 * body) & ((np.minimum(o, c) - l) < 0.3 * rng),
        "engulf_up": engulf_up,
        "engulf_dn": (c < o) & prev_up & (body > prev_body * 1.2),
        "break_hh20": break_hh20,
        "break_ll20": c < ll20,
        "dryup_then_up": (np.roll(v, 1) < 0.5 * np.roll(vmed, 1)) & (c > o),
        "vol3_and_break": (v > 3 * vmed) & (c > hh20),
        "squeeze_break_up": (atr_pctl100 < 0.2) & break_hh20,
        "vol_confirm_dn": big_dn_2std & (v > 2 * vmed),
        "engulf_confirm": engulf_confirm,
        "hammer_confirm": hammer_confirm,
        "tod_00_06": big_dn_2std & (hour >= 0) & (hour < 6),
        "tod_06_12": big_dn_2std & (hour >= 6) & (hour < 12),
        "tod_12_18": big_dn_2std & (hour >= 12) & (hour < 18),
        "tod_18_24": big_dn_2std & (hour >= 18) & (hour < 24),
    }


def is_short_candidate(name):
    """Short-side candidates: names with 'dn', shooting_star, break_hh20."""
    return ("dn" in name) or name in ("shooting_star", "break_hh20")
