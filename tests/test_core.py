import numpy as np
import pandas as pd
import alphascan as A


def synth(n=6000, seed=1):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.003, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + r.uniform(0, 0.002, n))
    l = np.minimum(o, c) * (1 - r.uniform(0, 0.002, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(dict(ts=ts, open=o, high=h, low=l, close=c,
                             volume=r.lognormal(3, 0.5, n)))


def test_fwd_matrix_matches_loop():
    c = synth(500)["close"].to_numpy()
    F = A.fwd_matrix(c, 5)
    for i in (0, 100, 494):
        for k in range(1, 6):
            assert abs(F[i, k - 1] - (c[i + k] / c[i] - 1) * 1e4) < 1e-2


def test_gate_deterministic_and_random_walk_has_no_edge():
    d = synth()
    r1 = A.run_gate(d, 12.0, kmax=24, nnull=100, seed=7)
    r2 = A.run_gate(d, 12.0, kmax=24, nnull=100, seed=7)
    pd.testing.assert_frame_equal(r1, r2)
    assert (r1.verdict != "enters").all()


def test_breakeven_fee_bisection():
    g = np.array([20.0, -10.0, 15.0, -5.0])
    c = A.breakeven_fee(g)
    assert abs(A.profit_factor(g, c) - 1.0) < 1e-6
    assert abs(c - 2.5) < 1e-6


def test_quality_detects_gap():
    d = synth(300).drop(index=[100, 101]).reset_index(drop=True)
    q = A.quality(d, "30min")
    assert q["gaps"] == 1 and q["missing_bars"] == 2 and q["ok"]


def test_transfer_random_selection_not_significant():
    r = np.random.default_rng(3)
    rows = [dict(combo=i, window=w, pf=r.lognormal(0, 0.3), trades=50, pnl=0)
            for i in range(200) for w in ("k1", "k2", "k3")]
    out = A.transfer_test(pd.DataFrame(rows), ["k1"], ["k3"], n_random=300)
    assert out["k3"]["p"] > 0.05


def test_conditions_safe_and_correct():
    import pytest
    d = synth(1000)
    m = A.condition("close > SMA(close, 20) and volume > 0", d)
    sma = d.close.rolling(20).mean()
    assert (m == (d.close > sma).fillna(False).to_numpy()).all()
    x = A.condition("CROSSUP(EMA(close, 5), EMA(close, 20))", d)
    assert x.dtype == bool and 0 < x.sum() < 200
    assert A.condition("close < BBANDS(close, 20, 2, 2)[2]", d).sum() > 0
    for bad in ("__import__('os')", "close.__class__", "open(1)", "lambda: 1", "close + 1"):
        with pytest.raises(A.conditions.ConditionError):
            A.condition(bad, d)


def test_gate_with_conditions_only():
    d = synth()
    r = A.run_gate(d, 12.0, kmax=24, nnull=50, include_registry=False,
                   conditions={"rsi_low": "RSI(close, 14) < 30"})
    assert list(r.candidate) == ["rsi_low"]
