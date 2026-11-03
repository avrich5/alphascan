"""MCP server: the gate as tools for AI agents.

Run:  alphascan-mcp            (stdio)
      alphascan-mcp --http     (streamable HTTP, SDK default host/port)

Data is fetched from public exchange APIs by the server process; nothing
leaves the machine except those public requests.
"""
import sys
import numpy as np
import pandas as pd

from . import conditions as COND
from . import data as DATA
from . import gate as GATE
from . import registry as REG
from . import transfer as TR

try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = """alphascan checks whether a trading entry idea is worth any further work.
It measures the median price move after every firing of an entry condition, net of market
drift, with a bootstrap confidence interval, and compares the lower bound with the trading
cost. Verdicts: enters (worth the next step), below_cost (empty), too_rare (under 100 events).
Use holdout_from to keep a period the condition was not judged on. Every extra hypothesis
raises the chance of a lucky pass: report how many you tested. Statistics only, not advice."""

server = _Server("alphascan", instructions=INSTRUCTIONS)
_CACHE = {}
_TESTED = {"n": 0}


def _candles(exchange, symbol, timeframe, since, until):
    key = (exchange, symbol, timeframe, since, until)
    if key not in _CACHE:
        _CACHE[key] = DATA.fetch_ohlcv(exchange, symbol, timeframe, since, until)
    return _CACHE[key]


def _rows(df):
    rows = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    for r in rows:
        for k in ("n", "bar"):
            if r.get(k) is not None:
                r[k] = int(r[k])
    return rows


@server.tool()
def list_functions() -> dict:
    """Functions and series usable in condition expressions, TA-Lib grouped."""
    ta = COND._talib()
    groups = ta.get_function_groups() if ta else {}
    return {"series": list(COND.SERIES), "helpers": sorted(COND.HELPERS),
            "talib_groups": groups, "examples": [
                "RSI(close, 14) < 30",
                "CROSSUP(EMA(close, 12), EMA(close, 26)) and close > SMA(close, 200)",
                "close < BBANDS(close, 20, 2, 2)[2]",
                "volume > 3 * SMA(volume, 50) and close < open"]}


@server.tool()
def list_registry() -> dict:
    """Built-in entry candidates (registry) with their conditions."""
    return {"version": REG.REGISTRY_VERSION, "candidates": REG.DESCRIPTIONS}


@server.tool()
def validate_condition(expression: str) -> dict:
    """Check an expression without fetching data: syntax, names, that it yields true/false per bar."""
    n = 400
    r = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    d = pd.DataFrame(dict(open=np.r_[c[0], c[:-1]], high=c * 1.01, low=c * 0.99, close=c,
                          volume=r.lognormal(3, 0.5, n)))
    try:
        m = COND.evaluate(expression, d)
        return {"ok": True, "fires_on_test_series": int(m.sum())}
    except COND.ConditionError as e:
        return {"ok": False, "error": str(e)}


@server.tool()
def check_data(exchange: str, symbol: str, timeframe: str, since: str, until: str | None = None) -> dict:
    """Fetch public candles (ccxt exchange id, e.g. binance, bybit, okx) and report quality and sha256."""
    return DATA.quality(_candles(exchange, symbol, timeframe, since, until), timeframe)


@server.tool()
def run_gate(exchange: str, symbol: str, timeframe: str, since: str,
             conditions: dict[str, str], round_trip_cost: float = 12.0,
             until: str | None = None, holdout_from: str | None = None,
             side: str = "long", horizon_bars: int = 48,
             funding_per_period: float | None = None, include_registry: bool = False,
             seed: int = GATE.DEFAULT_SEED) -> dict:
    """Gate entry conditions on public candles.

    conditions: {name: expression}, e.g. {"rsi_oversold": "RSI(close, 14) < 30"}.
    round_trip_cost: entry + exit cost per 10 000 notional (0.06% per side = 12).
    holdout_from: date; the verdict is computed before it and the same statistics
    are reported after it, so a pass can be checked on data it was not judged on.
    side: long or short. horizon_bars: max holding horizon in bars.
    funding_per_period: perpetual funding per 8h per 10 000 notional, optional.
    """
    d = _candles(exchange, symbol, timeframe, since, until)
    q = DATA.quality(d, timeframe)
    if not q["ok"]:
        return {"error": "data quality gate failed", "data": q}
    try:
        for e in conditions.values():
            COND.evaluate(e, d.head(10))
    except COND.ConditionError as e:
        return {"error": str(e)}
    _TESTED["n"] += len(conditions) + (len(REG.DESCRIPTIONS) if include_registry else 0)
    kw = dict(round_trip_cost=round_trip_cost, kmax=horizon_bars, side=side, seed=seed,
              fund_per_period=funding_per_period, include_registry=include_registry)
    parts = {"all": d}
    if holdout_from:
        cut = pd.Timestamp(holdout_from, tz="UTC")
        parts = {"judged": d[d.ts < cut].reset_index(drop=True),
                 "holdout": d[d.ts >= cut].reset_index(drop=True)}
    out = {"data": q, "settings": kw,
           "hypotheses_tested_this_session": _TESTED["n"]}
    for label, part in parts.items():
        m = {n: COND.evaluate(e, part) for n, e in conditions.items()}
        out[label] = _rows(GATE.run_gate(part, extra=m, **kw))
    out["note"] = ("Lower CI bound above cost = enters. With many hypotheses some passes are luck; "
                   "trust a pass that holds in the holdout period.")
    return out


@server.tool()
def breakeven_fee(trade_results: list[float], target_profit_factor: float = 1.0) -> dict:
    """Cost per side (per 10 000 notional) at which gross per-trade results fall to the target profit factor."""
    g = np.asarray(trade_results, dtype=float)
    return {"fee_per_side": TR.breakeven_fee(g, target_profit_factor),
            "profit_factor_at_zero_cost": TR.profit_factor(g), "trades": int(len(g))}


def main():
    if "--http" in sys.argv:
        server.run("streamable-http")
    else:
        server.run()


if __name__ == "__main__":
    main()
