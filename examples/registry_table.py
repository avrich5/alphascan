"""Reproduces the README table: registry candidates on public Binance
spot candles, 30m, 2024-01-01 .. 2026-11-01, long side, no funding,
round-trip cost 12 per 10 000 notional, horizon up to 48 bars (24h)."""
import sys
import pandas as pd
import alphascan as A

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
rows = []
for s in SYMBOLS:
    d = A.fetch_ohlcv("binance", s, "30m", "2024-01-01", "2026-11-01")
    q = A.quality(d, "30m")
    print(s, q, file=sys.stderr)
    g = A.run_gate(d, 12.0, kmax=48)
    g["symbol"] = s
    g["data_sha256"] = q["sha256"]
    rows.append(g)
out = pd.concat(rows)
out.to_csv(sys.argv[1] if len(sys.argv) > 1 else "registry_table.csv", index=False)
