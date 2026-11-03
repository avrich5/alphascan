"""CLI:
  python -m alphascan --file btc_30m.csv --tf 30m --cost 12 --out gate.csv
  python -m alphascan --exchange binance --symbol BTC/USDT --tf 30m \
      --since 2024-01-01 --cost 12 --out gate.csv
"""
import argparse, json
from . import data, gate


def main():
    p = argparse.ArgumentParser(prog="alphascan")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--exchange")
    p.add_argument("--symbol"); p.add_argument("--since"); p.add_argument("--until")
    p.add_argument("--tf", required=True)
    p.add_argument("--cost", type=float, required=True,
                   help="round-trip cost per 10 000 notional")
    p.add_argument("--kmax", type=int, default=48)
    p.add_argument("--funding", type=float, default=None,
                   help="funding per period per 10 000 notional")
    p.add_argument("--side", choices=["long", "short"], default="long")
    p.add_argument("--nnull", type=int, default=200)
    p.add_argument("--seed", type=int, default=gate.DEFAULT_SEED)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = (data.load_ohlcv(a.file) if a.file else
         data.fetch_ohlcv(a.exchange, a.symbol, a.tf, a.since, a.until))
    q = data.quality(d, a.tf)
    print(json.dumps(q))
    if not q["ok"]:
        raise SystemExit("data quality gate failed")
    r = gate.run_gate(d, a.cost, a.kmax, a.funding, side=a.side, nnull=a.nnull, seed=a.seed)
    r["data_sha256"] = q["sha256"]
    r.to_csv(a.out, index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
