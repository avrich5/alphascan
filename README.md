# alphascan

A statistical gate for trading entry ideas. Before you search parameters, it answers:

1. **Does this entry condition move price at all?** Median forward move after every firing of the condition, net of market drift, compared with a placebo built from the same condition shifted in time. Only the conservative estimate counts, and it must clear your round-trip cost.
2. **If you picked the best of many combinations, does it transfer?** Selection on training windows, comparison against random sets of the same size on unseen windows, with a p-value.
3. **At what cost does it stop working?** Breakeven fee per side, so you know whether an idea is alive at taker fees, only at maker fees, or nowhere.

Entry conditions are written as plain expressions over candles and any TA-Lib indicator, and the whole gate is available to AI agents as an MCP server.

## Why

To stop spending weeks on empty hypotheses. Most trading ideas fail not in the backtest but before it: the entry condition does not move price more than it costs to trade. Checking that takes minutes, not a parameter search.

A memorable event is not a hypothesis. Everyone knows that buying the bottom of the 10 October 2025 crash paid off; that is hindsight about one bar. A hypothesis is a condition that fires again and again ("a candle falling more than 2 sigma"), and the question is what happens after all of its firings, net of drift and costs. Some of those answers are obvious. Many are not, and they are the ones worth measuring before anything else.

No data ships with the package. Bring your own candles or fetch them from any public exchange API (ccxt). Every result carries the sha256 of the data and the random seed; same inputs give the same output.

## Install

```
pip install .              # core: numpy, pandas
pip install ".[fetch]"     # + ccxt for public exchange data
pip install ".[ta]"        # + TA-Lib indicators in conditions
pip install ".[mcp]"       # + MCP server (Python >= 3.10)
```

## Conditions

```
RSI(close, 14) < 30
CROSSUP(EMA(close, 12), EMA(close, 26)) and close > SMA(close, 200)
close < BBANDS(close, 20, 2, 2)[2]
volume > 3 * SMA(volume, 50) and close < open
```

Series: `open high low close volume`. Any TA-Lib function with positional arguments (series first, then parameters in TA-Lib order); `[i]` picks an output of multi-output indicators. Helpers: `SHIFT(x, n)`, `CROSSUP(a, b)`, `CROSSDOWN(a, b)`, `ABS`, `MAX2`, `MIN2`. Arithmetic, comparisons, `and or not`. Expressions are parsed against a whitelist; no code is executed. All indicators use past bars only.

## Use

```
python -m alphascan --exchange binance --symbol BTC/USDT --tf 30m \
    --since 2024-01-01 --cost 12 --out gate.csv
```

```python
import alphascan as A
d = A.fetch_ohlcv("binance", "BTC/USDT", "30m", "2024-01-01")
print(A.quality(d, "30m"))
g = A.run_gate(d, round_trip_cost=12.0, kmax=48, include_registry=False,
               conditions={"rsi_low": "RSI(close, 14) < 30"})
```

`round_trip_cost` is entry plus exit cost per 10 000 notional (taker 0.06% per side = 12).

## MCP server

```
alphascan-mcp            # stdio
```

Claude Desktop / Claude Code config:

```json
{"mcpServers": {"alphascan": {"command": "alphascan-mcp"}}}
```

Tools: `run_gate` (conditions on public candles, optional `holdout_from` date to judge on one period and report another), `validate_condition`, `list_functions`, `list_registry`, `check_data`, `breakeven_fee`. Every `run_gate` answer reports how many hypotheses were tested in the session.

## Method

- **Forward curve.** For each bar, the move on 10 000 notional after 1..kmax bars; entry at the close of the event bar.
- **Drift.** Median curve of random bars is subtracted; an event is judged by its excess, never by the raw move.
- **Median, not mean.** Means are dominated by tails; a few lucky trades make any idea look good.
- **Peak.** The statistic is the best point of the median excess curve over horizons 1..kmax.
- **Placebo.** Taking the best horizon is biased upward, and firings of one condition cluster in time, so even a meaningless condition shows a positive peak. The reference is the same condition mask shifted circularly by random offsets (same count, same clustering, no link to the future), scored the same way. `edge = peak - median(placebo)`, `edge_lo = peak - 95th percentile(placebo)`.
- **Verdict.** `enters` only if `edge_lo` exceeds the round-trip cost.
- **Funding** (optional) for perpetuals: step penalty per funding period.
- **Transfer test.** In-sample selection is not evidence. A signal exists only if the selected set beats random sets on windows it has not seen.

## Calibration

`python examples/calibration.py` runs the gate on simulated random walks (no edge exists) and on series with a planted edge.

| | result |
|---|---|
| false passes, cost 0 | 23 of 264 (8.7%) |
| false passes, cost 12 | 0 of 264 |
| planted edge +20 per 10 000, cost 12, ~450 events | found 1 of 10 |

The price of this calibration is power: with a few hundred firings the gate finds an edge only when it is roughly twice the cost. That is the intended trade-off for a gate whose job is to reject.

## Example: the registry on public data

Binance spot, 30m candles, 2024-01-01 .. 2026-11-01, buy at the close of the event bar, horizon up to 24 hours, round-trip cost 12 per 10 000 (0.06% per side). Cell: `edge (edge_lo)` per 10 000 notional. Reproduce: `python examples/registry_table.py`.

| candidate | condition (buy at close) | BTC | ETH | SOL |
|---|---|---|---|---|
| `vol_spike_up` | volume > 3x 50-bar median, green candle | 2.1 (-7.0) | 6.1 (-5.1) | 13.9 (-2.7) |
| `vol_spike_dn` | volume > 3x 50-bar median, red candle | 7.1 (-1.1) | 4.5 (-7.3) | 8.5 (-8.7) |
| `big_up_2std` | bar return above +2 sigma of last 50 bars | 2.5 (-6.4) | 5.1 (-4.1) | 16.3 (2.4) |
| `big_dn_2std` | bar return below -2 sigma of last 50 bars | 3.4 (-4.0) | 14.7 (5.5) | 8.2 (-3.6) |
| `hammer` | hammer candle | 0.0 (-3.4) | 4.3 (0.4) | -4.0 (-9.2) |
| `shooting_star` | shooting star candle | 3.1 (-0.4) | -1.3 (-6.2) | -4.7 (-11.6) |
| `engulf_up` | bullish engulfing | 0.3 (-2.6) | -0.6 (-4.1) | -0.9 (-5.9) |
| `engulf_dn` | bearish engulfing | 0.9 (-1.9) | 1.8 (-1.7) | -0.6 (-6.0) |
| `break_hh20` | close above previous 20-bar high | -7.9 (-14.6) | -12.6 (-23.2) | 1.1 (-12.3) |
| `break_ll20` | close below previous 20-bar low | 6.5 (-1.4) | 6.1 (-3.7) | 10.0 (-1.3) |
| `dryup_then_up` | previous bar volume < half median, then green candle | -1.7 (-8.0) | -6.7 (-17.3) | -11.7 (-24.3) |
| `vol3_and_break` | volume spike and close above 20-bar high | -2.7 (-14.3) | -6.9 (-20.9) | 16.1 (-6.2) |
| `squeeze_break_up` | ATR in lowest 20% of 100 bars and 20-bar breakout | -0.5 (-15.1) | -20.4 (-40.8) | -19.2 (-45.9) |
| `vol_confirm_dn` | -2 sigma drop on volume > 2x median | 3.1 (-6.0) | 15.2 (4.4) | 5.7 (-11.1) |
| `engulf_confirm` | bullish engulfing, next bar closes above its high | -0.6 (-6.6) | -6.4 (-13.4) | -2.2 (-12.5) |
| `hammer_confirm` | hammer, next bar closes above its high | -4.9 (-10.7) | 2.8 (-4.1) | 3.5 (-8.5) |
| `tod_00_06` | -2 sigma drop between 00 and 06 UTC | -0.5 (-22.1) | -2.8 (-24.6) | 3.1 (-26.1) |
| `tod_06_12` | -2 sigma drop between 06 and 12 UTC | 12.7 (-3.3) | 4.1 (-17.1) | 1.8 (-33.8) |
| `tod_12_18` | -2 sigma drop between 12 and 18 UTC | -1.1 (-11.4) | 13.7 (-1.2) | 2.4 (-19.8) |
| `tod_18_24` | -2 sigma drop between 18 and 24 UTC | 24.9 (11.9) | 27.3 (7.3) | 19.7 (-9.9) |

What the table says:

- **Nothing clears a taker cost.** 0 of 60 pass. The strongest, a sharp drop after 18:00 UTC, has an edge of 20-27 per 10 000; its conservative bound reaches 11.9 on BTC, just under the cost of 12.
- **Textbook candle patterns are empty.** Hammer, shooting star, engulfing, with or without confirmation: edge within a few units of zero.
- **Breakouts give back.** Buying a close above the 20-bar high has a negative edge on BTC and ETH, and the bound is far below zero: this is evidence against the idea, not just absence of evidence. Volume or a volatility squeeze on top makes it worse.
- **Buying weakness on BTC and ETH** (sharp drops, new lows, red volume spikes) carries the largest positive edges, below cost on this timeframe.

Passing is the right to the next step (parameters, then the transfer test on unseen windows); it is not a strategy.

## Entry candidate registry

`alphascan.registry` holds named conditions; `list_registry` in the MCP server describes them. The registry only grows; a name never changes meaning.

## What it is not

Not trading advice, not a backtester with execution, not a price predictor. It measures whether a hypothesis survives contact with costs and with data it was not fitted on.

## License

MIT, Copyright (c) 2026 Andriy Chyrva.
