"""alphascan: statistical gate for trading entry ideas.

Answers one question before any parameter search: does this entry
condition move price, net of market drift, by more than it costs to
trade? And if a combination was picked in-sample, does it transfer?
"""
from .registry import candidates, REGISTRY_VERSION
from .curves import fwd_matrix, drift_median, median_excess, bootstrap_peak_ci
from .data import load_ohlcv, fetch_ohlcv, quality, fingerprint
from .funding import fund_adjusted_peak, funding_schedule
from .gate import run_gate
from .conditions import evaluate as condition, functions as condition_functions
from .transfer import profit_factor, breakeven_fee, transfer_test

__version__ = "0.2.0"
