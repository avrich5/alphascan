"""Entry conditions as text expressions over OHLCV and TA-Lib indicators.

    "RSI(close, 14) < 30 and close > SMA(close, 200)"
    "CROSSUP(EMA(close, 12), EMA(close, 26))"
    "BBANDS(close, 20, 2, 2)[2] > close"        # lower band above close
    "volume > 3 * SMA(volume, 50)"

Evaluated with a whitelist over a parsed syntax tree: no Python code
runs, only series (open, high, low, close, volume), numbers, arithmetic,
comparisons, and/or/not, indexing of multi-output indicators, TA-Lib
functions (positional args, series first, then parameters in TA-Lib
order) and the helpers below. Every indicator uses past bars only.
"""
import ast
import numpy as np

SERIES = ("open", "high", "low", "close", "volume")


def _shift(x, n):
    n = int(n)
    y = np.full_like(np.asarray(x, dtype=float), np.nan)
    if n > 0:
        y[n:] = x[:-n]
    else:
        y[:] = x
    return y


def _crossup(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return (a > b) & (_shift(a, 1) <= _shift(b, 1))


def _crossdown(a, b):
    return _crossup(b, a)


HELPERS = {"SHIFT": _shift, "CROSSUP": _crossup, "CROSSDOWN": _crossdown,
           "ABS": np.abs, "MAX2": np.maximum, "MIN2": np.minimum}

_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
        ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
        ast.Div: np.divide, ast.Pow: np.power}


def _talib():
    try:
        import talib
        return talib
    except ImportError:
        return None


def functions():
    """Names usable in expressions."""
    ta = _talib()
    names = sorted(HELPERS)
    if ta is not None:
        names += sorted(ta.get_functions())
    return names


class ConditionError(ValueError):
    pass


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Constant):
        raise ConditionError("only numbers are allowed as constants")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ConditionError("unknown name %r; series are %s" % (node.id, SERIES))
    if isinstance(node, ast.BoolOp):
        vals = [np.asarray(_eval(v, env), dtype=bool) for v in node.values]
        f = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = vals[0]
        for v in vals[1:]:
            out = f(out, v)
        return out
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        if isinstance(node.op, ast.Not):
            return ~np.asarray(v, dtype=bool)
        if isinstance(node.op, ast.USub):
            return -v
        raise ConditionError("operator not allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.Compare):
        left, out = _eval(node.left, env), None
        for op, comp in zip(node.ops, node.comparators):
            if type(op) not in _CMP:
                raise ConditionError("comparison not allowed")
            right = _eval(comp, env)
            r = _CMP[type(op)](left, right)
            out = r if out is None else out & r
            left = right
        return out
    if isinstance(node, ast.Subscript):
        v = _eval(node.value, env)
        idx = node.slice
        if not (isinstance(v, tuple) and isinstance(idx, ast.Constant) and isinstance(idx.value, int)):
            raise ConditionError("indexing only for multi-output indicators, e.g. BBANDS(...)[0]")
        return v[idx.value]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        args = [_eval(a, env) for a in node.args]
        if name in HELPERS:
            return HELPERS[name](*args)
        ta = _talib()
        if ta is not None and name in ta.get_functions():
            args = [np.ascontiguousarray(a, dtype=float) if isinstance(a, np.ndarray) else a for a in args]
            out = getattr(ta, name)(*args)
            return tuple(out) if isinstance(out, (list, tuple)) else out
        raise ConditionError("unknown function %r%s" % (name, "" if ta else " (pip install TA-Lib)"))
    raise ConditionError("syntax not allowed: %s" % type(node).__name__)


def evaluate(expr, d):
    """Boolean mask over the bars of OHLCV frame d; NaN warm-up = False."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ConditionError("syntax error: %s" % e.msg)
    env = {k: d[k].to_numpy(dtype=float) for k in SERIES}
    with np.errstate(invalid="ignore", divide="ignore"):
        m = _eval(tree, env)
    m = np.asarray(m)
    if m.dtype != bool or m.shape != (len(d),):
        raise ConditionError("expression must be a condition (true/false per bar)")
    return m


def evaluate_many(exprs, d):
    """dict name -> expression  ->  dict name -> mask."""
    return {name: evaluate(e, d) for name, e in exprs.items()}
