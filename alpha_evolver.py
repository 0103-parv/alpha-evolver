"""Alpha Evolver: a self improving research agent for trading signals.

This module is the entry point and the synthetic data generator. Later modules
add the DSL, backtester, memory store, and evolution loop. The offline path runs
on numpy alone; every other dependency is optional and guarded.
"""

import argparse
import math
import warnings

import numpy as np

# Guarded optional imports. The offline path must run with only numpy, so each
# extra dep sets a HAVE flag instead of being required.
try:
    import anthropic  # noqa: F401
    HAVE_ANTHROPIC = True
except ImportError:
    HAVE_ANTHROPIC = False

try:
    import weave  # noqa: F401
    HAVE_WEAVE = True
except ImportError:
    HAVE_WEAVE = False

try:
    import matplotlib  # noqa: F401
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

try:
    import yfinance  # noqa: F401
    HAVE_YFINANCE = True
except ImportError:
    HAVE_YFINANCE = False


PANEL_FIELDS = ("open", "high", "low", "close", "volume", "returns", "fwd_ret")


def synthetic_panel(T=1300, N=40, seed=7):
    """Build a synthetic price and volume panel with two planted edges.

    Edge one is a multi lag reversal: ret[t] leans against the average of the
    prior three returns. Edge two is a weak latent volume edge: a hidden AR(0.8)
    state u drives both volume and a tiny negative push on the next return, so
    volume carries a faint signal about where returns go next.

    Returns a dict of (T, N) float arrays keyed by PANEL_FIELDS.
    """
    rng = np.random.default_rng(seed)

    # Idiosyncratic daily noise, the bulk of each return.
    idio = rng.normal(0.0, 0.012, size=(T, N))

    # Latent AR(0.8) state. Innovation std is set so u has roughly unit
    # stationary variance, which keeps the planted volume edge weak as intended.
    u_innov_std = math.sqrt(1.0 - 0.8 ** 2)
    u_noise = rng.normal(0.0, u_innov_std, size=(T, N))
    u = np.zeros((T, N))
    u[0] = u_noise[0]
    for t in range(1, T):
        u[t] = 0.8 * u[t - 1] + u_noise[t]

    # Returns. The reversal and volume edge are layered on top of the idio noise.
    # The recurrence reads its own lags so it stays sequential over time, but it
    # is vectorized across the N assets.
    ret = np.zeros((T, N))
    for t in range(T):
        r = idio[t].copy()
        if t >= 3:
            r += -0.025 * (ret[t - 1] + ret[t - 2] + ret[t - 3]) / 3.0
        if t >= 1:
            r += -0.0003 * u[t - 1]
        ret[t] = r

    # Prices. Close compounds the returns; open is the prior close.
    close = 100.0 * np.cumprod(1.0 + ret, axis=0)
    open_ = close / (1.0 + ret)

    # Volume. It loads on the day's move size and on the latent state u, so u is
    # observable only through the noisy volume series.
    vol_noise = rng.normal(0.0, 1.0, size=(T, N))
    volume = np.exp(11.0 + 0.4 * np.abs(ret) / 0.012 + 0.4 * u + 1.0 * vol_noise)

    # Intraday range tracks the size of the move. No extra randomness is drawn.
    high = np.maximum(open_, close) * (1.0 + 0.5 * np.abs(ret))
    low = np.minimum(open_, close) * (1.0 - 0.5 * np.abs(ret))

    # Forward return is the next day return. The last row has no next day.
    fwd_ret = np.zeros((T, N))
    fwd_ret[:-1] = ret[1:]
    fwd_ret[-1] = 0.0

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "returns": ret,
        "fwd_ret": fwd_ret,
    }


# ---------------------------------------------------------------------------
# The DSL: a sandboxed S expression language for alphas.
#
# A node is either a field name (leaf) or a list [op, ...children]. Cross
# sectional ops act per day along axis 1. Time series ops are causal: the value
# at t only ever reads x up to and including t. valid() gates every node before
# ev() touches it.
# ---------------------------------------------------------------------------

FIELDS = ("open", "high", "low", "close", "volume", "returns")
UNARY = ("neg", "abs", "sign", "log1p")
BINARY = ("add", "sub", "mul", "div", "min", "max")
CS_OPS = ("cs_rank", "cs_demean", "cs_zscore")
TS_OPS = ("ts_delta", "ts_mean", "ts_std", "ts_zscore",
          "ts_sum", "ts_min", "ts_max", "ts_rank")
WINDOWS = (2, 3, 5, 10, 20)


def valid(node):
    """True only if node is inside the grammar. Never raises."""
    try:
        if isinstance(node, str):
            return node in FIELDS
        if not isinstance(node, (list, tuple)) or len(node) < 1:
            return False
        op = node[0]
        if not isinstance(op, str):
            return False
        if op in UNARY or op in CS_OPS:
            return len(node) == 2 and valid(node[1])
        if op in BINARY:
            return len(node) == 3 and valid(node[1]) and valid(node[2])
        if op in TS_OPS:
            return (len(node) == 3 and valid(node[1])
                    and type(node[2]) is int and node[2] in WINDOWS)
        return False
    except Exception:
        return False


def to_str(node):
    """Canonical formula string. Used later as the memory key."""
    if isinstance(node, str):
        return node
    op = node[0]
    if op in UNARY or op in CS_OPS:
        return f"{op}({to_str(node[1])})"
    if op in BINARY:
        return f"{op}({to_str(node[1])}, {to_str(node[2])})"
    if op in TS_OPS:
        return f"{op}({to_str(node[1])}, {node[2]})"
    raise ValueError(f"unknown op {op}")


def size(node):
    """Node count. Window ints are not nodes and are not counted."""
    if isinstance(node, str):
        return 1
    op = node[0]
    if op in BINARY:
        return 1 + size(node[1]) + size(node[2])
    if op in UNARY or op in CS_OPS or op in TS_OPS:
        return 1 + size(node[1])
    raise ValueError(f"unknown op {op}")


def _rolling(x, w, reducer):
    """Causal rolling reduce along axis 0. out[t] = reducer(x[max(0,t-w+1):t+1])."""
    out = np.empty_like(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for t in range(x.shape[0]):
            lo = max(0, t - w + 1)
            out[t] = reducer(x[lo:t + 1], axis=0)
    return out


def _cs_rank(x):
    """Per row percentile in [0,1] via ordinal argsort. Nan stays nan."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for i in range(T):
        row = x[i]
        mask = np.isfinite(row)
        m = int(mask.sum())
        if m == 0:
            continue
        order = np.argsort(row[mask], kind="mergesort")
        ranks = np.empty(m)
        ranks[order] = np.arange(m, dtype=float)
        out[i, mask] = ranks / (m - 1) if m > 1 else 0.5
    return out


def _ts_rank(x, w):
    """Percentile rank of the current value within the trailing window."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t - w + 1)
        win = x[lo:t + 1]
        cur = x[t]
        # nan comparisons are False, so nan window entries drop out naturally.
        less = (win < cur).sum(axis=0).astype(float)
        k_eff = np.isfinite(win).sum(axis=0)
        denom = k_eff - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(denom > 0, less / denom, 0.5)
        out[t] = np.where(np.isfinite(cur), pct, np.nan)
    return out


def ev(node, F):
    """Evaluate a node against field dict F. Assumes node passed valid()."""
    if isinstance(node, str):
        return np.asarray(F[node], dtype=float)

    op = node[0]

    if op in UNARY:
        x = ev(node[1], F)
        if op == "neg":
            return -x
        if op == "abs":
            return np.abs(x)
        if op == "sign":
            return np.sign(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log1p(x)

    if op in BINARY:
        a = ev(node[1], F)
        b = ev(node[2], F)
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "min":
            return np.minimum(a, b)
        if op == "max":
            return np.maximum(a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a / b

    if op in CS_OPS:
        x = ev(node[1], F)
        if op == "cs_rank":
            return _cs_rank(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(x, axis=1, keepdims=True)
            if op == "cs_demean":
                return x - mean
            std = np.nanstd(x, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (x - mean) / std
        return np.where(std == 0, 0.0, z)

    if op in TS_OPS:
        x = ev(node[1], F)
        w = node[2]
        if op == "ts_mean":
            return _rolling(x, w, np.nanmean)
        if op == "ts_sum":
            return _rolling(x, w, np.nansum)
        if op == "ts_std":
            return _rolling(x, w, np.nanstd)
        if op == "ts_min":
            return _rolling(x, w, np.nanmin)
        if op == "ts_max":
            return _rolling(x, w, np.nanmax)
        if op == "ts_rank":
            return _ts_rank(x, w)
        if op == "ts_delta":
            out = np.full_like(x, np.nan, dtype=float)
            out[w:] = x[w:] - x[:-w]
            return out
        # ts_zscore
        mean = _rolling(x, w, np.nanmean)
        std = _rolling(x, w, np.nanstd)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (x - mean) / std
        return np.where(std == 0, 0.0, z)

    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# The backtest: the honest no lookahead verifier.
#
# Weights are dollar neutral and gross 1 each day. Weights formed on day t earn
# fwd_ret at day t, which is ret[t+1], so nothing reads the future. Headline
# skill is out of sample Sharpe (last slice); the search optimizes in sample.
# ---------------------------------------------------------------------------


def _sharpe(r):
    """Annualized Sharpe of a daily return series. 0 if degenerate."""
    if r.size == 0:
        return 0.0
    sd = float(np.std(r))
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(np.mean(r) / sd * math.sqrt(252.0))


def _row_corr(a, b):
    """Per day cross sectional Pearson correlation. Nan where a row is flat."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        c = num / den
    c[den == 0] = np.nan
    return c


def backtest(node, F, split, cost=0.0001):
    """Evaluate node and score it. Returns the stats dict from CLAUDE.md."""
    fwd = np.asarray(F["fwd_ret"], dtype=float)
    sig = np.nan_to_num(ev(node, F), nan=0.0, posinf=0.0, neginf=0.0)

    # Dollar neutral each day, then scale to gross 1.
    w = sig - sig.mean(axis=1, keepdims=True)
    gross = np.abs(w).sum(axis=1, keepdims=True)
    safe = gross.copy()
    safe[safe == 0.0] = 1.0
    w = w / safe

    # Weights at day t earn fwd_ret at day t, which is ret[t+1].
    port = (w * fwd).sum(axis=1)

    # Turnover is the one way fraction of the book traded, half the L1 weight
    # change. Prior weights are zero before day 0.
    w_prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    turnover = 0.5 * np.abs(w - w_prev).sum(axis=1)

    net = port - cost * turnover

    corr = _row_corr(sig, fwd)
    ic = float(np.nanmean(corr)) if np.isfinite(corr).any() else 0.0

    return {
        "is_sharpe": _sharpe(net[:split]),
        "oos_sharpe": _sharpe(net[split:]),
        "turnover": float(turnover.mean()),
        "size": size(node),
        "ic": ic,
    }


def fitness(stats):
    """Search objective. In sample Sharpe with soft turnover and size penalties."""
    return (stats["is_sharpe"]
            - 0.10 * max(0.0, stats["turnover"] - 1.0)
            - 0.02 * stats["size"])


def load_yfinance_panel(seed=7):
    """Placeholder for the real market data loader.

    Wired in a later module. Kept guarded so the synthetic default never needs
    yfinance installed.
    """
    if not HAVE_YFINANCE:
        raise RuntimeError(
            "yfinance is not installed. Use --data synthetic or pip install yfinance."
        )
    raise NotImplementedError(
        "yfinance data path is not wired yet. Use --data synthetic for now."
    )


def build_panel(args):
    """Pick a data source and return the panel dict."""
    if args.data == "synthetic":
        return synthetic_panel(seed=args.seed)
    return load_yfinance_panel(seed=args.seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Alpha Evolver")
    p.add_argument("--mode", choices=["offline", "claude"], default="offline")
    p.add_argument("--data", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--pop", type=int, default=56)
    p.add_argument("--weave", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("config:")
    print(f"  mode        {args.mode}")
    print(f"  data        {args.data}")
    print(f"  generations {args.generations}")
    print(f"  pop         {args.pop}")
    print(f"  weave       {args.weave}")
    print(f"  seed        {args.seed}")

    panel = build_panel(args)

    print("panel:")
    for name in PANEL_FIELDS:
        print(f"  {name:8s} {panel[name].shape}")

    # Quick sanity on the planted edges. Not part of acceptance, just a check
    # that the reversal and volume edge are present in the generated data.
    ret = panel["returns"]
    fwd = panel["fwd_ret"]
    flat_r = ret[:-1].ravel()
    flat_next = ret[1:].ravel()
    ac1 = float(np.corrcoef(flat_r, flat_next)[0, 1])
    vol = panel["volume"]
    flat_vol = vol[:-1].ravel()
    flat_fwd = fwd[:-1].ravel()
    vol_edge = float(np.corrcoef(flat_vol, -flat_fwd)[0, 1])
    print("sanity:")
    print(f"  lag1 return autocorr {ac1:+.4f}  (reversal, want negative)")
    print(f"  corr(volume, -fwd)   {vol_edge:+.4f}  (volume edge, want positive)")

    # DSL demo: validate, print, evaluate one expression on the panel.
    expr = ["cs_rank", ["neg", ["ts_mean", "returns", 2]]]
    out = ev(expr, panel)
    nonfinite = int((~np.isfinite(out)).sum())
    print("dsl:")
    print(f"  valid       {valid(expr)}")
    print(f"  expr        {to_str(expr)}")
    print(f"  size        {size(expr)}")
    print(f"  out shape   {out.shape}")
    print(f"  non finite  {nonfinite}")

    # Backtest demo on a 2 day reversal signal.
    T = panel["returns"].shape[0]
    split = int(0.70 * T)
    node = ["neg", ["ts_mean", "returns", 2]]
    stats = backtest(node, panel, split)
    print("backtest:")
    print(f"  expr        {to_str(node)}")
    print(f"  split       {split}")
    print(f"  is_sharpe   {stats['is_sharpe']:+.3f}")
    print(f"  oos_sharpe  {stats['oos_sharpe']:+.3f}")
    print(f"  turnover    {stats['turnover']:.3f}")
    print(f"  size        {stats['size']}")
    print(f"  ic          {stats['ic']:+.4f}")
    print(f"  fitness     {fitness(stats):+.3f}")


if __name__ == "__main__":
    main()
