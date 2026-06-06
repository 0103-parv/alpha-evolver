"""Alpha Evolver: a self improving research agent for trading signals.

This module is the entry point and the synthetic data generator. Later modules
add the DSL, backtester, memory store, and evolution loop. The offline path runs
on numpy alone; every other dependency is optional and guarded.
"""

import argparse
import math

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


if __name__ == "__main__":
    main()
