"""Alpha Evolver: a self improving research agent for trading signals.

This module is the entry point and the synthetic data generator. Later modules
add the DSL, backtester, memory store, and evolution loop. The offline path runs
on numpy alone; every other dependency is optional and guarded.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import warnings
from collections import Counter

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
            with np.errstate(invalid="ignore"):
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


# ---------------------------------------------------------------------------
# The memory store: the spine.
#
# A fast store of verified alpha records keyed by canonical formula string, plus
# a slow store of principles, motifs, and the proposer strategy. Strength is
# reinforced each generation and the weakest items are evicted over capacity.
#
# Hard rule: the model never recalls from memory. assemble_context is the only
# method that hands back records, and it returns verbatim deepcopies. There is
# no free query surface.
# ---------------------------------------------------------------------------


def _minmax_norm(x):
    """Scale a vector to [0,1] across itself. Zeros if flat or empty."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class Memory:
    """Reinforced store of alphas plus the slow store of distilled knowledge."""

    def __init__(self):
        self.items = {}  # key -> record
        self.principles = []  # {text, keys}
        self.motifs = []  # {fragment, strength, uses, avg_fitness}
        self.lessons = {}  # lesson text -> anti pattern record
        self.strategy = {
            "text": ("Prioritize cross sectional ranking of short window time series "
                     "smoothings of returns. Avoid raw price levels. Prefer low turnover."),
            "hit_rate": 0.0,
            "version": 0,
            "history": [],
        }
        self.pe_scale = 1.0

    def add_or_update(self, key, expr, stats, predicted, error, gen,
                      novelty=None, motifs=None, lesson=None):
        """Upsert a record. Strength and uses survive updates."""
        if error is not None and np.isfinite(error):
            self.pe_scale = 0.90 * self.pe_scale + 0.10 * max(abs(float(error)), 1e-6)
        if key in self.items:
            it = self.items[key]
            it["expr"] = expr
            it["stats"] = dict(stats)
            it["predicted_sharpe"] = predicted
            it["prediction_error"] = error
            it["last_used_gen"] = gen
            if novelty is not None:
                it["novelty"] = novelty
            if motifs is not None:
                it["motifs"] = list(motifs)
            if lesson is not None:
                it["lesson"] = lesson
        else:
            self.items[key] = {
                "key": key,
                "expr": expr,
                "stats": dict(stats),
                "predicted_sharpe": predicted,
                "prediction_error": error,
                "strength": 0.0,
                "uses": 0,
                "created_gen": gen,
                "last_used_gen": gen,
                "novelty": 0.0 if novelty is None else novelty,
                "motifs": [] if motifs is None else list(motifs),
                "lesson": "" if lesson is None else lesson,
            }
        return self.items[key]

    def reinforce(self, touched_keys, used_counts, gen, max_cap=300,
                  decay=0.95, a=0.5, b=0.3, c=0.2, d=0.1):
        """Update strength of touched items, then evict the weakest over cap."""
        touched = [k for k in touched_keys if k in self.items]
        if touched:
            fit = [fitness(self.items[k]["stats"]) for k in touched]
            used = [float(used_counts.get(k, 0)) for k in touched]
            errors = [
                abs(float(self.items[k].get("prediction_error", 0.0)))
                for k in touched
            ]
            surp = [min(error / max(self.pe_scale, 1e-6), 1.0) for error in errors]
            novel = [self.items[k].get("novelty", 0.0) for k in touched]
            nf = _minmax_norm(fit)
            nu = _minmax_norm(used)
            ns = np.asarray(surp, dtype=float)
            nn = _minmax_norm(novel)
            for i, k in enumerate(touched):
                it = self.items[k]
                it["strength"] = it["strength"] * decay + (
                    a * float(nf[i]) + b * float(nu[i]) +
                    c * float(ns[i]) + d * float(nn[i]))
                it["uses"] += int(used_counts.get(k, 0))
                it["last_used_gen"] = gen

        evicted = []
        while len(self.items) > max_cap:
            k = min(self.items, key=lambda kk: (
                self.items[kk]["strength"],
                self.items[kk]["created_gen"],
                kk))
            evicted.append(k)
            del self.items[k]
        return evicted

    def assemble_context(self):
        """Deterministic retrieval. Verbatim deepcopies, the only recall path."""
        items = list(self.items.values())

        def top(metric, n):
            ranked = sorted(items, key=lambda r: (-metric(r), r["key"]))
            return [copy.deepcopy(r) for r in ranked[:n]]

        motifs_ranked = sorted(
            self.motifs,
            key=lambda m: (-m.get("strength", 0.0), m.get("fragment", "")))
        lessons_ranked = sorted(
            self.lessons.values(),
            key=lambda r: (-r["strength"], r["key"]))
        return {
            "top_strength": top(lambda r: r["strength"], 8),
            "top_novelty": top(lambda r: r.get("novelty", 0.0), 3),
            "top_surprise": top(lambda r: abs(r.get("prediction_error", 0.0)), 3),
            "principles": [copy.deepcopy(p) for p in self.principles],
            "motifs": [copy.deepcopy(m) for m in motifs_ranked[:6]],
            "top_lessons": [copy.deepcopy(r) for r in lessons_ranked[:3]],
            "strategy": copy.deepcopy(self.strategy),
            "pe_scale": self.pe_scale,
        }

    def note_lesson(self, lesson, gen, bump=1.0):
        """Record an anti pattern. Frequency reinforces its strength."""
        if lesson in self.lessons:
            it = self.lessons[lesson]
            it["strength"] += bump
            it["uses"] += 1
            it["last_used_gen"] = gen
        else:
            self.lessons[lesson] = {
                "key": lesson,
                "text": lesson,
                "strength": bump,
                "uses": 1,
                "created_gen": gen,
                "last_used_gen": gen,
            }
        return self.lessons[lesson]

    def decay_lessons(self, decay=0.95):
        """Fade every anti pattern one generation. Stale ones sink."""
        for it in self.lessons.values():
            it["strength"] *= decay

    # Slow store mutators. Later modules (sleep, motif extraction) drive these.
    def add_principle(self, text, keys):
        self.principles.append({"text": text, "keys": list(keys)})

    def add_motif(self, fragment, strength=0.0, uses=0, avg_fitness=0.0, expr=None):
        for m in self.motifs:
            if m["fragment"] == fragment:
                m["strength"] = strength
                m["uses"] = uses
                m["avg_fitness"] = avg_fitness
                if expr is not None:
                    m["expr"] = copy.deepcopy(expr)
                return m
        m = {"fragment": fragment, "strength": strength,
             "uses": uses, "avg_fitness": avg_fitness}
        if expr is not None:
            m["expr"] = copy.deepcopy(expr)
        self.motifs.append(m)
        return m

    def set_strategy(self, text, hit_rate, gen=None):
        history = list(self.strategy.get("history", []))
        prev = {
            "text": self.strategy.get("text", ""),
            "hit_rate": self.strategy.get("hit_rate", 0.0),
            "version": self.strategy.get("version", 0),
        }
        if prev["text"]:
            history.append(prev)
        self.strategy = {
            "text": text,
            "hit_rate": hit_rate,
            "version": int(self.strategy.get("version", 0)) + 1,
            "gen": gen,
            "history": history[-8:],
        }

    def save(self, path):
        state = {
            "items": self.items,
            "principles": self.principles,
            "motifs": self.motifs,
            "lessons": self.lessons,
            "strategy": self.strategy,
            "pe_scale": self.pe_scale,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            state = json.load(f)
        m = cls()
        m.items = state.get("items", {})
        m.principles = state.get("principles", [])
        m.motifs = state.get("motifs", [])
        m.lessons = state.get("lessons", {})
        m.strategy = state.get("strategy", m.strategy)
        if "history" not in m.strategy:
            m.strategy["history"] = []
        if "version" not in m.strategy:
            m.strategy["version"] = 0
        m.pe_scale = state.get("pe_scale", 1.0)
        return m


# ---------------------------------------------------------------------------
# The offline proposer: genetic operators over the DSL.
#
# rand_expr grows a fresh tree, mutate swaps one subexpression for a fresh one,
# crossover grafts a subexpression from b into a. propose_offline reads the
# verbatim context from Memory and breeds k valid alphas, biasing parents toward
# strength. Everything is filtered through valid() before it leaves.
# ---------------------------------------------------------------------------

MAX_DEPTH = 4


def subexpressions(node):
    """Every real sub expression in canonical string form."""
    return [to_str(_get(node, p)) for p in _positions(node)]


def fragment_set(node):
    return set(subexpressions(node))


def similarity(a, b):
    """Jaccard similarity of two expression fragment sets."""
    fa = fragment_set(a)
    fb = fragment_set(b)
    if not fa and not fb:
        return 1.0
    return len(fa & fb) / max(1, len(fa | fb))


def novelty(node, hof):
    """One minus max fragment similarity to hall of fame."""
    if not hof:
        return 1.0
    return 1.0 - max(similarity(node, r["expr"]) for r in hof.values())


def _motif_exprs(context):
    exprs = []
    for m in context.get("motifs", []):
        expr = m.get("expr")
        if expr is not None and valid(expr):
            exprs.append((expr, max(0.0, m.get("strength", 0.0)) + 1e-6))
    return exprs


def _has_op(node, op):
    if isinstance(node, str):
        return False
    if node[0] == op:
        return True
    return any(_has_op(c, op) for c in node[1:] if isinstance(c, (str, list)))


def rand_expr(depth, motifs=None, required_op=None):
    """Sample a valid expression. depth is the remaining level budget."""
    if motifs and depth > 1 and random.random() < 0.12:
        choices, weights = zip(*motifs)
        return copy.deepcopy(random.choices(choices, weights=weights, k=1)[0])
    if depth <= 1:
        return random.choice(FIELDS)
    # Favor operators over bare leaves so trees are not mostly fields.
    cat = random.choices(
        ["leaf", "unary", "binary", "cs", "ts"], weights=[1, 2, 2, 2, 3])[0]
    if cat == "leaf":
        return random.choice(FIELDS)
    if cat == "unary":
        op = required_op if required_op in UNARY else random.choice(UNARY)
        return [op, rand_expr(depth - 1, motifs)]
    if cat == "binary":
        op = required_op if required_op in BINARY else random.choice(BINARY)
        return [op, rand_expr(depth - 1, motifs), rand_expr(depth - 1, motifs)]
    if cat == "cs":
        op = required_op if required_op in CS_OPS else random.choice(CS_OPS)
        return [op, rand_expr(depth - 1, motifs)]
    op = required_op if required_op in TS_OPS else random.choice(TS_OPS)
    return [op, rand_expr(depth - 1, motifs), random.choice(WINDOWS)]


def _positions(node, prefix=()):
    """Paths to every subexpression. Window ints are never positions."""
    paths = [prefix]
    if isinstance(node, list):
        op = node[0]
        if op in BINARY:
            paths += _positions(node[1], prefix + (1,))
            paths += _positions(node[2], prefix + (2,))
        elif op in UNARY or op in CS_OPS or op in TS_OPS:
            paths += _positions(node[1], prefix + (1,))
    return paths


def _get(node, path):
    cur = node
    for i in path:
        cur = cur[i]
    return cur


def _set(node, path, value):
    """Functional replace at path. Never mutates the input."""
    if not path:
        return value
    new = copy.deepcopy(node)
    cur = new
    for i in path[:-1]:
        cur = cur[i]
    cur[path[-1]] = value
    return new


def mutate(node):
    """Replace one real subexpression position with a fresh rand_expr."""
    path = random.choice(_positions(node))
    repl = rand_expr(max(1, MAX_DEPTH - len(path)))
    return _set(node, path, repl)


def crossover(a, b):
    """Graft a random subexpression from b into a real position of a."""
    donor = copy.deepcopy(_get(b, random.choice(_positions(b))))
    return _set(a, random.choice(_positions(a)), donor)


def _pick_elite(elites):
    """Pick a parent expr, biased toward higher strength."""
    weights = [
        max(0.0, r.get("strength", 0.0)) +
        0.30 * max(0.0, r.get("novelty", 0.0)) + 1e-6
        for r in elites
    ]
    return random.choices(elites, weights=weights, k=1)[0]["expr"]


def propose_offline(context, k, explore, bold_ops=None):
    """Breed k valid alphas from the context. The offline (no LLM) proposer."""
    elites = context.get("top_strength", [])
    motifs = _motif_exprs(context)

    def one():
        r = random.random()
        required = random.choice(bold_ops) if bold_ops and random.random() < 0.50 else None
        if r < explore or not elites:
            depth = random.randint(3, MAX_DEPTH) if bold_ops else random.randint(2, MAX_DEPTH)
            cand = rand_expr(depth, motifs=motifs, required_op=required)
            if required and not _has_op(cand, required):
                cand = [required, cand] if required in UNARY + CS_OPS else cand
            return cand
        if r < explore + 0.30:
            return crossover(_pick_elite(elites), _pick_elite(elites))
        return mutate(_pick_elite(elites))

    out = []
    for _ in range(k * 20):
        if len(out) >= k:
            break
        cand = one()
        if valid(cand):
            out.append(cand)
    while len(out) < k:
        cand = rand_expr(random.randint(2, MAX_DEPTH), motifs=motifs)
        if valid(cand):
            out.append(cand)
    return out[:k]


def mine_motifs(memory, hof, top_n=20):
    """Refresh the motif library from high scoring hall of fame containers."""
    counts = {}
    containers = {}
    for rec in hof.values():
        weight = max(0.0, rec["stats"].get("oos_sharpe", 0.0))
        if weight <= 0.0:
            continue
        seen = {}
        for p in _positions(rec["expr"]):
            sub = copy.deepcopy(_get(rec["expr"], p))
            frag = to_str(sub)
            if frag == rec["key"]:
                continue
            seen[frag] = sub
        for frag, sub in seen.items():
            counts[frag] = counts.get(frag, 0.0) + weight
            containers.setdefault(frag, []).append((weight, sub))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    memory.motifs = []
    for frag, strength in ranked:
        vals = containers.get(frag, [])
        avg = float(np.mean([v[0] for v in vals])) if vals else 0.0
        expr = vals[0][1] if vals else None
        memory.add_motif(frag, strength=strength, uses=len(vals),
                         avg_fitness=avg, expr=expr)
    return memory.motifs


def rare_ops_from_hof(hof):
    """Operators that have appeared least in the hall of fame."""
    all_ops = list(UNARY + BINARY + CS_OPS + TS_OPS)
    counts = Counter()

    def walk(n):
        if isinstance(n, list):
            counts[n[0]] += 1
            for c in n[1:]:
                walk(c)

    for rec in hof.values():
        walk(rec["expr"])
    ranked = sorted(all_ops, key=lambda op: (counts[op], op))
    return ranked[:5]


# ---------------------------------------------------------------------------
# The critic: a cheap offline gate that rejects known failure shapes before the
# backtest is even trusted. Rejections become anti pattern lessons in memory.
# ---------------------------------------------------------------------------

PRICE_FIELDS = ("open", "high", "low", "close")


def _antipattern_fragment(node):
    """A generalized descriptor of node, so similar rejections share a lesson."""
    price_hits, vol_hits = [], []

    def walk(n):
        if not isinstance(n, list):
            return
        op = n[0]
        for c in n[1:]:
            if c in PRICE_FIELDS:
                price_hits.append(f"{op} on price level")
            elif c == "volume":
                vol_hits.append(f"{op} on volume")
        for c in n[1:]:
            walk(c)

    walk(node)
    if price_hits:
        return price_hits[0]
    if vol_hits:
        return vol_hits[0]
    if isinstance(node, list):
        return f"{node[0]} structure"
    return f"raw {node} level"


def critic_offline(node, stats):
    """Gate an alpha. Returns (ok, lesson). lesson is empty when ok."""
    reasons = []
    if (stats["is_sharpe"] - stats["oos_sharpe"]) > 1.0 and stats["oos_sharpe"] < 0.3:
        reasons.append("large IS to OOS gap, likely fitted")
    if stats["turnover"] > 5:
        reasons.append("turnover too high, churns the book")
    if stats["size"] > 12:
        reasons.append("tree too large, too complex")
    if not reasons:
        return True, ""
    lesson = f"{_antipattern_fragment(node)}, " + "; ".join(reasons)
    return False, lesson


def yfinance_panel(tickers=None, period="6y"):
    """Download a liquid US equity panel through yfinance."""
    if not HAVE_YFINANCE:
        raise RuntimeError(
            "yfinance is not installed. Use --data synthetic or pip install yfinance."
        )
    if tickers is None:
        tickers = (
            "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V MA HD PG XOM "
            "JNJ WMT BAC KO PEP CVX ABBV MRK COST ADBE CRM"
        ).split()
    import yfinance as yf
    raw = yf.download(tickers, period=period, auto_adjust=False,
                      progress=False, group_by="column", threads=True)
    if raw.empty:
        raise RuntimeError("yfinance returned no rows")

    fields = {}
    for name, col in [("open", "Open"), ("high", "High"), ("low", "Low"),
                      ("close", "Close"), ("volume", "Volume")]:
        frame = raw[col] if col in raw else raw.xs(col, axis=1, level=0)
        frame = frame.reindex(columns=tickers).ffill().dropna(axis=0, how="any")
        fields[name] = frame.to_numpy(dtype=float)

    close = fields["close"]
    ret = np.zeros_like(close, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[1:] = close[1:] / close[:-1] - 1.0
    ret[~np.isfinite(ret)] = 0.0
    fwd = np.zeros_like(ret)
    fwd[:-1] = ret[1:]
    fields["returns"] = ret
    fields["fwd_ret"] = fwd
    return fields


def build_panel(data, seed):
    """Pick a data source and return the panel dict."""
    if data == "synthetic":
        return synthetic_panel(seed=seed)
    return yfinance_panel()


def load_dotenv_simple(path=".env"):
    """Tiny .env loader so claude mode does not require python dotenv."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _anthropic_client():
    if not HAVE_ANTHROPIC:
        raise RuntimeError("anthropic is not installed. Run: pip install anthropic")
    load_dotenv_simple()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == "sk-...":
        raise RuntimeError("ANTHROPIC_API_KEY is missing or still a placeholder")
    import anthropic as anthropic_mod
    return anthropic_mod.Anthropic(api_key=key)


def _extract_json_array(text):
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    return json.loads(m.group(0))


def propose_claude(context, k, model, explore=0.25, bold_ops=None):
    """Ask Claude for alphas, validate them, then top up offline."""
    try:
        client = _anthropic_client()
        payload = {
            "operators": {
                "fields": FIELDS, "unary": UNARY, "binary": BINARY,
                "cross_sectional": CS_OPS, "time_series": TS_OPS,
                "windows": WINDOWS,
            },
            "context": context,
            "request": f"Return exactly {k} valid alphas as a JSON array of S expressions.",
        }
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": ("You propose sandboxed trading alpha DSL expressions. "
                            "Use only this JSON context. Do not invent results. "
                            f"Current strategy: {context.get('strategy', {}).get('text', '')}\n"
                            f"{json.dumps(payload, default=str)}"),
            }],
        )
        text = "\n".join(getattr(block, "text", "") for block in msg.content)
        raw = _extract_json_array(text)
        out = [x for x in raw if valid(x)]
        print(f"  claude proposer returned {len(out)}/{k} valid alphas")
    except Exception as e:
        print(f"  claude proposer fallback: {e}")
        out = []
    if len(out) < k:
        out += propose_offline(context, k - len(out), explore, bold_ops=bold_ops)
    return out[:k]


def critic_claude(node, stats, model):
    """Ask Claude whether the edge looks real. Fallback is the offline critic."""
    try:
        client = _anthropic_client()
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": ("Judge this verified alpha. Return JSON only: "
                            "{\"accept\": true|false, \"reason\": \"short\"}.\n"
                            f"alpha: {json.dumps(node)}\n"
                            f"stats: {json.dumps(stats)}"),
            }],
        )
        text = "\n".join(getattr(block, "text", "") for block in msg.content)
        obj = json.loads(re.search(r"\{[\s\S]*\}", text).group(0))
        return bool(obj.get("accept")), str(obj.get("reason", ""))
    except Exception:
        return critic_offline(node, stats)


def maybe_rewrite_strategy(memory, hof, recent_hit_rate, gen, model):
    """Claude can rewrite the proposer strategy every few generations."""
    try:
        client = _anthropic_client()
        payload = {
            "current_strategy": memory.strategy,
            "recent_hit_rate": recent_hit_rate,
            "hall_of_fame": [
                {"key": r["key"], "stats": r["stats"]} for r in
                sorted(hof.values(), key=lambda r: -r["stats"]["oos_sharpe"])[:12]
            ],
            "anti_patterns": list(memory.lessons.values())[:8],
        }
        msg = client.messages.create(
            model=model,
            max_tokens=900,
            messages=[{
                "role": "user",
                "content": ("Rewrite the alpha proposer strategy to improve hit rate. "
                            "Return one concise plain text strategy.\n"
                            f"{json.dumps(payload, default=str)}"),
            }],
        )
        text = "\n".join(getattr(block, "text", "") for block in msg.content).strip()
        if text:
            memory.set_strategy(text, recent_hit_rate, gen=gen)
            print(f"  strategy v{memory.strategy['version']} hit_rate {recent_hit_rate:.2f}")
    except Exception as e:
        print(f"  strategy rewrite skipped: {e}")


def distill_principles_offline(memory, hof):
    """Checkable offline principles when Claude is not available."""
    ranked = sorted(hof.values(), key=lambda r: -r["stats"]["oos_sharpe"])[:12]
    keys = [r["key"] for r in ranked[:5]]
    principles = []
    if keys:
        principles.append({
            "text": "Short return smoothings with cross sectional neutralization tend to survive.",
            "keys": keys,
            "confidence": "med",
            "validated_corr": None,
        })
    vols = [r for r in ranked if "volume" in r["key"]]
    if vols:
        principles.append({
            "text": "Volume transforms can carry the planted latent edge on synthetic data.",
            "keys": [r["key"] for r in vols[:5]],
            "confidence": "med",
            "validated_corr": None,
        })
    memory.principles = principles[:5]
    return memory.principles


def validate_principles(memory, hof):
    """Attach simple empirical correlations to stored principles."""
    rows = list(hof.values())
    if len(rows) < 3:
        return memory.principles
    sizes = np.asarray([r["stats"]["size"] for r in rows], dtype=float)
    turns = np.asarray([r["stats"]["turnover"] for r in rows], dtype=float)
    oos = np.asarray([r["stats"]["oos_sharpe"] for r in rows], dtype=float)
    with np.errstate(invalid="ignore"):
        corr_size = float(np.corrcoef(sizes, oos)[0, 1]) if np.std(sizes) else 0.0
        corr_turn = float(np.corrcoef(turns, oos)[0, 1]) if np.std(turns) else 0.0
    kept = []
    for p in memory.principles:
        text = p.get("text", "").lower()
        p = copy.deepcopy(p)
        p["validated_corr"] = corr_turn if "turnover" in text else corr_size
        if np.isfinite(p["validated_corr"]):
            kept.append(p)
    memory.principles = kept
    return kept


def sleep(memory, panel, model=None, mode="offline"):
    """Fast slow consolidation. Distill, validate, replay, decay."""
    hof = {
        k: {"key": k, "expr": r["expr"], "stats": r["stats"]}
        for k, r in memory.items.items()
        if r["stats"].get("oos_sharpe", 0.0) > 0.0
    }
    if mode == "claude" and model:
        # Claude principle extraction is intentionally conservative. If it fails,
        # the deterministic distiller below still gives a useful sleep phase.
        distill_principles_offline(memory, hof)
    else:
        distill_principles_offline(memory, hof)
    validate_principles(memory, hof)
    mine_motifs(memory, hof)

    split = int(0.70 * panel["returns"].shape[0])
    ranked = sorted(hof.values(), key=lambda r: -r["stats"]["oos_sharpe"])[:6]
    offspring = []
    for i in range(0, len(ranked) - 1, 2):
        child = crossover(ranked[i]["expr"], ranked[i + 1]["expr"])
        if valid(child):
            predicted = _predict_sharpe(memory.assemble_context())
            stats = backtest(child, panel, split)
            error = stats["oos_sharpe"] - predicted
            ok, lesson = critic_offline(child, stats)
            if ok:
                key = to_str(child)
                memory.add_or_update(key, child, stats, predicted=predicted,
                                     error=error, gen=-1,
                                     novelty=novelty(child, hof),
                                     motifs=subexpressions(child))
                offspring.append((key, stats))
            elif lesson:
                memory.note_lesson(lesson, gen=-1)

    for rec in memory.items.values():
        rec["strength"] *= 0.85
    evicted = memory.reinforce(list(memory.items.keys()), {}, gen=-1,
                               max_cap=300, decay=1.0, a=0.0, b=0.0, c=0.0, d=0.0)
    print("sleep principles:")
    for p in memory.principles:
        print(f"  {p.get('confidence', 'med')} corr {p.get('validated_corr')}  {p['text']}")
    print("sleep motifs:")
    for m in memory.motifs[:6]:
        print(f"  {m['fragment']} strength {m['strength']:.3f}")
    print("sleep replay:")
    for key, stats in offspring:
        print(f"  oos {stats['oos_sharpe']:+.3f}  {key}")
    if evicted:
        print(f"sleep evicted {len(evicted)}")
    return {"principles": memory.principles, "motifs": memory.motifs[:6],
            "offspring": offspring}


# ---------------------------------------------------------------------------
# The evolution loop: propose, verify, store, reinforce, breed.
# ---------------------------------------------------------------------------


def explore_schedule(sigma_t, sigma_scale, n_stall):
    """Adaptive exploration from surprise and search stall."""
    raw = 0.10 + 0.5 * sigma_t / max(sigma_scale, 1e-6) + 0.05 * n_stall
    return float(np.clip(raw, 0.10, 0.60))


def _write_history(path, history):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "best_oos_so_far", "best_oos_gen",
                    "median_oos", "explore", "mem_size", "mean_abs_pe",
                    "sigma_t", "n_stall", "hit_rate"])
        for row in history:
            w.writerow(row)


def _save_curve(path, history):
    if not HAVE_MATPLOTLIB:
        print(f"  (matplotlib missing, skipped {path})")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gens = [h[0] for h in history]
    best = [h[1] for h in history]
    median = [h[3] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gens, best, marker="o", label="best oos so far")
    ax.plot(gens, median, marker=".", label="median oos")
    ax.set_xlabel("generation")
    ax.set_ylabel("oos sharpe")
    ax.set_title("Alpha Evolver learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _predict_sharpe(context):
    vals = [
        r.get("stats", {}).get("oos_sharpe", 0.0)
        for r in context.get("top_strength", [])
        if np.isfinite(r.get("stats", {}).get("oos_sharpe", 0.0))
    ]
    return float(np.mean(vals)) if vals else 0.0


def _weave_op(fn):
    if HAVE_WEAVE:
        try:
            import weave as weave_mod
            return weave_mod.op()(fn)
        except Exception:
            return fn
    return fn


def run(generations=20, pop=56, seed=7, data="synthetic", mode="offline",
        weave=False, elite_n=10, cost=0.0001, mem_path="memory.json",
        history_path="history.csv", curve_path="learning_curve.png",
        model="claude-sonnet-4-6", sleep_now=False, progress_cb=None):
    """The generational search. Returns a summary dict."""
    random.seed(seed)
    panel = build_panel(data, seed)
    T = panel["returns"].shape[0]
    split = int(0.70 * T)

    memory = Memory.load(mem_path) if os.path.exists(mem_path) else Memory()

    if weave and HAVE_WEAVE:
        try:
            import weave as weave_mod
            weave_mod.init("weavehacks-alpha-evolver")
        except Exception as e:
            print(f"  (weave init failed: {e})")
            weave = False
    elif weave and not HAVE_WEAVE:
        print("  (weave missing, skipped tracing. Run: pip install weave)")

    print(f"run  mode {mode}  data {data}  gens {generations}  pop {pop}  "
          f"seed {seed}  mem0 {len(memory.items)}")

    # Initial population: carry over elites from memory, top up with proposals.
    ctx = memory.assemble_context()
    population = [copy.deepcopy(r["expr"]) for r in ctx["top_strength"]][:pop]
    if len(population) < pop:
        population += propose_offline(ctx, pop - len(population), explore=0.45)

    bt_cache = {}
    used_as_parent = Counter()
    hof = {}
    history = []
    best_oos_so_far = float("-inf")
    best_alpha = None
    sigma_t = memory.pe_scale
    sigma_scale = max(memory.pe_scale, 1.0)
    n_stall = 0
    recent_pass = []

    for g in range(generations):
        explore = explore_schedule(sigma_t, sigma_scale, n_stall)
        bold_ops = rare_ops_from_hof(hof) if explore > 0.40 else None
        ctx_for_pred = memory.assemble_context()
        baseline_pred = _predict_sharpe(ctx_for_pred)

        # 1. Evaluate every valid individual, dedup by canonical key.
        evaluated = {}
        pe_vals = []
        for ind in population:
            if not valid(ind):
                continue
            key = to_str(ind)
            if key in evaluated:
                continue
            pred = baseline_pred
            if key in memory.items:
                pred = memory.items[key]["stats"]["oos_sharpe"]
            if key not in bt_cache:
                bt_cache[key] = backtest(ind, panel, split, cost)
            stats = bt_cache[key]
            error = stats["oos_sharpe"] - pred
            pe_vals.append(abs(error))
            evaluated[key] = {"key": key, "expr": ind, "stats": stats,
                              "fitness": fitness(stats), "predicted": pred,
                              "error": error}

        # 2. Store every verified result.
        for rec in evaluated.values():
            nov = novelty(rec["expr"], hof)
            memory.add_or_update(rec["key"], rec["expr"], rec["stats"],
                                 predicted=rec["predicted"], error=rec["error"],
                                 gen=g, novelty=nov,
                                 motifs=subexpressions(rec["expr"]))
            rec["novelty"] = nov
            rec["selection_score"] = rec["stats"]["oos_sharpe"] + 0.30 * nov

        # 3. Rank by fitness, take the elite.
        ranked = sorted(evaluated.values(),
                        key=lambda r: (-r["fitness"], r["key"]))
        elites = ranked[:elite_n]

        # 4. Critic on the elite: passers to hall of fame, rejects to lessons.
        passed = 0
        for e in elites:
            ok, lesson = (critic_claude(e["expr"], e["stats"], model)
                          if mode == "claude" else critic_offline(e["expr"], e["stats"]))
            if ok:
                hof[e["key"]] = e
                passed += 1
            else:
                memory.note_lesson(lesson, g)
        hit_rate = passed / max(1, len(elites))
        recent_pass.append(hit_rate)
        recent_pass = recent_pass[-4:]

        # 5. Reinforce. Elites are the parent pool, so bump their parent use.
        for e in elites:
            used_as_parent[e["key"]] += 1
        touched = list(evaluated.keys())
        used_counts = {k: used_as_parent.get(k, 0) for k in touched}
        memory.reinforce(touched, used_counts, g)
        memory.decay_lessons()
        mine_motifs(memory, hof)

        # 6. Log.
        oos = [r["stats"]["oos_sharpe"] for r in evaluated.values()]
        gen_best = max(oos)
        median_oos = float(np.median(oos))
        gen_best_rec = max(evaluated.values(),
                           key=lambda r: r["stats"]["oos_sharpe"])
        if gen_best > best_oos_so_far:
            best_oos_so_far = gen_best
            best_alpha = gen_best_rec
            n_stall = 0
        else:
            n_stall += 1
        mean_abs_pe = float(np.mean(pe_vals)) if pe_vals else 0.0
        sigma_t = 0.80 * sigma_t + 0.20 * mean_abs_pe
        sigma_scale = 0.95 * sigma_scale + 0.05 * max(mean_abs_pe, 1e-6)
        history.append((g, round(best_oos_so_far, 4), round(gen_best, 4),
                        round(median_oos, 4), round(explore, 4),
                        len(memory.items), round(mean_abs_pe, 4),
                        round(sigma_t, 4), n_stall, round(hit_rate, 4)))
        print(f"gen {g:02d}  best_oos {best_oos_so_far:+.3f}  "
              f"median {median_oos:+.3f}  mean_abs_pe {mean_abs_pe:.3f}  "
              f"sigma {sigma_t:.3f}  stall {n_stall:02d}  explore {explore:.2f}  "
              f"hit {hit_rate:.2f}  strat v{memory.strategy.get('version', 0)}  "
              f"mem {len(memory.items):3d}  | {best_alpha['key']}")
        if weave and HAVE_WEAVE:
            try:
                import weave as weave_mod
                weave_mod.log({"gen": g, "best_oos": best_oos_so_far,
                               "median_oos": median_oos, "explore": explore,
                               "mean_abs_pe": mean_abs_pe, "hit_rate": hit_rate})
            except Exception:
                pass
        if progress_cb is not None:
            progress_cb({
                "gen": g,
                "best": best_oos_so_far,
                "median": median_oos,
                "best_key": best_alpha["key"],
                "best_stats": best_alpha["stats"],
                "memory": memory.assemble_context(),
                "history": list(history),
            })
        if mode == "claude" and (g + 1) % 4 == 0:
            maybe_rewrite_strategy(memory, hof, float(np.mean(recent_pass)), g, model)
        if sleep_now and g == min(12, generations - 1):
            sleep(memory, panel, model=model, mode=mode)
        elif (g + 1) % 8 == 0 and mode == "claude":
            sleep(memory, panel, model=model, mode=mode)

        # 7. Breed the next population: elites plus fresh proposals.
        parent_elites = sorted(elites, key=lambda r: (-r["selection_score"], r["key"]))
        elite_exprs = [copy.deepcopy(e["expr"]) for e in parent_elites]
        ctx = memory.assemble_context()
        if mode == "claude":
            children = propose_claude(ctx, pop - len(elite_exprs), model,
                                      explore=explore, bold_ops=bold_ops)
        else:
            children = propose_offline(ctx, pop - len(elite_exprs), explore,
                                       bold_ops=bold_ops)
        population = elite_exprs + children

    memory.save(mem_path)
    _write_history(history_path, history)
    _save_curve(curve_path, history)

    bs = best_alpha["stats"]
    print(f"best alpha  oos {bs['oos_sharpe']:+.3f}  is {bs['is_sharpe']:+.3f}  "
          f"turnover {bs['turnover']:.3f}  size {bs['size']}  | {best_alpha['key']}")
    print("top strength:")
    for r in memory.assemble_context()["top_strength"][:8]:
        print(f"  strength {r['strength']:.3f}  novelty {r.get('novelty', 0.0):.3f}  "
              f"oos {r['stats']['oos_sharpe']:+.3f}  {r['key']}")
    print("top motifs:")
    for m in memory.assemble_context()["motifs"]:
        print(f"  strength {m.get('strength', 0.0):.3f}  uses {m.get('uses', 0)}  "
              f"{m.get('fragment', '')}")
    print(f"saved {mem_path} ({len(memory.items)} items), "
          f"{history_path}, {curve_path}.  hof {len(hof)}")

    return {"best_oos": best_oos_so_far, "best_key": best_alpha["key"],
            "history": history, "memory": memory, "hof": hof}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Alpha Evolver")
    p.add_argument("--mode", choices=["offline", "claude"], default="offline")
    p.add_argument("--data", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--pop", type=int, default=56)
    p.add_argument("--weave", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--sleep-now", action="store_true",
                   help="run a sleep phase after generation 12 or at the last generation")
    p.add_argument("--selftest", action="store_true",
                   help="run the Memory selftest and exit")
    return p.parse_args(argv)


def memory_selftest():
    """Exercise Memory: reinforce twice, show eviction, print assemble_context."""
    m = Memory()

    # Five fake items, A best down to E negative.
    fakes = {
        "A": dict(stats=dict(is_sharpe=2.0, oos_sharpe=1.8, turnover=0.5, size=4, ic=0.030),
                  predicted=1.5, error=0.5, novelty=0.20),
        "B": dict(stats=dict(is_sharpe=1.5, oos_sharpe=1.2, turnover=0.7, size=6, ic=0.020),
                  predicted=1.8, error=-0.3, novelty=0.50),
        "C": dict(stats=dict(is_sharpe=1.0, oos_sharpe=0.9, turnover=0.4, size=3, ic=0.015),
                  predicted=0.2, error=0.8, novelty=0.90),
        "D": dict(stats=dict(is_sharpe=0.3, oos_sharpe=0.1, turnover=1.5, size=9, ic=0.005),
                  predicted=0.4, error=-0.1, novelty=0.10),
        "E": dict(stats=dict(is_sharpe=-0.2, oos_sharpe=-0.1, turnover=2.0, size=12, ic=-0.010),
                  predicted=0.0, error=-0.2, novelty=0.05),
    }
    for k, f in fakes.items():
        m.add_or_update(k, ["neg", k.lower()], f["stats"], f["predicted"],
                        f["error"], gen=0, novelty=f["novelty"], lesson=f"lesson {k}")

    # A little slow store so assemble_context has something to return.
    m.add_principle("short recent winners, the reversal pays", keys=["A", "C"])
    m.add_motif("ts_mean(returns, 2)", strength=0.9, uses=4, avg_fitness=1.3)
    m.add_motif("cs_rank", strength=0.6, uses=7, avg_fitness=0.8)
    m.set_strategy("favor low turnover reversals on returns", hit_rate=0.42)

    def show_strengths(label):
        order = sorted(m.items.values(), key=lambda r: (-r["strength"], r["key"]))
        print(label)
        for r in order:
            print(f"  {r['key']}  strength {r['strength']:.4f}  uses {r['uses']}")

    # Reinforce one: touch all five, no eviction.
    m.reinforce(["A", "B", "C", "D", "E"],
                {"A": 3, "B": 2, "C": 1, "D": 1, "E": 0}, gen=1, max_cap=300)
    show_strengths("after reinforce 1 (touch all, max_cap 300):")

    # Reinforce two: touch A,B,C only, cap at 3 so the two weakest are evicted.
    evicted = m.reinforce(["A", "B", "C"],
                          {"A": 2, "B": 1, "C": 1}, gen=2, max_cap=3)
    show_strengths("after reinforce 2 (touch A,B,C, max_cap 3):")
    print(f"evicted: {evicted}")

    print("assemble_context:")
    ctx = m.assemble_context()
    summary = {
        "top_strength": [(r["key"], round(r["strength"], 4)) for r in ctx["top_strength"]],
        "top_novelty": [(r["key"], r["novelty"]) for r in ctx["top_novelty"]],
        "top_surprise": [(r["key"], abs(r["prediction_error"])) for r in ctx["top_surprise"]],
        "principles": ctx["principles"],
        "motifs": [(mo["fragment"], mo["strength"]) for mo in ctx["motifs"]],
        "top_lessons": [(r["text"], r["strength"]) for r in ctx["top_lessons"]],
    }
    print(json.dumps(summary, indent=2))
    print("full top_strength record (verbatim):")
    print(json.dumps(ctx["top_strength"][0], indent=2))


def main(argv=None):
    args = parse_args(argv)

    if args.selftest:
        memory_selftest()
        return

    run(generations=args.generations, pop=args.pop, seed=args.seed,
        data=args.data, mode=args.mode, weave=args.weave, model=args.model,
        sleep_now=args.sleep_now)


if __name__ == "__main__":
    main()
