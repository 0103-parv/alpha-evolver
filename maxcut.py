"""Max Cut creativity laboratory (step 1): graph generators, a sandboxed heuristic
DSL, a bounded local search executor, and a verifier with known graph self tests.

A heuristic is a small program:
    init  expression  scores vertices to seed the partition (static features only)
    move  expression  scores vertex flips during local search (all features)
    steps_per_node in {1, 2, 4, 8}, restarts in {1, 2, 4}, tabu in {0, 1, 3, 5}

Each step the executor flips the highest scoring eligible vertex, allows
temporarily harmful moves, and always preserves the best cut seen. Total work is
bounded by steps_per_node * n * restarts, so execution always halts. The domain
contract and the evolution engine wiring come in later steps. Offline numpy only.
"""
import argparse
import math

import numpy as np


class MaxCutError(Exception):
    """A program or graph is malformed. The executor fails closed by raising."""


# ---------------------------------------------------------------------------
# Graphs: weighted, undirected, a symmetric matrix with zero diagonal.
# ---------------------------------------------------------------------------

def make_graph(n, edges, family="custom", seed=0):
    if n < 0:
        raise MaxCutError(f"bad vertex count {n}")
    W = np.zeros((n, n), dtype=float)
    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise MaxCutError(f"edge out of range: {(u, v, w)}")
        if u == v:
            continue
        W[u, v] = w
        W[v, u] = w
    return {"n": n, "W": W, "family": family, "seed": seed,
            "m": int((W > 0).sum() // 2)}


def validate_graph(g):
    if not all(k in g for k in ("n", "W")):
        raise MaxCutError("graph missing fields")
    W = g["W"]
    if W.shape != (g["n"], g["n"]):
        raise MaxCutError("graph shape mismatch")
    if not np.isfinite(W).all():
        raise MaxCutError("graph has non finite weights")
    if not np.allclose(W, W.T):
        raise MaxCutError("graph not symmetric")
    if np.any(np.diag(W) != 0):
        raise MaxCutError("graph has self loops")
    return True


def cut_value(g, side):
    """Total weight of edges crossing the partition. The ground truth verifier."""
    W = g["W"]
    side = np.asarray(side)
    diff = side[:, None] != side[None, :]
    return float((W * diff).sum() / 2.0)


# ---- generators (deterministic given a seed) ------------------------------

def er_random(n, seed, avg_deg=4.0, wlo=1.0, whi=1.0):
    rng = np.random.default_rng(seed)
    p = min(1.0, avg_deg / max(n - 1, 1))
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < p
    w = rng.uniform(wlo, whi, len(iu[0]))
    edges = [(int(iu[0][k]), int(iu[1][k]), float(w[k]))
             for k in range(len(iu[0])) if keep[k]]
    fam = "er_dense" if avg_deg >= 0.25 * n else "er_sparse"
    return make_graph(n, edges, fam, seed)


def community(n, seed, k=4, p_in=0.45, p_out=0.04):
    rng = np.random.default_rng(seed)
    block = rng.integers(0, k, n)
    iu = np.triu_indices(n, 1)
    same = block[iu[0]] == block[iu[1]]
    p = np.where(same, p_in, p_out)
    keep = rng.random(len(iu[0])) < p
    w = rng.uniform(1.0, 2.0, len(iu[0]))
    edges = [(int(iu[0][k]), int(iu[1][k]), float(w[k]))
             for k in range(len(iu[0])) if keep[k]]
    return make_graph(n, edges, "community", seed)


def geometric(n, seed, radius=None):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    if radius is None:
        radius = math.sqrt(3.0 / max(n, 1))
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                 pts[:, None, 1] - pts[None, :, 1])
    iu = np.triu_indices(n, 1)
    keep = d[iu] < radius
    edges = [(int(iu[0][k]), int(iu[1][k]), 1.0)
             for k in range(len(iu[0])) if keep[k]]
    return make_graph(n, edges, "geometric", seed)


def near_bipartite(n, seed, cross_p=0.4, noise_p=0.03):
    rng = np.random.default_rng(seed)
    part = rng.integers(0, 2, n)
    iu = np.triu_indices(n, 1)
    cross = part[iu[0]] != part[iu[1]]
    p = np.where(cross, cross_p, noise_p)
    keep = rng.random(len(iu[0])) < p
    edges = [(int(iu[0][k]), int(iu[1][k]), 1.0)
             for k in range(len(iu[0])) if keep[k]]
    return make_graph(n, edges, "near_bipartite", seed)


def regular(n, seed, d=3):
    rng = np.random.default_rng(seed)
    stubs = list(range(n)) * d
    rng.shuffle(stubs)
    seen = {}
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v:
            seen[(min(u, v), max(u, v))] = 1.0
    edges = [(u, v, w) for (u, v), w in seen.items()]
    return make_graph(n, edges, "regular", seed)


def heavy_tail(n, seed):
    rng = np.random.default_rng(seed)
    edges, targets = [], [0]
    for u in range(1, n):
        chosen = set()
        while len(chosen) < min(2, u):
            chosen.add(int(targets[rng.integers(0, len(targets))]))
        for v in chosen:
            edges.append((u, v, float(rng.uniform(1.0, 3.0))))
            targets.extend([u, v])
    return make_graph(n, edges, "heavy_tail", seed)


_GENS = {"er_sparse": lambda n, s: er_random(n, s, avg_deg=3.0),
         "er_dense": lambda n, s: er_random(n, s, avg_deg=max(4.0, 0.3 * n)),
         "community": community, "geometric": geometric,
         "near_bipartite": near_bipartite, "regular": regular,
         "heavy_tail": heavy_tail}
_BASE = ["er_sparse", "er_dense", "community", "geometric", "near_bipartite"]


def build_suite(kind):
    """Deterministic graph suites. Train and dev share families; test withholds
    regular and heavy tail families and uses larger sizes."""
    out = []
    if kind == "train":
        for i in range(20):
            fam = _BASE[i % 5]
            out.append(_GENS[fam](40 + (i // 5) * 6, 1000 + i))
    elif kind == "dev":
        for i in range(10):
            fam = _BASE[i % 5]
            out.append(_GENS[fam](72 + (i // 5) * 6, 2000 + i))
    elif kind == "test":
        fams = _BASE + ["regular", "heavy_tail"]
        for i in range(14):
            fam = fams[i % 7]
            out.append(_GENS[fam](64 + (i // 7) * 8, 3000 + i))
    elif kind == "replay":
        for i in range(8):
            fam = _BASE[i % 5]
            out.append(_GENS[fam](48 + (i // 5) * 6, 4000 + i))
    else:
        raise MaxCutError(f"unknown suite {kind}")
    return out


# ---------------------------------------------------------------------------
# The heuristic DSL.
# ---------------------------------------------------------------------------

STATIC_FEATURES = ("degree", "weighted_degree", "neighbor_degree", "triangle_score")
DYNAMIC_FEATURES = ("flip_gain", "same_side_weight", "cross_side_weight",
                    "side", "flip_age", "step_progress")
CONSTS = {"k0": 0.0, "k1": 1.0, "k2": 2.0, "khalf": 0.5, "kneg1": -1.0}
UNARY = ("neg", "abs", "rank", "zscore", "norm")
BINARY = ("add", "sub", "mul", "sdiv", "min", "max")

INIT_LEAVES = STATIC_FEATURES + tuple(CONSTS)
MOVE_LEAVES = STATIC_FEATURES + DYNAMIC_FEATURES + tuple(CONSTS)
MAX_DEPTH = 5
STEPS_CHOICES = (1, 2, 4, 8)
RESTART_CHOICES = (1, 2, 4)
TABU_CHOICES = (0, 1, 3, 5)


def valid(node, leaves, depth=0):
    if depth > MAX_DEPTH:
        return False
    if isinstance(node, str):
        return node in leaves or node in CONSTS
    if not isinstance(node, list) or not node:
        return False
    op = node[0]
    if op in UNARY:
        return len(node) == 2 and valid(node[1], leaves, depth + 1)
    if op in BINARY:
        return (len(node) == 3 and valid(node[1], leaves, depth + 1)
                and valid(node[2], leaves, depth + 1))
    return False


def complexity(node):
    if isinstance(node, str):
        return 1
    return 1 + sum(complexity(c) for c in node[1:])


def validate_program(p):
    if not isinstance(p, dict):
        raise MaxCutError("program is not a dict")
    for key in ("init", "move", "steps_per_node", "restarts", "tabu"):
        if key not in p:
            raise MaxCutError(f"program missing {key}")
    if not valid(p["init"], INIT_LEAVES):
        raise MaxCutError("invalid init expression (static features only)")
    if not valid(p["move"], MOVE_LEAVES):
        raise MaxCutError("invalid move expression")
    if p["steps_per_node"] not in STEPS_CHOICES:
        raise MaxCutError("invalid steps_per_node")
    if p["restarts"] not in RESTART_CHOICES:
        raise MaxCutError("invalid restarts")
    if p["tabu"] not in TABU_CHOICES:
        raise MaxCutError("invalid tabu window")
    return True


def _rank(a):
    n = len(a)
    return np.argsort(np.argsort(a)).astype(float) / (n - 1 if n > 1 else 1)


def _ev(node, feats, n):
    if isinstance(node, str):
        if node in CONSTS:
            return np.full(n, CONSTS[node], dtype=float)
        if node in feats:
            return np.asarray(feats[node], dtype=float)
        raise MaxCutError(f"unknown leaf {node}")
    op = node[0]
    if op in UNARY:
        a = _ev(node[1], feats, n)
        if op == "neg":
            return -a
        if op == "abs":
            return np.abs(a)
        if op == "rank":
            return _rank(a)
        if op == "zscore":
            s = a.std()
            return (a - a.mean()) / (s if s > 1e-9 else 1.0)
        if op == "norm":
            lo, hi = a.min(), a.max()
            return (a - lo) / ((hi - lo) if hi - lo > 1e-9 else 1.0)
    if op in BINARY:
        a = _ev(node[1], feats, n)
        b = _ev(node[2], feats, n)
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "sdiv":
            return a / np.where(np.abs(b) < 1e-6, 1e-6, b)
        if op == "min":
            return np.minimum(a, b)
        if op == "max":
            return np.maximum(a, b)
    raise MaxCutError(f"bad op {op}")


def evaluate(node, feats, n):
    with np.errstate(all="ignore"):
        val = _ev(node, feats, n)
    return np.nan_to_num(np.clip(val, -1e9, 1e9), nan=0.0)


def random_expr(rng, leaves, depth):
    if depth <= 1 or rng.random() < 0.35:
        pool = tuple(leaves) + tuple(CONSTS)
        return str(pool[rng.integers(0, len(pool))])
    if rng.random() < 0.4:
        return [UNARY[rng.integers(0, len(UNARY))], random_expr(rng, leaves, depth - 1)]
    return [BINARY[rng.integers(0, len(BINARY))],
            random_expr(rng, leaves, depth - 1), random_expr(rng, leaves, depth - 1)]


def random_program(rng):
    return {"init": random_expr(rng, STATIC_FEATURES, 3),
            "move": random_expr(rng, STATIC_FEATURES + DYNAMIC_FEATURES, 4),
            "steps_per_node": int(STEPS_CHOICES[rng.integers(0, 4)]),
            "restarts": int(RESTART_CHOICES[rng.integers(0, 3)]),
            "tabu": int(TABU_CHOICES[rng.integers(0, 4)])}


# ---------------------------------------------------------------------------
# Static features and the bounded executor.
# ---------------------------------------------------------------------------

def static_features(g):
    W = g["W"]
    A = (W > 0).astype(float)
    deg = A.sum(1)
    wdeg = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        neigh = np.where(deg > 0, (A @ deg) / np.where(deg > 0, deg, 1.0), 0.0)
    tri = 0.5 * (A * (A @ A)).sum(1)
    feats = {"degree": deg, "weighted_degree": wdeg,
             "neighbor_degree": neigh, "triangle_score": tri}
    return {k: np.nan_to_num(v.astype(float)) for k, v in feats.items()}


def _init_partition(init_node, static, n):
    score = evaluate(init_node, static, n)
    side = (score >= np.median(score)).astype(np.int64)
    if side.min() == side.max():            # degenerate, give the search a seam
        side = (np.arange(n) % 2).astype(np.int64)
    return side


# A simple greedy reference: flip the highest gain vertex.
GREEDY = {"init": "weighted_degree", "move": "flip_gain",
          "steps_per_node": 8, "restarts": 4, "tabu": 0}


def run_heuristic(g, program, seed=0):
    """Return the best cut found. Bounded and fail closed."""
    validate_graph(g)
    validate_program(program)
    n, W = g["n"], g["W"]
    if n == 0:
        return 0.0
    static = static_features(g)
    neighbors = [np.nonzero(W[v])[0] for v in range(n)]
    rng = np.random.default_rng(seed)
    best = 0.0
    for r in range(program["restarts"]):
        if r == 0:
            side = _init_partition(program["init"], static, n)
        else:
            side = rng.integers(0, 2, n).astype(np.int64)
        same = side[:, None] == side[None, :]
        same_w = (W * same).sum(1)
        cross_w = (W * ~same).sum(1)
        cut = float(cross_w.sum() / 2.0)
        best = max(best, cut)
        last_flip = np.full(n, -(10 ** 9), dtype=np.int64)
        total = program["steps_per_node"] * n
        tabu = program["tabu"]
        for step in range(total):
            feats = dict(static)
            feats["flip_gain"] = same_w - cross_w
            feats["same_side_weight"] = same_w
            feats["cross_side_weight"] = cross_w
            feats["side"] = side.astype(float)
            feats["flip_age"] = np.minimum((step - last_flip).astype(float), 1e6)
            feats["step_progress"] = np.full(n, step / max(total, 1), dtype=float)
            score = evaluate(program["move"], feats, n)
            eligible = (step - last_flip) > tabu
            if not eligible.any():
                break
            score = np.where(eligible, score, -np.inf)
            v = int(np.argmax(score))
            gain = same_w[v] - cross_w[v]
            sv = side[v]
            for u in neighbors[v]:
                w = W[v, u]
                if side[u] == sv:
                    same_w[u] -= w
                    cross_w[u] += w
                else:
                    cross_w[u] -= w
                    same_w[u] += w
            same_w[v], cross_w[v] = cross_w[v], same_w[v]
            side[v] = 1 - sv
            cut += gain
            if cut > best:
                best = cut
    return best


# ---------------------------------------------------------------------------
# Self tests on known graphs.
# ---------------------------------------------------------------------------

def selftest():
    empty = make_graph(4, [])
    single = make_graph(2, [(0, 1, 1.0)])
    triangle = make_graph(3, [(0, 1, 1), (0, 2, 1), (1, 2, 1)])
    k4 = make_graph(4, [(i, j, 1) for i in range(4) for j in range(i + 1, 4)])
    bip = make_graph(6, [(u, v, 1) for u in range(3) for v in range(3, 6)])

    assert cut_value(empty, [0, 0, 0, 0]) == 0.0
    assert cut_value(single, [0, 1]) == 1.0
    assert cut_value(triangle, [0, 1, 1]) == 2.0
    assert cut_value(k4, [0, 0, 1, 1]) == 4.0
    assert cut_value(bip, [0, 0, 0, 1, 1, 1]) == 9.0
    print("verifier: empty 0, single 1, triangle 2, K4 4, bipartite 9  ok")

    for g, opt, name in [(triangle, 2, "triangle"), (k4, 4, "K4"),
                         (bip, 9, "bipartite (K3,3)"), (single, 1, "single edge")]:
        got = run_heuristic(g, GREEDY, seed=0)
        assert got == opt, f"{name}: greedy got {got}, optimum {opt}"
    print("executor greedy reaches the optimum on single, triangle, K4, bipartite  ok")

    bad_programs = [
        {"init": "flip_gain", "move": "flip_gain", "steps_per_node": 4,
         "restarts": 1, "tabu": 0},                       # dynamic feature in init
        {"init": "degree", "move": "not_a_feature", "steps_per_node": 4,
         "restarts": 1, "tabu": 0},                       # unknown leaf
        {"init": "degree", "move": "flip_gain", "steps_per_node": 3,
         "restarts": 1, "tabu": 0},                       # invalid steps
        {"init": "degree", "move": "flip_gain", "steps_per_node": 4,
         "restarts": 1, "tabu": 7},                       # invalid tabu
    ]
    for bad in bad_programs:
        try:
            run_heuristic(triangle, bad)
            raise AssertionError(f"program did not fail closed: {bad}")
        except MaxCutError:
            pass
    asym = make_graph(2, [(0, 1, 1.0)])
    asym["W"][0, 1] = 5.0                                 # break symmetry
    try:
        run_heuristic(asym, GREEDY)
        raise AssertionError("asymmetric graph did not fail closed")
    except MaxCutError:
        pass
    print("fail closed: dynamic feature in init, unknown op, bad params, bad graph  ok")
    print("maxcut selftest passed")


def demo():
    rng = np.random.default_rng(0)
    print("greedy vs best of 8 random heuristics, by family:")
    for fam in _BASE + ["regular", "heavy_tail"]:
        g = _GENS[fam](48, 7)
        greedy = run_heuristic(g, GREEDY)
        rand = max(run_heuristic(g, random_program(rng), seed=s) for s in range(8))
        opt_bound = g["W"].sum() / 2.0                    # cut <= total weight
        print(f"  {fam:15s} n={g['n']:3d} m={g['m']:4d}  greedy {greedy:7.1f}  "
              f"rand {rand:7.1f}  (<= {opt_bound:7.1f})")
    print("suites:")
    for kind in ("train", "dev", "test", "replay"):
        s = build_suite(kind)
        fams = sorted({gg["family"] for gg in s})
        print(f"  {kind:7s} {len(s):2d} graphs  families {fams}")


def main():
    p = argparse.ArgumentParser(description="Max Cut laboratory (step 1)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
    if args.demo:
        demo()
    if not (args.selftest or args.demo):
        selftest()
        demo()


if __name__ == "__main__":
    main()
