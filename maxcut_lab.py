"""Max Cut creativity laboratory for Alpha Evolver.

The offline path uses only numpy and the standard library. Programs describe
reusable Max Cut heuristics. They never contain a graph answer.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from typing import Protocol

import numpy as np


STATIC_FIELDS = (
    "degree",
    "weighted_degree",
    "neighbor_degree",
    "triangle_score",
)
DYNAMIC_FIELDS = (
    "flip_gain",
    "same_weight",
    "cross_weight",
    "side",
    "flip_age",
    "step_progress",
)
CONSTANTS = ("c_neg1", "c_neg05", "c_0", "c_05", "c_1", "c_2")
FIELDS = STATIC_FIELDS + DYNAMIC_FIELDS + CONSTANTS
UNARY = ("neg", "abs", "sign", "square", "rank", "zscore")
BINARY = ("add", "sub", "mul", "safe_div", "min", "max")
STEPS_PER_NODE = (1, 2, 4, 8)
RESTARTS = (1, 2, 4)
TABU_WINDOWS = (0, 1, 3, 5)
MAX_EXPR_DEPTH = 5
MAX_EXPR_SIZE = 50
MAX_TOTAL_STEPS = 4096
VERIFIER_VERSION = "maxcut-v1"
HARD_MAX_GENERATIONS = 250
HARD_MAX_MINUTES = 90
MAX_MEMORY_EPISODES = 5000
MAX_MEMORY_ITEMS = 600

GRAPH_VARIANTS = {
    "ours": {"surprise": "none", "novelty": True, "quarantine": True},
    "plain": {"surprise": "none", "novelty": False, "quarantine": False},
    "surprise_noquar": {
        "surprise": "inverted_u", "novelty": True, "quarantine": False},
    "quarantine_only": {"surprise": "none", "novelty": True, "quarantine": True},
    "quar_mono": {"surprise": "monotonic", "novelty": True, "quarantine": True},
    "inverted_u": {"surprise": "inverted_u", "novelty": True, "quarantine": True},
}
COGNITIVE_MODES = ("dream", "focus", "challenge", "recover", "sleep")
CONTROLLER_FEATURES = (
    "bias",
    "train_improvement",
    "dev_improvement",
    "calibration_improvement",
    "novel_trusted_rate",
    "diversity",
    "duplicate_rate",
    "quarantine_rate",
    "contradiction_rate",
    "compute_fraction",
)


def graph_from_edges(n, edges, graph_id="manual", family="manual"):
    """Build and validate an undirected weighted graph."""
    if type(n) is not int or n < 0:
        raise ValueError("graph node count must be a nonnegative integer")
    w = np.zeros((n, n), dtype=float)
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) not in (2, 3):
            raise ValueError("each edge must be (u, v) or (u, v, weight)")
        u, v = edge[:2]
        weight = 1.0 if len(edge) == 2 else float(edge[2])
        if type(u) is not int or type(v) is not int or not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge endpoint is outside graph")
        if u == v or not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("edges require distinct endpoints and positive finite weights")
        w[u, v] += weight
        w[v, u] += weight
    return validate_graph({
        "id": str(graph_id),
        "family": str(family),
        "weights": w,
    })


def validate_graph(graph):
    """Reject malformed graphs before a program can observe them."""
    if not isinstance(graph, dict) or "weights" not in graph:
        raise ValueError("graph requires a weight matrix")
    w = np.asarray(graph["weights"], dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weight matrix must be square")
    if w.shape[0] > 512:
        raise ValueError("graph exceeds the node limit")
    if not np.isfinite(w).all() or (w < 0.0).any():
        raise ValueError("weights must be finite and nonnegative")
    if not np.allclose(w, w.T) or not np.allclose(np.diag(w), 0.0):
        raise ValueError("graph must be undirected with no self edges")
    return {
        "id": str(graph.get("id", "graph")),
        "family": str(graph.get("family", "unknown")),
        "weights": w.copy(),
    }


def graph_fingerprint(graph):
    """Stable evidence identifier for train and development graphs."""
    g = validate_graph(graph)
    payload = g["weights"].tobytes() + g["family"].encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def graph_context(graph, include_fingerprint=True):
    """Observable graph context used for scoped memory."""
    g = validate_graph(graph)
    w = g["weights"]
    n = w.shape[0]
    possible = max(1, n * (n - 1) / 2)
    upper = float(np.triu(w, 1).sum())
    positive = w[w > 0.0]
    context = {
        "graph_id": g["id"],
        "family": g["family"],
        "nodes": int(n),
        "density": float(np.count_nonzero(np.triu(w, 1)) / possible),
        "weight_regime": "unit" if positive.size and np.allclose(positive, 1.0) else "weighted",
        "total_weight": upper,
    }
    if include_fingerprint:
        context["fingerprint"] = graph_fingerprint(g)
    return context


def _weighted_graph(rng, n, mask, family, graph_id):
    weights = rng.uniform(0.5, 1.5, size=(n, n))
    w = np.where(mask, weights, 0.0)
    w = np.triu(w, 1)
    w += w.T
    return validate_graph({"id": graph_id, "family": family, "weights": w})


def _random_graph(rng, n, p, family, graph_id):
    mask = rng.random((n, n)) < p
    return _weighted_graph(rng, n, mask, family, graph_id)


def _community_graph(rng, n, graph_id):
    groups = np.arange(n) % max(2, min(4, n // 8))
    same = groups[:, None] == groups[None, :]
    p = np.where(same, 0.38, 0.06)
    return _weighted_graph(rng, n, rng.random((n, n)) < p, "community", graph_id)


def _geometric_graph(rng, n, graph_id):
    xy = rng.random((n, 2))
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    radius = 0.28 if n <= 48 else 0.22
    return _weighted_graph(rng, n, dist < radius, "geometric", graph_id)


def _near_bipartite_graph(rng, n, graph_id):
    side = np.arange(n) % 2
    cross = side[:, None] != side[None, :]
    p = np.where(cross, 0.34, 0.025)
    return _weighted_graph(rng, n, rng.random((n, n)) < p, "near_bipartite", graph_id)


def _regular_graph(rng, n, graph_id):
    mask = np.zeros((n, n), dtype=bool)
    offsets = (1, 2, 3)
    for i in range(n):
        for offset in offsets:
            j = (i + offset) % n
            mask[i, j] = True
            mask[j, i] = True
    return _weighted_graph(rng, n, mask, "regular", graph_id)


def _heavy_tail_graph(rng, n, graph_id):
    mask = np.zeros((n, n), dtype=bool)
    degree = np.ones(n, dtype=float)
    for i in range(1, n):
        k = min(3, i)
        probs = degree[:i] / degree[:i].sum()
        chosen = rng.choice(i, size=k, replace=False, p=probs)
        for j in chosen:
            mask[i, j] = True
            mask[j, i] = True
            degree[i] += 1.0
            degree[j] += 1.0
    return _weighted_graph(rng, n, mask, "heavy_tail", graph_id)


def build_graph_suite(kind="train", seed=7):
    """Build deterministic train, development, sealed, or replay suites."""
    rng = np.random.default_rng(seed + {
        "train": 0,
        "dev": 10000,
        "sealed": 20000,
        "replay": 30000,
    }[kind])
    graphs = []
    if kind == "train":
        families = ("sparse", "dense", "community", "geometric", "near_bipartite")
        counts, sizes = 4, (20, 24, 28, 32)
    elif kind == "dev":
        families = ("sparse", "dense", "community", "geometric", "near_bipartite")
        counts, sizes = 2, (40, 48)
    elif kind == "sealed":
        families = ("regular", "heavy_tail")
        counts, sizes = 7, (52, 56, 60, 64, 68, 72, 76)
    elif kind == "replay":
        families = ("sparse", "community", "geometric", "near_bipartite")
        counts, sizes = 2, (32, 40)
    else:
        raise ValueError(f"unknown graph suite {kind}")
    for family in families:
        for i in range(counts):
            n = sizes[i % len(sizes)]
            graph_id = f"{kind}-{family}-{i}"
            if family == "sparse":
                graph = _random_graph(rng, n, 0.10, family, graph_id)
            elif family == "dense":
                graph = _random_graph(rng, n, 0.42, family, graph_id)
            elif family == "community":
                graph = _community_graph(rng, n, graph_id)
            elif family == "geometric":
                graph = _geometric_graph(rng, n, graph_id)
            elif family == "near_bipartite":
                graph = _near_bipartite_graph(rng, n, graph_id)
            elif family == "regular":
                graph = _regular_graph(rng, n, graph_id)
            else:
                graph = _heavy_tail_graph(rng, n, graph_id)
            graphs.append(graph)
    return graphs


def valid_expr(node, depth=0, allowed_fields=None):
    """True only when an expression is inside the Max Cut grammar."""
    if depth > MAX_EXPR_DEPTH:
        return False
    if isinstance(node, str):
        return node in (FIELDS if allowed_fields is None else allowed_fields)
    if not isinstance(node, (list, tuple)) or not node or not isinstance(node[0], str):
        return False
    if node[0] in UNARY:
        return len(node) == 2 and valid_expr(node[1], depth + 1, allowed_fields)
    if node[0] in BINARY:
        return (len(node) == 3 and valid_expr(node[1], depth + 1, allowed_fields)
                and valid_expr(node[2], depth + 1, allowed_fields))
    return False


def expr_size(node):
    if isinstance(node, str):
        return 1
    return 1 + sum(expr_size(child) for child in node[1:])


def expr_to_str(node):
    if isinstance(node, str):
        return node
    return f"{node[0]}(" + ",".join(expr_to_str(child) for child in node[1:]) + ")"


def valid_program(program):
    """Reject malformed or excessively expensive heuristic programs."""
    try:
        return (
            isinstance(program, dict)
            and set(program) == {"init", "move", "steps_per_node", "restarts", "tabu_window"}
            and valid_expr(program["init"], allowed_fields=STATIC_FIELDS + CONSTANTS)
            and valid_expr(program["move"])
            and expr_size(program["init"]) + expr_size(program["move"]) <= MAX_EXPR_SIZE
            and program["steps_per_node"] in STEPS_PER_NODE
            and program["restarts"] in RESTARTS
            and program["tabu_window"] in TABU_WINDOWS
        )
    except Exception:
        return False


def program_key(program):
    if not valid_program(program):
        raise ValueError("invalid Max Cut program")
    canonical = {
        "init": expr_to_str(program["init"]),
        "move": expr_to_str(program["move"]),
        "steps_per_node": program["steps_per_node"],
        "restarts": program["restarts"],
        "tabu_window": program["tabu_window"],
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def program_size(program):
    return expr_size(program["init"]) + expr_size(program["move"]) + 3


def _rank(x):
    if x.size <= 1:
        return np.full_like(x, 0.5, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    return ranks / (x.size - 1)


def eval_expr(node, features):
    """Evaluate a safe expression to one score per vertex."""
    if isinstance(node, str):
        return np.asarray(features[node], dtype=float)
    op = node[0]
    if op in UNARY:
        x = eval_expr(node[1], features)
        if op == "neg":
            return -x
        if op == "abs":
            return np.abs(x)
        if op == "sign":
            return np.sign(x)
        if op == "square":
            return np.clip(x, -1e3, 1e3) ** 2
        if op == "rank":
            return _rank(x)
        sd = float(np.std(x))
        return np.zeros_like(x) if sd <= 1e-12 else (x - float(np.mean(x))) / sd
    a = eval_expr(node[1], features)
    b = eval_expr(node[2], features)
    if op == "add":
        out = a + b
    elif op == "sub":
        out = a - b
    elif op == "mul":
        out = a * b
    elif op == "min":
        out = np.minimum(a, b)
    elif op == "max":
        out = np.maximum(a, b)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.abs(b) > 1e-9, a / b, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def _static_features(graph):
    w = graph["weights"]
    adjacent = (w > 0.0).astype(float)
    degree = adjacent.sum(axis=1)
    weighted_degree = w.sum(axis=1)
    safe_degree = np.where(degree > 0.0, degree, 1.0)
    neighbor_degree = adjacent @ degree / safe_degree
    triangle_score = np.diag(adjacent @ adjacent @ adjacent) / 2.0
    return {
        "degree": degree,
        "weighted_degree": weighted_degree,
        "neighbor_degree": neighbor_degree,
        "triangle_score": triangle_score,
    }


def _features(graph, static, side, flip_age, step, total_steps,
              same_weight=None, cross_weight=None):
    w = graph["weights"]
    if same_weight is None or cross_weight is None:
        same = side[:, None] == side[None, :]
        same_weight = (w * same).sum(axis=1)
        cross_weight = (w * ~same).sum(axis=1)
    n = side.size
    features = dict(static)
    features.update({
        "flip_gain": same_weight - cross_weight,
        "same_weight": same_weight,
        "cross_weight": cross_weight,
        "side": side.astype(float),
        "flip_age": flip_age.astype(float),
        "step_progress": np.full(n, step / max(1, total_steps), dtype=float),
        "c_neg1": np.full(n, -1.0),
        "c_neg05": np.full(n, -0.5),
        "c_0": np.zeros(n),
        "c_05": np.full(n, 0.5),
        "c_1": np.ones(n),
        "c_2": np.full(n, 2.0),
    })
    return features


def cut_weight(graph, side):
    """Exact weight of the cut represented by a boolean side vector."""
    g = validate_graph(graph)
    side = np.asarray(side, dtype=bool)
    if side.shape != (g["weights"].shape[0],):
        raise ValueError("partition size does not match graph")
    return float(np.triu(g["weights"] * (side[:, None] != side[None, :]), 1).sum())


def execute_program(program, graph):
    """Run a bounded reusable heuristic and return its best observed cut."""
    if not valid_program(program):
        raise ValueError("invalid Max Cut program")
    g = validate_graph(graph)
    n = g["weights"].shape[0]
    if n == 0:
        return {"cut": 0.0, "quality": 1.0, "side": np.zeros(0, dtype=bool), "steps": 0}
    requested_steps = program["steps_per_node"] * n
    total_steps = min(requested_steps, MAX_TOTAL_STEPS)
    key = program_key(program)
    best_cut = -1.0
    best_side = np.zeros(n, dtype=bool)
    static = _static_features(g)
    for restart in range(program["restarts"]):
        seed_bytes = hashlib.sha256(f"{g['id']}|{key}|{restart}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "little"))
        blank_side = np.zeros(n, dtype=bool)
        init_features = _features(g, static, blank_side, np.full(n, n), 0, total_steps)
        init_score = eval_expr(program["init"], init_features)
        if restart:
            init_score = init_score + rng.normal(0.0, 0.10, size=n)
        side = init_score >= float(np.median(init_score))
        if side.all() or not side.any():
            side = rng.random(n) < 0.5
        flip_age = np.full(n, n + program["tabu_window"], dtype=int)
        current = cut_weight(g, side)
        same = side[:, None] == side[None, :]
        same_weight = (g["weights"] * same).sum(axis=1)
        cross_weight = (g["weights"] * ~same).sum(axis=1)
        if current > best_cut:
            best_cut, best_side = current, side.copy()
        for step in range(total_steps):
            features = _features(
                g, static, side, flip_age, step, total_steps,
                same_weight=same_weight, cross_weight=cross_weight)
            score = eval_expr(program["move"], features)
            eligible = flip_age > program["tabu_window"]
            if not eligible.any():
                flip_age += 1
                continue
            score = np.where(eligible, score, -np.inf)
            vertex = int(np.argmax(score))
            old_side = bool(side[vertex])
            gain = float(same_weight[vertex] - cross_weight[vertex])
            weights = g["weights"][vertex]
            were_same = side == old_side
            were_cross = ~were_same
            same_weight[were_same] -= weights[were_same]
            cross_weight[were_same] += weights[were_same]
            same_weight[were_cross] += weights[were_cross]
            cross_weight[were_cross] -= weights[were_cross]
            same_weight[vertex], cross_weight[vertex] = (
                cross_weight[vertex], same_weight[vertex])
            side[vertex] = ~side[vertex]
            flip_age += 1
            flip_age[vertex] = 0
            current += gain
            if current > best_cut:
                best_cut, best_side = current, side.copy()
    upper = float(np.triu(g["weights"], 1).sum())
    quality = 1.0 if upper <= 1e-12 else best_cut / upper
    return {
        "cut": float(best_cut),
        "quality": float(quality),
        "side": best_side,
        "steps": int(total_steps * program["restarts"]),
    }


def baseline_program():
    """Simple gain driven local search used by tests and comparisons."""
    return {
        "init": ["rank", "degree"],
        "move": "flip_gain",
        "steps_per_node": 4,
        "restarts": 2,
        "tabu_window": 1,
    }


class DomainContract(Protocol):
    """Minimum interface for a creativity laboratory."""

    name: str
    verifier_version: str

    def valid(self, candidate): ...
    def key(self, candidate): ...
    def fresh(self, preferred_ops=None): ...
    def mutate(self, candidate, preferred_ops=None): ...
    def crossover(self, left, right): ...
    def evaluate_train(self, candidate): ...
    def evaluate_dev(self, candidate): ...
    def stress_verify(self, train, dev, error, pe_scale, quarantine): ...
    def complexity(self, candidate): ...
    def novelty(self, candidate, records): ...
    def evaluate_sealed(self, candidate): ...
    def context(self): ...


class MaxCutDomain:
    """Domain contract consumed by the graph evolution engine."""

    name = "maxcut"
    verifier_version = VERIFIER_VERSION

    def __init__(self, seed=7, suite_kind="standard"):
        self.seed = seed
        self.train = build_graph_suite("train" if suite_kind == "standard" else "replay", seed)
        self.dev = build_graph_suite("dev" if suite_kind == "standard" else "replay", seed + 1)
        self.sealed = build_graph_suite("sealed", seed) if suite_kind == "standard" else []

    def valid(self, candidate):
        return valid_program(candidate)

    def key(self, candidate):
        return program_key(candidate)

    def fresh(self, preferred_ops=None):
        return random_program(preferred_ops=preferred_ops)

    def mutate(self, candidate, preferred_ops=None):
        return mutate_program(candidate, preferred_ops=preferred_ops)

    def crossover(self, left, right):
        return crossover_program(left, right)

    def evaluate_train(self, candidate):
        return evaluate_program(candidate, self.train)

    def evaluate_dev(self, candidate):
        return evaluate_program(candidate, self.dev)

    def stress_verify(self, train, dev, error, pe_scale, quarantine):
        return _graph_candidate_status(train, dev, error, pe_scale, quarantine)

    def complexity(self, candidate):
        return program_size(candidate)

    def novelty(self, candidate, records):
        return program_novelty(candidate, records)

    def evaluate_sealed(self, candidate):
        return evaluate_program(candidate, self.sealed) if self.sealed else None

    def context(self):
        contexts = [graph_context(graph, include_fingerprint=False) for graph in self.train]
        return {
            "families": sorted({g["family"] for g in self.train}),
            "nodes_min": min(g["weights"].shape[0] for g in self.train),
            "nodes_max": max(g["weights"].shape[0] for g in self.train),
            "density_min": min(context["density"] for context in contexts),
            "density_max": max(context["density"] for context in contexts),
            "weight_regime": "weighted",
        }

    def evidence_context(self):
        return {
            "summary": self.context(),
            "train_fingerprints": [graph_fingerprint(graph) for graph in self.train],
            "dev_fingerprints": [graph_fingerprint(graph) for graph in self.dev],
        }


def _contains_sealed(value):
    if isinstance(value, str):
        return "sealed" in value.lower()
    if isinstance(value, dict):
        return any(_contains_sealed(key) or _contains_sealed(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_sealed(item) for item in value)
    return False


def _context_match_score(stored_contexts, context):
    """Match memory using family, size, density, and weight regime."""
    families = set(context.get("families", []))
    scores = []
    for stored in stored_contexts:
        stored_families = set(stored.get("families", []))
        family = len(families & stored_families) / max(1, len(families | stored_families))
        size_overlap = not (
            stored.get("nodes_max", 0) < context.get("nodes_min", 0)
            or context.get("nodes_max", 0) < stored.get("nodes_min", 0)
        )
        density_overlap = not (
            stored.get("density_max", 0.0) < context.get("density_min", 0.0)
            or context.get("density_max", 0.0) < stored.get("density_min", 0.0)
        )
        weight = stored.get("weight_regime") == context.get("weight_regime")
        scores.append(
            0.55 * family + 0.20 * float(size_overlap)
            + 0.15 * float(density_overlap) + 0.10 * float(weight))
    return max(scores, default=0.0)


class GraphMemory:
    """Evidence grounded contextual memory for Max Cut experiments."""

    schema_version = 1

    def __init__(self):
        self.items = {}
        self.episodes = []
        self.cards = []
        self.replay_results = []
        self.pe_scale = 0.05
        self.controller = {}

    @classmethod
    def load(cls, path):
        if not path or not os.path.exists(path):
            return cls()
        with open(path) as handle:
            state = json.load(handle)
        if _contains_sealed(state):
            raise ValueError("sealed test evidence found in graph memory")
        memory = cls()
        memory.items = state.get("items", {})
        memory.episodes = state.get("episodes", [])
        memory.cards = state.get("cards", [])
        memory.replay_results = state.get("replay_results", [])
        memory.pe_scale = float(state.get("pe_scale", 0.05))
        memory.controller = state.get("controller", {})
        memory.validate()
        return memory

    def save(self, path):
        if not path:
            return
        self.compact()
        state = {
            "schema_version": self.schema_version,
            "items": self.items,
            "episodes": self.episodes,
            "cards": self.cards,
            "replay_results": self.replay_results,
            "pe_scale": self.pe_scale,
            "controller": self.controller,
        }
        if _contains_sealed(state):
            raise ValueError("refusing to write sealed test evidence into graph memory")
        with open(path, "w") as handle:
            json.dump(state, handle, indent=2)

    def compact(self):
        """Bound memory while preserving every episode cited by a card."""
        protected = {
            episode_id
            for card in self.cards
            for field in ("evidence_episode_ids", "contradiction_episode_ids")
            for episode_id in card.get(field, [])
        }
        recent = [
            episode["episode_id"]
            for episode in self.episodes[-MAX_MEMORY_EPISODES:]
        ]
        keep = set(protected)
        for episode_id in reversed(recent):
            if len(keep) >= MAX_MEMORY_EPISODES:
                break
            keep.add(episode_id)
        self.episodes = [
            episode for episode in self.episodes
            if episode["episode_id"] in keep
        ]
        existing = {episode["episode_id"] for episode in self.episodes}
        for item in self.items.values():
            item["episode_ids"] = [
                episode_id for episode_id in item.get("episode_ids", [])
                if episode_id in existing
            ]
        self.validate()

    def validate(self):
        episode_ids = {episode["episode_id"] for episode in self.episodes}
        if len(episode_ids) != len(self.episodes):
            raise ValueError("graph memory contains duplicate episode ids")
        for card in self.cards:
            evidence = card.get("evidence_episode_ids", [])
            contradictions = card.get("contradiction_episode_ids", [])
            if not evidence or not set(evidence).issubset(episode_ids):
                raise ValueError("memory card has missing evidence")
            if not set(contradictions).issubset(episode_ids):
                raise ValueError("memory card has missing contradiction evidence")
            if card.get("status") == "trusted":
                seeds = {
                    self.episode_by_id(episode_id)["run_seed"]
                    for episode_id in evidence
                }
                if len(seeds) < 2:
                    raise ValueError("trusted memory card lacks independent seeds")
        if _contains_sealed({
                "items": self.items,
                "episodes": self.episodes,
                "cards": self.cards,
                "replay_results": self.replay_results,
        }):
            raise ValueError("sealed test evidence found in graph memory")
        return True

    def episode_by_id(self, episode_id):
        for episode in self.episodes:
            if episode["episode_id"] == episode_id:
                return episode
        raise KeyError(episode_id)

    def predict(self, context, program=None, train=None, items=None):
        """Predict development quality before development verification."""
        source = self.items if items is None else items
        weighted = []
        train_quality = (
            None if train is None else float(train.get("mean_quality", 0.0)))
        for item in source.values():
            if item.get("status") != "trusted" or not valid_program(item.get("program")):
                continue
            context_score = _context_match_score(item.get("contexts", []), context)
            scoped = sorted(
                item.get("context_stats", {}).values(),
                key=lambda row: -_context_match_score([row.get("context", {})], context),
            )
            scoped_best = scoped[0] if scoped else {}
            item_train = float(scoped_best.get("mean_train", item.get("mean_train", 0.0)))
            item_dev = float(scoped_best.get("mean_dev", item.get("mean_dev", 0.0)))
            program_score = (
                program_similarity(program, item["program"])
                if program is not None else 0.5
            )
            train_score = (
                math.exp(-abs(item_train - train_quality) / 0.05)
                if train_quality is not None else 0.5
            )
            corroboration = min(1.0, len(item.get("episode_ids", [])) / 3.0)
            weight = (
                context_score
                * (0.10 + program_score ** 2)
                * (0.25 + 0.75 * train_score)
                * (0.5 + corroboration)
            )
            if weight > 0.0:
                gap = item_train - item_dev
                weighted.append((weight, gap))
        weighted.sort(reverse=True)
        weighted = weighted[:12]
        if weighted:
            weights = np.asarray([row[0] for row in weighted], dtype=float)
            gaps = np.asarray([row[1] for row in weighted], dtype=float)
            mean_gap = float(np.average(gaps, weights=weights))
            neighbor_prediction = (
                float(np.clip(train_quality - mean_gap, 0.0, 1.0))
                if train_quality is not None
                else float(np.average([
                    float(item.get("mean_dev", 0.0))
                    for item in source.values()
                    if item.get("status") == "trusted"
                ]))
            )
            variance = float(np.average(
                (gaps - mean_gap) ** 2, weights=weights))
            coverage = min(1.0, len(weighted) / 8.0)
        else:
            neighbor_prediction = (
                float(np.clip(train_quality - 0.04, 0.0, 1.0))
                if train_quality is not None else 0.60
            )
            variance, coverage = 0.05, 0.0
        predicted = neighbor_prediction
        confidence = float(np.clip(
            coverage / (1.0 + 20.0 * variance + 5.0 * self.pe_scale), 0.0, 1.0))
        return predicted, confidence

    def retrieve(self, context):
        families = set(context.get("families", []))
        items = sorted(
            [
                item for item in self.items.values()
                if item.get("status") == "trusted"
            ],
            key=lambda item: (
                -_context_match_score(item.get("contexts", []), context),
                -float(item.get("strength", 0.0)),
                -float(item.get("mean_dev", 0.0)),
                item["key"],
            ))
        cards = [
            card for card in self.cards
            if card.get("status") == "trusted"
            and (card.get("scope") == "all" or card.get("scope") in families)
        ]
        cards.sort(key=lambda card: (-float(card.get("confidence", 0.0)), card["card_id"]))
        contradiction_ids = sorted({
            episode_id
            for card in cards
            for episode_id in card.get("contradiction_episode_ids", [])
        })
        return {
            "items": copy.deepcopy(items[:12]),
            "cards": copy.deepcopy(cards[:12]),
            "contradictions": [
                copy.deepcopy(self.episode_by_id(episode_id))
                for episode_id in contradiction_ids[:24]
            ],
        }

    def preferred_ops(self, context):
        ops = []
        for card in self.retrieve(context)["cards"]:
            action = card.get("do", "")
            if action.startswith("prefer operator "):
                op = action.removeprefix("prefer operator ").strip()
                if op in UNARY + BINARY:
                    ops.append(op)
        return ops[:8]

    def top_programs(self, context, count=12):
        return [
            copy.deepcopy(item["program"])
            for item in self.retrieve(context)["items"][:count]
            if valid_program(item.get("program"))
        ]

    def record_episode(self, row, domain, run_seed, gen, mode):
        """Append immutable evidence and aggregate the candidate by context."""
        base = f"{run_seed}|{gen}|{row['key']}|{len(self.episodes)}"
        episode_id = "ep-" + hashlib.sha256(base.encode()).hexdigest()[:16]
        episode = {
            "episode_id": episode_id,
            "run_seed": int(run_seed),
            "generation": int(gen),
            "mode": mode,
            "program_key": row["key"],
            "program": copy.deepcopy(row["program"]),
            "lineage": copy.deepcopy(row.get("lineage", {})),
            "context": domain.evidence_context(),
            "prediction": float(row["predicted"]),
            "confidence": float(row.get("confidence", 0.0)),
            "train": {
                key: copy.deepcopy(row["train"][key])
                for key in ("mean_quality", "family_std", "fitness", "family_scores")
            },
            "dev": {
                key: copy.deepcopy(row["dev"][key])
                for key in ("mean_quality", "family_std", "fitness", "family_scores")
            },
            "prediction_error": float(row["prediction_error"]),
            "confidence_error": float(
                abs(row["prediction_error"]) - (1.0 - row.get("confidence", 0.0))),
            "verifier_version": domain.verifier_version,
            "status": row["status"],
            "lesson": row.get("lesson", ""),
        }
        if _contains_sealed(episode):
            raise ValueError("sealed test evidence cannot become an episode")
        self.episodes.append(episode)
        item = self.items.get(row["key"])
        if item is None:
            item = {
                "key": row["key"],
                "program": copy.deepcopy(row["program"]),
                "episode_ids": [],
                "contexts": [],
                "context_stats": {},
                "mean_train": 0.0,
                "mean_dev": 0.0,
                "strength": 0.0,
                "status": row["status"],
                "prediction_error": float(row["prediction_error"]),
            }
            self.items[row["key"]] = item
        item["episode_ids"].append(episode_id)
        context = domain.context()
        if context not in item["contexts"]:
            item["contexts"].append(copy.deepcopy(context))
        context_key = json.dumps(context, sort_keys=True, separators=(",", ":"))
        context_stats = item.setdefault("context_stats", {}).setdefault(
            context_key,
            {"context": copy.deepcopy(context), "count": 0, "mean_train": 0.0, "mean_dev": 0.0},
        )
        context_stats["count"] += 1
        context_stats["mean_train"] += (
            row["train"]["mean_quality"] - context_stats["mean_train"]
        ) / context_stats["count"]
        context_stats["mean_dev"] += (
            row["dev"]["mean_quality"] - context_stats["mean_dev"]
        ) / context_stats["count"]
        count = len(item["episode_ids"])
        item["mean_train"] += (row["train"]["mean_quality"] - item["mean_train"]) / count
        item["mean_dev"] += (row["dev"]["mean_quality"] - item["mean_dev"]) / count
        item["status"] = row["status"]
        item["prediction_error"] = float(row["prediction_error"])
        item["strength"] = float(row.get("strength", item.get("strength", 0.0)))
        self.pe_scale = 0.95 * self.pe_scale + 0.05 * max(
            abs(float(row["prediction_error"])), 1e-5)
        return episode_id

    def sync_strengths(self, records):
        for key, row in records.items():
            if key in self.items:
                self.items[key]["strength"] = float(row.get("strength", 0.0))
                self.items[key]["status"] = row.get("status", "candidate")
        if len(self.items) > MAX_MEMORY_ITEMS:
            ranked = sorted(
                self.items.values(),
                key=lambda item: (-float(item.get("strength", 0.0)), item["key"]))
            self.items = {item["key"]: item for item in ranked[:MAX_MEMORY_ITEMS]}

    def distill_cards(self):
        """Distill scoped decision cards with explicit support and contradiction."""
        episodes = [
            episode for episode in self.episodes
            if episode.get("status") == "trusted"
        ]
        if not episodes:
            self.cards = []
            return []
        cards = []
        families = sorted({
            family
            for episode in episodes
            for family in episode["dev"]["family_scores"]
        })
        operators = sorted(UNARY + BINARY)
        for family in families:
            all_scores = [
                episode["dev"]["family_scores"][family]
                for episode in episodes
                if family in episode["dev"]["family_scores"]
            ]
            if not all_scores:
                continue
            median = float(np.median(all_scores))
            for op in operators:
                with_op = [
                    episode for episode in episodes
                    if op in program_operators(episode["program"])
                    and family in episode["dev"]["family_scores"]
                ]
                support = [
                    episode for episode in with_op
                    if episode["dev"]["family_scores"][family] >= median + 0.005
                ]
                contradictions = [
                    episode for episode in with_op
                    if episode["dev"]["family_scores"][family] <= median - 0.02
                ]
                if not support:
                    continue
                seeds = {episode["run_seed"] for episode in support}
                confidence = len(support) / max(1, len(support) + len(contradictions))
                confidence *= min(1.0, len(seeds) / 2.0)
                if len(contradictions) > len(support) * 1.5:
                    status = "contradicted"
                elif len(seeds) >= 2 and confidence >= 0.55:
                    status = "trusted"
                else:
                    status = "candidate"
                evidence = []
                seen_seeds = set()
                for episode in support:
                    if episode["run_seed"] not in seen_seeds:
                        evidence.append(episode["episode_id"])
                        seen_seeds.add(episode["run_seed"])
                for episode in reversed(support):
                    if episode["episode_id"] not in evidence:
                        evidence.append(episode["episode_id"])
                    if len(evidence) >= 12:
                        break
                cards.append({
                    "card_id": f"{family}:{op}",
                    "when": f"solving weighted Max Cut on {family} graphs",
                    "do": f"prefer operator {op}",
                    "preserve": "verify on unseen development graph seeds",
                    "avoid": "treating train quality as sufficient evidence",
                    "verify": f"compare {family} development quality against the current median",
                    "scope": family,
                    "evidence_episode_ids": evidence[:12],
                    "contradiction_episode_ids": [
                        episode["episode_id"] for episode in contradictions[-12:]],
                    "confidence": float(confidence),
                    "status": status,
                })
        self.cards = sorted(cards, key=lambda card: card["card_id"])
        self.validate()
        return self.cards

    def contradiction_rate(self):
        relevant = [card for card in self.cards if card.get("status") != "candidate"]
        if not relevant:
            return 0.0
        return sum(card.get("status") == "contradicted" for card in relevant) / len(relevant)

    def add_replay_result(self, result):
        clean = copy.deepcopy(result)
        if _contains_sealed(clean):
            raise ValueError("sealed test evidence cannot enter replay memory")
        self.replay_results.append(clean)
        self.replay_results = self.replay_results[-50:]


class UtilityController:
    """Online ridge models that value each cognitive mode."""

    def __init__(self, state=None, ridge=0.5):
        dimension = len(CONTROLLER_FEATURES)
        state = state or {}
        self.ridge = float(state.get("ridge", ridge))
        self.a = {}
        self.b = {}
        self.counts = {}
        for mode in COGNITIVE_MODES:
            self.a[mode] = np.asarray(
                state.get("a", {}).get(mode, np.eye(dimension) * self.ridge),
                dtype=float)
            self.b[mode] = np.asarray(
                state.get("b", {}).get(mode, np.zeros(dimension)),
                dtype=float)
            self.counts[mode] = int(state.get("counts", {}).get(mode, 0))
        self.last_mode = state.get("last_mode")
        last_features = state.get("last_features")
        self.last_features = (
            None if last_features is None else np.asarray(last_features, dtype=float))
        self.low_value_windows = int(state.get("low_value_windows", 0))
        self.contamination_windows = int(state.get("contamination_windows", 0))
        self.last_sleep_gen = int(state.get("last_sleep_gen", 0))

    def to_state(self):
        return {
            "ridge": self.ridge,
            "a": {mode: self.a[mode].tolist() for mode in COGNITIVE_MODES},
            "b": {mode: self.b[mode].tolist() for mode in COGNITIVE_MODES},
            "counts": dict(self.counts),
            "last_mode": self.last_mode,
            "last_features": (
                None if self.last_features is None else self.last_features.tolist()),
            "low_value_windows": self.low_value_windows,
            "contamination_windows": self.contamination_windows,
            "last_sleep_gen": self.last_sleep_gen,
        }

    def predictions(self, features):
        x = np.asarray(features, dtype=float)
        return {
            mode: float(x @ np.linalg.solve(self.a[mode], self.b[mode]))
            for mode in COGNITIVE_MODES
        }

    def observe(self, utility):
        if self.last_mode is None or self.last_features is None:
            return
        x = self.last_features
        mode = self.last_mode
        self.a[mode] += np.outer(x, x)
        self.b[mode] += x * float(utility)
        self.counts[mode] += 1

    def choose(self, features, gen, policy="learned"):
        x = np.asarray(features, dtype=float)
        predictions = self.predictions(x)
        flat = max(x[1], x[2], x[3]) < 0.002
        quarantine = x[7]
        if quarantine > 0.90:
            self.contamination_windows += 1
        else:
            self.contamination_windows = 0
        safety_stop = self.contamination_windows >= 3

        if gen - self.last_sleep_gen >= 25:
            mode = "sleep"
        elif policy == "fixed":
            mode = "dream" if flat else "focus"
        elif policy == "rules":
            if quarantine > 0.65:
                mode = "recover"
            elif x[6] > 0.60 or flat:
                mode = "dream"
            elif x[4] < 0.02:
                mode = "challenge"
            else:
                mode = "focus"
        else:
            under_sampled = [
                mode for mode in COGNITIVE_MODES
                if self.counts[mode] < 2
            ]
            mode = under_sampled[0] if under_sampled else max(
                COGNITIVE_MODES, key=lambda item: (predictions[item], item))

        all_nonpositive = all(value <= 0.0 for value in predictions.values())
        if policy == "fixed":
            self.low_value_windows = 0
        elif flat and (all_nonpositive if policy == "learned" else True):
            self.low_value_windows += 1
        else:
            self.low_value_windows = 0
        stop_requested = gen >= 30 and self.low_value_windows >= 3

        self.last_mode = mode
        self.last_features = x.copy()
        if mode == "sleep":
            self.last_sleep_gen = gen
        return {
            "mode": mode,
            "predictions": predictions,
            "stop_requested": stop_requested,
            "safety_stop": safety_stop,
        }


def controller_features(history, memory, elapsed_fraction, previous_hall_size):
    """Summarize the most recent decision window for the utility controller."""
    window = history[-5:]
    if not window:
        return np.asarray([1.0] + [0.0] * (len(CONTROLLER_FEATURES) - 1))
    first, last = window[0], window[-1]
    train_improvement = max(0.0, last["best_train"] - first["best_train"])
    dev_improvement = max(0.0, last["best_dev"] - first["best_dev"])
    calibration_improvement = max(0.0, first["mean_abs_pe"] - last["mean_abs_pe"])
    trusted_now = sum(item.get("status") == "trusted" for item in memory.items.values())
    novel_rate = max(0, trusted_now - previous_hall_size) / max(1, trusted_now)
    duplicate_rate = float(np.mean([row["duplicates"] for row in window]))
    quarantine_rate = float(np.mean([row["quarantine"] for row in window]))
    return np.asarray([
        1.0,
        train_improvement,
        dev_improvement,
        calibration_improvement,
        novel_rate,
        1.0 - duplicate_rate,
        duplicate_rate,
        quarantine_rate,
        memory.contradiction_rate(),
        float(np.clip(elapsed_fraction, 0.0, 1.0)),
    ], dtype=float)


def controller_utility(features):
    """Value useful progress and diversity while charging contamination and compute."""
    x = np.asarray(features, dtype=float)
    return float(
        0.30 * x[1]
        + 1.00 * x[2]
        + 0.20 * x[3]
        + 0.10 * x[4]
        + 0.05 * x[5]
        - 0.40 * x[7]
        - 0.20 * x[8]
        - 0.02 * x[9]
    )


def _expr_positions(node, prefix=()):
    positions = [prefix]
    if isinstance(node, list):
        for index, child in enumerate(node[1:], 1):
            positions.extend(_expr_positions(child, prefix + (index,)))
    return positions


def _expr_get(node, path):
    current = node
    for index in path:
        current = current[index]
    return current


def _expr_set(node, path, value):
    if not path:
        return copy.deepcopy(value)
    out = copy.deepcopy(node)
    current = out
    for index in path[:-1]:
        current = current[index]
    current[path[-1]] = copy.deepcopy(value)
    return out


def expr_operators(node):
    out = []
    if isinstance(node, list):
        out.append(node[0])
        for child in node[1:]:
            out.extend(expr_operators(child))
    return out


def program_operators(program):
    return expr_operators(program["init"]) + expr_operators(program["move"])


def program_fragments(program):
    fragments = set()
    for root in (program["init"], program["move"]):
        for path in _expr_positions(root):
            fragments.add(expr_to_str(_expr_get(root, path)))
    fragments.add(f"steps={program['steps_per_node']}")
    fragments.add(f"restarts={program['restarts']}")
    fragments.add(f"tabu={program['tabu_window']}")
    return fragments


def program_similarity(left, right):
    a, b = program_fragments(left), program_fragments(right)
    return len(a & b) / max(1, len(a | b))


def program_novelty(program, records):
    if not records:
        return 1.0
    return 1.0 - max(program_similarity(program, row["program"]) for row in records)


def random_expr(depth=MAX_EXPR_DEPTH, preferred_ops=None, allowed_fields=None):
    fields = FIELDS if allowed_fields is None else allowed_fields
    if depth <= 1 or random.random() < 0.22:
        return random.choice(fields)
    preferred = [op for op in (preferred_ops or []) if op in UNARY + BINARY]
    if preferred and random.random() < 0.45:
        op = random.choice(preferred)
    else:
        op = random.choice(UNARY + BINARY)
    if op in UNARY:
        return [op, random_expr(depth - 1, preferred_ops, fields)]
    return [
        op,
        random_expr(depth - 1, preferred_ops, fields),
        random_expr(depth - 1, preferred_ops, fields),
    ]


def random_program(preferred_ops=None):
    for _ in range(100):
        program = {
            "init": random_expr(
                random.randint(2, MAX_EXPR_DEPTH),
                preferred_ops,
                STATIC_FIELDS + CONSTANTS),
            "move": random_expr(random.randint(2, MAX_EXPR_DEPTH), preferred_ops),
            "steps_per_node": random.choices(STEPS_PER_NODE, weights=(5, 3, 2, 1))[0],
            "restarts": random.choices(RESTARTS, weights=(5, 3, 1))[0],
            "tabu_window": random.choice(TABU_WINDOWS),
        }
        if valid_program(program):
            return program
    return baseline_program()


def mutate_program(program, preferred_ops=None):
    for _ in range(100):
        out = copy.deepcopy(program)
        choice = random.choice(("init", "move", "steps", "restarts", "tabu"))
        if choice in ("init", "move"):
            root = out[choice]
            path = random.choice(_expr_positions(root))
            remaining = max(1, MAX_EXPR_DEPTH - len(path))
            allowed_fields = STATIC_FIELDS + CONSTANTS if choice == "init" else FIELDS
            out[choice] = _expr_set(
                root, path, random_expr(remaining, preferred_ops, allowed_fields))
        elif choice == "steps":
            out["steps_per_node"] = random.choices(STEPS_PER_NODE, weights=(5, 3, 2, 1))[0]
        elif choice == "restarts":
            out["restarts"] = random.choices(RESTARTS, weights=(5, 3, 1))[0]
        else:
            out["tabu_window"] = random.choice(TABU_WINDOWS)
        if valid_program(out):
            return out
    return copy.deepcopy(program)


def crossover_program(left, right):
    for _ in range(100):
        out = copy.deepcopy(left)
        field = random.choice(("init", "move"))
        donor_field = random.choice(("init", "move"))
        target = random.choice(_expr_positions(out[field]))
        donor = _expr_get(right[donor_field], random.choice(_expr_positions(right[donor_field])))
        out[field] = _expr_set(out[field], target, donor)
        if random.random() < 0.35:
            scalar = random.choice(("steps_per_node", "restarts", "tabu_window"))
            out[scalar] = right[scalar]
        if valid_program(out):
            return out
    return copy.deepcopy(left)


def evaluate_program(program, graphs):
    """Evaluate one reusable program across a graph suite."""
    if not valid_program(program):
        raise ValueError("invalid Max Cut program")
    outcomes = []
    for graph in graphs:
        result = execute_program(program, graph)
        outcomes.append({
            "graph_id": graph["id"],
            "family": graph["family"],
            "quality": result["quality"],
            "cut": result["cut"],
        })
    qualities = np.asarray([row["quality"] for row in outcomes], dtype=float)
    families = {}
    for family in sorted({row["family"] for row in outcomes}):
        families[family] = float(np.mean([
            row["quality"] for row in outcomes if row["family"] == family]))
    family_values = np.asarray(list(families.values()), dtype=float)
    complexity = program_size(program)
    step_cost = program["steps_per_node"] * program["restarts"] / 32.0
    mean_quality = float(np.mean(qualities)) if qualities.size else 0.0
    family_std = float(np.std(family_values)) if family_values.size else 0.0
    search_fitness = mean_quality - 0.25 * family_std - 0.002 * complexity - 0.01 * step_cost
    return {
        "mean_quality": mean_quality,
        "family_std": family_std,
        "worst_family": float(np.min(family_values)) if family_values.size else 0.0,
        "complexity": complexity,
        "step_cost": float(step_cost),
        "fitness": float(search_fitness),
        "family_scores": families,
        "outcomes": outcomes,
    }


def productive_surprise(error, scale):
    ratio = abs(float(error)) / max(float(scale), 1e-9)
    return float(ratio * math.exp(1.0 - ratio))


def _graph_candidate_status(train, dev, error, pe_scale, quarantine):
    gap = train["mean_quality"] - dev["mean_quality"]
    reasons = []
    if gap > 0.12:
        reasons.append("large train to development gap")
    if dev["family_std"] > 0.18:
        reasons.append("unstable across graph families")
    if dev["mean_quality"] < 0.50:
        reasons.append("weak development quality")
    extreme = abs(error) / max(pe_scale, 1e-9) > 3.0
    if extreme and (gap > 0.06 or dev["family_std"] > 0.12):
        reasons.append("extreme surprise failed development stress verification")
    if quarantine and reasons:
        return "quarantined", "; ".join(reasons)
    return "trusted", ""


def _select_mode(gen, stall, duplicate_rate, quarantine_rate):
    if quarantine_rate > 0.65:
        return "recover"
    if stall >= 3 or duplicate_rate > 0.60 or gen % 5 == 0:
        return "dream"
    return "focus"


def _proposal_population(elites, count, mode, preferred_ops=None):
    out = []
    explore = {
        "dream": 0.78,
        "focus": 0.28,
        "challenge": 0.58,
        "recover": 0.18,
        "sleep": 0.22,
    }.get(mode, 0.35)
    for _ in range(count * 20):
        if len(out) >= count:
            break
        if not elites or random.random() < explore:
            candidate = random_program(preferred_ops=preferred_ops)
        elif random.random() < 0.42 and len(elites) >= 2:
            left, right = random.sample(elites, 2)
            candidate = crossover_program(left["program"], right["program"])
        else:
            candidate = mutate_program(random.choice(elites)["program"], preferred_ops)
        if valid_program(candidate):
            out.append(candidate)
    while len(out) < count:
        out.append(random_program(preferred_ops=preferred_ops))
    return out


def _write_graph_history(path, history):
    if not path:
        return
    fields = [
        "gen", "best_train", "best_dev", "median_train", "mean_abs_pe",
        "mode", "duplicates", "quarantine", "trusted", "elapsed_seconds",
        "controller_utility", "stop_requested",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in history])


def run(
        generations=250, pop=56, seed=7, max_minutes=90, variant="ours",
        stop_policy="fixed", memory_path="graph_memory.json",
        history_path="graph_history.csv", replay=True, quiet=False,
        suite_kind="standard", memory_obj=None, persist=True):
    """Run the offline Max Cut search with contextual evidence memory."""
    if stop_policy not in ("fixed", "rules", "learned"):
        raise ValueError(f"unknown stop policy {stop_policy}")
    random.seed(seed)
    domain = MaxCutDomain(seed=seed, suite_kind=suite_kind)
    cfg = dict(GRAPH_VARIANTS[variant])
    memory = memory_obj if memory_obj is not None else GraphMemory.load(memory_path)
    controller = UtilityController(memory.controller)
    retrieved_at_start = memory.retrieve(domain.context())
    preferred_ops = memory.preferred_ops(domain.context())
    recalled = memory.top_programs(domain.context(), count=max(1, pop // 6))
    population = recalled + _proposal_population(
        [], pop - len(recalled), "dream", preferred_ops=preferred_ops)
    initial_keys = [domain.key(program) for program in population]
    memory_injection = {
        "retrieved_cards": len(retrieved_at_start["cards"]),
        "retrieved_contradictions": len(retrieved_at_start["contradictions"]),
        "recalled_programs": len(recalled),
        "preferred_ops": list(preferred_ops),
    }
    records = {}
    train_cache = {}
    dev_cache = {}
    hall = []
    history = []
    pe_scale = memory.pe_scale
    best_dev = float("-inf")
    best_record = None
    stall = 0
    mode = "dream"
    previous_hall_size = sum(
        item.get("status") == "trusted" for item in memory.items.values())
    start = time.time()
    stop_reason = "generation_budget"
    verifier_failure_windows = 0
    effective_minutes = min(
        HARD_MAX_MINUTES,
        max_minutes if max_minutes and max_minutes > 0 else HARD_MAX_MINUTES,
    )
    deadline = start + effective_minutes * 60.0
    if not quiet:
        print(f"maxcut run  gens {generations}  pop {pop}  seed {seed}  "
              f"variant {variant}  stop {stop_policy}")
    for gen in range(min(generations, HARD_MAX_GENERATIONS)):
        if deadline is not None and time.time() >= deadline:
            stop_reason = "time_budget"
            break
        evaluated = {}
        duplicates = 0
        verifier_failures = 0
        predictor_items = copy.deepcopy(memory.items)
        for program in population:
            key = domain.key(program)
            if key in evaluated:
                duplicates += 1
                continue
            try:
                if key in train_cache:
                    train = copy.deepcopy(train_cache[key])
                else:
                    train = domain.evaluate_train(program)
                    train_cache[key] = copy.deepcopy(train)
                predicted, confidence = memory.predict(
                    domain.context(), program=program, train=train, items=predictor_items)
                if key in dev_cache:
                    dev = copy.deepcopy(dev_cache[key])
                else:
                    dev = domain.evaluate_dev(program)
                    dev_cache[key] = copy.deepcopy(dev)
            except Exception:
                verifier_failures += 1
                continue
            error = dev["mean_quality"] - predicted
            novelty = domain.novelty(program, hall)
            status, lesson = domain.stress_verify(
                train, dev, error, pe_scale, cfg["quarantine"])
            if cfg["surprise"] == "none":
                surprise = 0.0
            elif cfg["surprise"] == "monotonic":
                surprise = min(abs(error) / max(pe_scale, 1e-9), 3.0) / 3.0
            else:
                surprise = productive_surprise(error, pe_scale)
            row = {
                "key": key,
                "program": copy.deepcopy(program),
                "train": train,
                "dev": dev,
                "predicted": predicted,
                "confidence": confidence,
                "prediction_error": error,
                "novelty": novelty,
                "surprise": surprise,
                "status": status,
                "lesson": lesson,
                "strength": records.get(key, {}).get("strength", 0.0),
                "lineage": {"mode": mode},
            }
            evaluated[key] = row
            records[key] = row
            pe_scale = 0.95 * pe_scale + 0.05 * max(abs(error), 1e-5)
            memory.record_episode(row, domain, seed, gen, mode)
        if verifier_failures > max(3, pop // 3):
            verifier_failure_windows += 1
        else:
            verifier_failure_windows = 0
        if verifier_failure_windows >= 3:
            stop_reason = "verifier_failure"
            break
        if not evaluated:
            stop_reason = "verifier_failure"
            break
        trusted = [row for row in evaluated.values() if row["status"] == "trusted"]
        reportable = trusted or list(evaluated.values())
        ranked = sorted(
            trusted,
            key=lambda row: -(
                row["train"]["fitness"]
                + (0.10 * row["novelty"] if cfg["novelty"] else 0.0)
                + 0.08 * row["surprise"]))
        elites = ranked[:max(2, min(12, pop // 4))]
        fit_values = [row["train"]["fitness"] for row in trusted]
        lo = min(fit_values) if fit_values else 0.0
        hi = max(fit_values) if fit_values else 1.0
        for row in records.values():
            row["strength"] *= 0.96
        for row in trusted:
            norm_fit = (row["train"]["fitness"] - lo) / max(hi - lo, 1e-9)
            row["strength"] += 0.60 * norm_fit + 0.20 * row["surprise"] + (
                0.10 * row["novelty"] if cfg["novelty"] else 0.0)
        memory.sync_strengths(records)
        hall = sorted(
            [row for row in records.values() if row["status"] == "trusted"],
            key=lambda row: (-row["dev"]["mean_quality"], row["key"]))[:200]
        gen_best = max(reportable, key=lambda row: row["dev"]["mean_quality"])
        if gen_best["dev"]["mean_quality"] > best_dev + 1e-9:
            best_dev = gen_best["dev"]["mean_quality"]
            best_record = copy.deepcopy(gen_best)
            stall = 0
        else:
            stall += 1
        duplicate_rate = duplicates / max(1, len(population))
        quarantine_rate = 1.0 - len(trusted) / max(1, len(evaluated))
        mean_abs_pe = float(np.mean([
            abs(row["prediction_error"]) for row in evaluated.values()]))
        history.append({
            "gen": gen,
            "best_train": best_record["train"]["mean_quality"],
            "best_dev": best_dev,
            "median_train": float(np.median([
                row["train"]["mean_quality"] for row in reportable])),
            "mean_abs_pe": mean_abs_pe,
            "mode": mode,
            "duplicates": duplicate_rate,
            "quarantine": quarantine_rate,
            "trusted": len(trusted),
            "elapsed_seconds": time.time() - start,
            "controller_utility": None,
            "stop_requested": False,
        })
        if not quiet:
            print(f"gen {gen:03d}  train {best_record['train']['mean_quality']:.4f}  "
                  f"dev {best_dev:.4f}  pe {mean_abs_pe:.4f}  mode {mode:7s}  "
                  f"quarantine {quarantine_rate:.2f}  duplicates {duplicate_rate:.2f}")
        should_stop = False
        if (gen + 1) % 5 == 0:
            elapsed_fraction = (
                (time.time() - start) / max(1.0, effective_minutes * 60.0)
            )
            features = controller_features(
                history, memory, elapsed_fraction, previous_hall_size)
            utility = controller_utility(features)
            controller.observe(utility)
            decision = controller.choose(features, gen + 1, stop_policy)
            mode = decision["mode"]
            history[-1]["controller_utility"] = utility
            history[-1]["stop_requested"] = decision["stop_requested"]
            previous_hall_size = sum(
                item.get("status") == "trusted" for item in memory.items.values())
            memory.controller = controller.to_state()
            if mode == "sleep":
                memory.distill_cards()
            if decision["safety_stop"]:
                stop_reason = "contamination"
                should_stop = True
            elif decision["stop_requested"]:
                stop_reason = "convergence"
                should_stop = True
        elif stop_policy == "fixed":
            mode = _select_mode(gen + 1, stall, duplicate_rate, quarantine_rate)
        if should_stop:
            break
        carry = [copy.deepcopy(row["program"]) for row in elites[:max(1, pop // 6)]]
        preferred_ops = memory.preferred_ops(domain.context())
        population = carry + _proposal_population(
            elites, pop - len(carry), mode, preferred_ops=preferred_ops)
        if persist and memory_path and (gen + 1) % 25 == 0:
            memory.distill_cards()
            memory.save(memory_path)
    if best_record is None:
        raise RuntimeError("Max Cut search produced no evaluable programs")
    sealed = domain.evaluate_sealed(best_record["program"])
    memory.distill_cards()
    memory.controller = controller.to_state()
    replay_result = None
    if replay and suite_kind == "standard":
        replay_result = paired_memory_replay(memory, seed + 40000)
        memory.add_replay_result(replay_result)
    if persist:
        memory.save(memory_path)
    _write_graph_history(history_path, history)
    if not quiet:
        print(f"best Max Cut program  train {best_record['train']['mean_quality']:.4f}  "
              f"dev {best_record['dev']['mean_quality']:.4f}  "
              f"sealed {sealed['mean_quality']:.4f}  stop {stop_reason}")
        if replay_result:
            print(f"memory replay  cold {replay_result['cold']:.4f}  "
                  f"warm {replay_result['warm']:.4f}  lift {replay_result['lift']:+.4f}")
            print(f"  diagnosis {replay_result['diagnosis']}  "
                  f"cards {replay_result['warm_injection']['retrieved_cards']}  "
                  f"recalled {replay_result['warm_injection']['recalled_programs']}  "
                  f"initial overlap {replay_result['initial_candidate_overlap']:.2f}")
        print(f"  {best_record['key']}")
    return {
        "best": best_record,
        "sealed": sealed,
        "history": history,
        "records": records,
        "stop_reason": stop_reason,
        "memory": memory,
        "memory_lift": None if replay_result is None else replay_result["lift"],
        "initial_keys": initial_keys,
        "memory_injection": memory_injection,
    }


def paired_memory_replay(memory, seed):
    """Measure causal memory lift with matched cold and warm replay searches."""
    cold = run(
        generations=12,
        pop=20,
        seed=seed,
        max_minutes=0,
        variant="ours",
        stop_policy="fixed",
        memory_path=None,
        history_path=None,
        replay=False,
        quiet=True,
        suite_kind="replay",
        memory_obj=GraphMemory(),
        persist=False,
    )
    warm_memory = copy.deepcopy(memory)
    warm = run(
        generations=12,
        pop=20,
        seed=seed,
        max_minutes=0,
        variant="ours",
        stop_policy="fixed",
        memory_path=None,
        history_path=None,
        replay=False,
        quiet=True,
        suite_kind="replay",
        memory_obj=warm_memory,
        persist=False,
    )
    cold_score = float(cold["best"]["dev"]["mean_quality"])
    warm_score = float(warm["best"]["dev"]["mean_quality"])
    cold_keys = set(cold["initial_keys"])
    warm_keys = set(warm["initial_keys"])
    trajectory_equal = [
        row["best_dev"] for row in cold["history"]
    ] == [
        row["best_dev"] for row in warm["history"]
    ]
    intervention_present = bool(
        warm["memory_injection"]["retrieved_cards"]
        or warm["memory_injection"]["recalled_programs"]
        or warm["memory_injection"]["preferred_ops"]
    )
    return {
        "seed": int(seed),
        "generations": 12,
        "cold": cold_score,
        "warm": warm_score,
        "lift": warm_score - cold_score,
        "intervention_present": intervention_present,
        "warm_injection": warm["memory_injection"],
        "initial_candidate_overlap": len(cold_keys & warm_keys) / max(1, len(cold_keys | warm_keys)),
        "initial_candidates_identical": cold_keys == warm_keys,
        "trajectory_identical": trajectory_equal,
        "diagnosis": (
            "no_memory_injected" if not intervention_present else
            "memory_did_not_change_candidates" if cold_keys == warm_keys else
            "different_candidates_converged" if trajectory_equal else
            "memory_changed_search"
        ),
    }


def graph_memory_selftest():
    """Exercise grounding, scope, contradictions, and the sealed firewall."""
    memory = GraphMemory()
    domain = MaxCutDomain(seed=19)
    good = baseline_program()
    bad = {
        "init": ["rank", "c_0"],
        "move": ["rank", "c_0"],
        "steps_per_node": 1,
        "restarts": 1,
        "tabu_window": 0,
    }
    for run_seed in (1, 2):
        for label, program, score in (
                ("good-a", good, 0.95),
                ("good-b", good, 0.94),
                ("good-c", good, 0.93),
                ("good-d", good, 0.92),
                ("good-e", good, 0.91),
                ("good-f", good, 0.90),
                ("bad", bad, 0.54)):
            family_scores = {
                family: score
                for family in domain.context()["families"]
            }
            stats = {
                "mean_quality": score,
                "family_std": 0.01,
                "fitness": score - 0.03,
                "family_scores": family_scores,
            }
            row = {
                "key": f"{label}-{run_seed}",
                "program": program,
                "lineage": {"parents": []},
                "predicted": 0.65,
                "confidence": 0.50,
                "prediction_error": score - 0.65,
                "train": stats,
                "dev": stats,
                "status": "trusted",
                "lesson": "",
                "strength": score,
            }
            memory.record_episode(row, domain, run_seed, 0, "focus")
    memory.distill_cards()
    good_prediction, _ = memory.predict(
        domain.context(), good, train={"mean_quality": 0.95})
    bad_prediction, _ = memory.predict(
        domain.context(), bad, train={"mean_quality": 0.54})
    assert good_prediction > bad_prediction
    trusted = [card for card in memory.cards if card["status"] == "trusted"]
    assert trusted
    assert all(card["evidence_episode_ids"] for card in trusted)
    assert any(card["contradiction_episode_ids"] for card in memory.cards)
    retrieved = memory.retrieve(domain.context())
    assert retrieved["cards"]
    assert retrieved["contradictions"]
    assert memory.validate()
    try:
        memory.add_replay_result({"suite": "sealed-forbidden"})
        raise AssertionError("sealed result entered graph memory")
    except ValueError:
        pass
    print(f"  contextual memory cards {len(memory.cards)} trusted {len(trusted)} "
          f"contradictions {len(retrieved['contradictions'])}")
    print(f"  candidate aware prediction good {good_prediction:.3f} bad {bad_prediction:.3f}")
    print("  evidence grounding, context retrieval, and sealed firewall passed")


def controller_selftest():
    """Exercise learned stopping, progress continuation, and safety overrides."""
    flat = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.70, 0.30, 0.10, 0.0, 0.10])
    improving = np.asarray([1.0, 0.01, 0.02, 0.01, 0.05, 0.90, 0.10, 0.05, 0.0, 0.10])
    contaminated = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.70, 0.95, 0.0, 0.10])

    stagnant = UtilityController()
    decisions = []
    for gen in range(5, 35, 5):
        stagnant.observe(controller_utility(flat))
        decisions.append(stagnant.choose(flat, gen, "learned"))
    assert not any(row["stop_requested"] for row in decisions if row is not decisions[-1])
    assert decisions[-1]["stop_requested"]

    progressing = UtilityController()
    for gen in range(5, 60, 5):
        progressing.observe(controller_utility(improving))
        decision = progressing.choose(improving, gen, "learned")
        assert not decision["stop_requested"]

    unsafe = UtilityController()
    safety = []
    for gen in (5, 10, 15):
        unsafe.observe(controller_utility(contaminated))
        safety.append(unsafe.choose(contaminated, gen, "rules"))
    assert safety[-1]["safety_stop"]

    forced = UtilityController()
    assert forced.choose(improving, 25, "learned")["mode"] == "sleep"
    assert min(9999, HARD_MAX_GENERATIONS) == 250
    assert min(9999, HARD_MAX_MINUTES) == 90
    print("  controller continues on progress and stops after sustained flat utility")
    print("  minimum generation, forced sleep, and contamination stop passed")


def maxcut_selftest():
    """Exercise the grammar, graph firewall, executor, and exact cut verifier."""
    program = baseline_program()
    assert valid_program(program)
    assert not valid_program({**program, "steps_per_node": 999999})
    assert not valid_program({**program, "move": ["unknown", "degree"]})
    assert not valid_program({**program, "init": "flip_gain"})

    empty = graph_from_edges(0, [], "empty")
    edge = graph_from_edges(2, [(0, 1)], "edge")
    triangle = graph_from_edges(3, [(0, 1), (1, 2), (2, 0)], "triangle")
    complete4 = graph_from_edges(
        4, [(i, j) for i in range(4) for j in range(i + 1, 4)], "complete4")
    bipartite = graph_from_edges(
        8, [(i, j) for i in range(4) for j in range(4, 8)], "bipartite")
    cases = [
        ("empty", empty, 0.0),
        ("single edge", edge, 1.0),
        ("triangle", triangle, 2.0),
        ("complete four", complete4, 4.0),
        ("bipartite", bipartite, 16.0),
    ]
    print("Max Cut verifier self test:")
    for name, graph, expected in cases:
        result = execute_program(program, graph)
        assert abs(result["cut"] - expected) < 1e-9, (name, result, expected)
        print(f"  {name:14s} cut {result['cut']:.1f} expected {expected:.1f}")

    try:
        graph_from_edges(2, [(0, 0)])
        raise AssertionError("self edge passed validation")
    except ValueError:
        pass
    try:
        execute_program({**program, "restarts": 99}, edge)
        raise AssertionError("excessive program passed validation")
    except ValueError:
        pass

    suites = {kind: build_graph_suite(kind, seed=3)
              for kind in ("train", "dev", "sealed", "replay")}
    assert [len(suites[k]) for k in ("train", "dev", "sealed", "replay")] == [20, 10, 14, 8]
    qualities = [execute_program(program, graph)["quality"] for graph in suites["train"]]
    print(f"  train suite      graphs 20 mean quality {np.mean(qualities):.4f}")
    print("  graph firewall, bounded executor, and suite sizes passed")
    graph_memory_selftest()
    controller_selftest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Max Cut creativity laboratory")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--pop", type=int, default=56)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-minutes", type=float, default=90)
    parser.add_argument("--variant", choices=sorted(GRAPH_VARIANTS), default="ours")
    parser.add_argument("--stop-policy", choices=("fixed", "rules", "learned"), default="rules")
    parser.add_argument("--memory-path", default="graph_memory.json")
    parser.add_argument("--history-path", default="graph_history.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        maxcut_selftest()
        return
    run(
        generations=args.generations,
        pop=args.pop,
        seed=args.seed,
        max_minutes=args.max_minutes,
        variant=args.variant,
        stop_policy=args.stop_policy,
        memory_path=args.memory_path,
        history_path=args.history_path,
    )


if __name__ == "__main__":
    main()
