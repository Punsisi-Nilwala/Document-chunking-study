"""Step 4 - evaluation utilities: relevance labels, metrics, splits, bootstrap."""
import json
import math
from pathlib import Path

import numpy as np

from .text_utils import normalize

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load_questions():
    return json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))


def relevant_chunks(chunks, evidence):
    """Indices of chunks that contain ANY gold-evidence phrase (after normalisation).

    The label depends only on the text, never on chunk size, so all chunk sizes
    are judged by exactly the same criterion.
    """
    ev = [normalize(e) for e in evidence]
    norm = [normalize(c.text) for c in chunks]
    return {i for i, t in enumerate(norm) if any(e in t for e in ev)}


def first_relevant_rank(order, rel):
    """1-based rank of the first relevant chunk in ``order`` (None if there is none)."""
    for pos, idx in enumerate(order, start=1):
        if idx in rel:
            return pos
    return None


def metrics_from_ranks(ranks, chunk_size, budget_words=1000):
    """Aggregate per-question ranks (None = no relevant chunk exists)."""
    r = np.array([np.inf if x is None else x for x in ranks], dtype=float)
    k_budget = max(1, math.ceil(budget_words / chunk_size))
    return {
        "hit@1": float(np.mean(r <= 1)),
        "hit@3": float(np.mean(r <= 3)),
        "hit@5": float(np.mean(r <= 5)),
        f"hit@{budget_words}w": float(np.mean(r <= k_budget)),
        "mrr@10": float(np.mean(np.where(r <= 10, 1.0 / r, 0.0))),
    }


def split_dev_test(questions, test_frac=0.3, seed=SEED):
    """Stratified (by category) hold-out split. Returns (dev_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    dev, test = [], []
    for cat in sorted({q["category"] for q in questions}):
        idx = [i for i, q in enumerate(questions) if q["category"] == cat]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_frac))
        test += idx[:n_test]
        dev += idx[n_test:]
    return sorted(dev), sorted(test)


def kfold_indices(idx, k=5, seed=SEED):
    """Plain shuffled k-fold over a list of indices -> list of (train, val)."""
    idx = np.array(idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [(np.concatenate([f for j, f in enumerate(folds) if j != i]), folds[i])
            for i in range(k)]


def bootstrap_ci(values, n_boot=2000, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI of the mean of per-question scores."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def per_question_scores(ranks, chunk_size, metric, budget_words=1000):
    """Per-question 0/1 (or reciprocal-rank) scores, used for bootstrap."""
    r = np.array([np.inf if x is None else x for x in ranks], dtype=float)
    if metric == "mrr@10":
        return np.where(r <= 10, 1.0 / r, 0.0)
    k = {"hit@1": 1, "hit@3": 3, "hit@5": 5}.get(metric)
    if k is None:
        k = max(1, math.ceil(budget_words / chunk_size))
    return (r <= k).astype(float)
