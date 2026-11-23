"""Step 5 - run every (method x chunk size) experiment and save tables.

    python -m src.run_experiments

Protocol
--------
* 103 hand-written questions, each with gold evidence phrase(s) from the handbook.
* Stratified 70/30 split -> DEV (72 q) and TEST (31 q). TEST is only used once,
  for the final tuned systems.
* Table A  : default hyper-parameters, no tuning -> evaluated on ALL questions
             (the cleanest view of the chunk-size effect).
* Table B  : BM25 k1/b tuned inside DEV with 5-fold CV (selection on the training
             folds, score on the held-out fold), then the CV-chosen setting is
             refit on DEV and scored on TEST.
"""
import itertools
import json

import numpy as np
import pandas as pd

from .chunking import chunk_words, pages_to_words
from .evaluate import (SEED, bootstrap_ci, first_relevant_rank, kfold_indices,
                       load_questions, metrics_from_ranks, per_question_scores,
                       relevant_chunks, split_dev_test)
from .extract import load_pages
from .retrieval import BM25Retriever, TfidfRetriever

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
RES = ROOT / "results"

SIZES = [100, 200, 300, 500, 1000]          # 200/500/1000 are the required ones
PRIMARY = [200, 500, 1000]
GRID = list(itertools.product([0.9, 1.2, 1.5, 2.0], [0.4, 0.75, 0.9]))   # (k1, b)
N_RANDOM = 50

# name -> (retriever kind, use_stem, overlap)
METHODS = {
    "random":         ("random", False, 0.0),
    "tfidf":          ("tfidf", False, 0.0),        # <- BASELINE
    "bm25":           ("bm25", False, 0.0),
    "bm25+stem":      ("bm25", True, 0.0),
    "bm25+stem+ov25": ("bm25", True, 0.25),
}
TUNED = ["bm25", "bm25+stem", "bm25+stem+ov25"]


def build_chunks(words, word_pages, size, overlap, questions):
    chunks = chunk_words(words, word_pages, size, overlap)
    rels = [relevant_chunks(chunks, q["evidence"]) for q in questions]
    return chunks, rels


def ranks_with(retriever, chunks, rels, questions):
    out = []
    for q, rel in zip(questions, rels):
        out.append(first_relevant_rank(retriever.rank(q["question"]), rel) if rel else None)
    return out


def random_ranks(n_chunks, rels, rng):
    """Expected behaviour of a retriever that returns chunks in random order."""
    ranks = []
    for rel in rels:
        if not rel:
            ranks.append(None)
            continue
        order = rng.permutation(n_chunks)
        ranks.append(first_relevant_rank(order, rel))
    return ranks


def main():
    RES.mkdir(exist_ok=True)
    pages = load_pages()
    words, word_pages = pages_to_words(pages)
    questions = load_questions()
    dev, test = split_dev_test(questions)
    print(f"{len(words)} words | {len(questions)} questions | dev={len(dev)} test={len(test)}")
    json.dump({"dev": dev, "test": test}, open(RES / "split.json", "w"))

    rows_default, rows_tuned, rank_rows = [], [], []
    store = {}                                   # (method,size) -> default ranks
    rng = np.random.default_rng(SEED)

    for size in SIZES:
        for name, (kind, stem_flag, overlap) in METHODS.items():
            chunks, rels = build_chunks(words, word_pages, size, overlap, questions)
            n_cov = sum(1 for r in rels if r)

            # ---------- Table A: default parameters, all questions ----------
            if kind == "random":
                reps = [random_ranks(len(chunks), rels, rng) for _ in range(N_RANDOM)]
                mets = [metrics_from_ranks(r, size) for r in reps]
                metrics = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
                ranks = reps[0]
                ci = {}
            else:
                ret = (TfidfRetriever(stem_flag) if kind == "tfidf"
                       else BM25Retriever(1.5, 0.75, stem_flag)).fit([c.text for c in chunks])
                ranks = ranks_with(ret, chunks, rels, questions)
                metrics = metrics_from_ranks(ranks, size)
                lo, hi = bootstrap_ci(per_question_scores(ranks, size, "mrr@10"))
                ci = {"mrr_ci_lo": lo, "mrr_ci_hi": hi}
            store[(name, size)] = ranks
            rows_default.append({"method": name, "chunk_size": size, "n_chunks": len(chunks),
                                 "evidence_covered": f"{n_cov}/{len(questions)}",
                                 **metrics, **ci})
            for q, r in zip(questions, ranks):
                rank_rows.append({"method": name, "chunk_size": size, "qid": q["id"],
                                  "category": q["category"], "rank": r})

            # ---------- Table B: tuned BM25, CV on dev, then test ----------
            if name in TUNED:
                grid_ranks = {}
                for k1, b in GRID:
                    ret = BM25Retriever(k1, b, stem_flag).fit([c.text for c in chunks])
                    grid_ranks[(k1, b)] = ranks_with(ret, chunks, rels, questions)

                def mean_mrr(params, idx):
                    return metrics_from_ranks([grid_ranks[params][i] for i in idx], size)["mrr@10"]

                fold_mets, picked = [], []
                for tr, va in kfold_indices(dev):
                    best = max(GRID, key=lambda p: mean_mrr(p, tr))
                    picked.append(best)
                    fold_mets.append(metrics_from_ranks([grid_ranks[best][i] for i in va], size))
                best_dev = max(GRID, key=lambda p: mean_mrr(p, dev))
                test_m = metrics_from_ranks([grid_ranks[best_dev][i] for i in test], size)
                row = {"method": name, "chunk_size": size, "best_k1": best_dev[0],
                       "best_b": best_dev[1]}
                for k in fold_mets[0]:
                    row[f"cv_{k}"] = float(np.mean([m[k] for m in fold_mets]))
                    row[f"cv_{k}_std"] = float(np.std([m[k] for m in fold_mets]))
                    row[f"test_{k}"] = test_m[k]
                rows_tuned.append(row)
        print("done size", size)

    df_a = pd.DataFrame(rows_default)
    df_b = pd.DataFrame(rows_tuned)
    df_a.to_csv(RES / "table_A_default_all_questions.csv", index=False, float_format="%.3f")
    df_b.to_csv(RES / "table_B_tuned_cv_and_test.csv", index=False, float_format="%.3f")
    pd.DataFrame(rank_rows).to_csv(RES / "per_question_ranks_default.csv", index=False)

    # ---------- paired bootstrap: is one chunk size really better? ----------
    paired = []
    for name in ["tfidf", "bm25", "bm25+stem", "bm25+stem+ov25"]:
        for a, b in itertools.combinations(PRIMARY, 2):
            for metric in ["mrr@10", "hit@1"]:
                sa = per_question_scores(store[(name, a)], a, metric)
                sb = per_question_scores(store[(name, b)], b, metric)
                diff = sb - sa
                lo, hi = bootstrap_ci(diff)
                paired.append({"method": name, "metric": metric, "size_a": a, "size_b": b,
                               "mean_diff(b-a)": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                               "significant": not (lo <= 0 <= hi)})
    pd.DataFrame(paired).to_csv(RES / "paired_bootstrap_chunk_sizes.csv", index=False,
                                float_format="%.3f")

    # ---------- does each improvement beat the TF-IDF baseline at the same chunk size? ----------
    vs_base = []
    for size in SIZES:
        for name in ["bm25", "bm25+stem", "bm25+stem+ov25"]:
            for metric in ["mrr@10", "hit@1"]:
                diff = (per_question_scores(store[(name, size)], size, metric)
                        - per_question_scores(store[("tfidf", size)], size, metric))
                lo, hi = bootstrap_ci(diff)
                vs_base.append({"chunk_size": size, "method": name, "metric": metric,
                                "mean_gain_over_tfidf": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                                "significant": not (lo <= 0 <= hi)})
    pd.DataFrame(vs_base).to_csv(RES / "paired_bootstrap_vs_baseline.csv", index=False,
                                 float_format="%.3f")

    # ---------- which questions flip between 200 and 1000 words? (hit@1) ----------
    flips = []
    for i, q in enumerate(questions):
        a = store[("bm25+stem+ov25", 200)][i]
        b = store[("bm25+stem+ov25", 1000)][i]
        flips.append({"qid": q["id"], "category": q["category"], "question": q["question"],
                      "rank_200w": a, "rank_1000w": b,
                      "only_200_correct": a == 1 and b != 1, "only_1000_correct": b == 1 and a != 1})
    pd.DataFrame(flips).to_csv(RES / "flips_200_vs_1000.csv", index=False)

    # ---------- by category ----------
    cat_rows = []
    for name in ["tfidf", "bm25+stem", "bm25+stem+ov25"]:
        for size in PRIMARY:
            r = store[(name, size)]
            for cat in sorted({q["category"] for q in questions}):
                idx = [i for i, q in enumerate(questions) if q["category"] == cat]
                m = metrics_from_ranks([r[i] for i in idx], size)
                cat_rows.append({"method": name, "chunk_size": size, "category": cat,
                                 "n": len(idx), "hit@1": m["hit@1"], "hit@3": m["hit@3"],
                                 "mrr@10": m["mrr@10"]})
    pd.DataFrame(cat_rows).to_csv(RES / "by_category.csv", index=False, float_format="%.3f")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nTABLE A (default params, all questions)\n", df_a.round(3).to_string(index=False))
    print("\nTABLE B (tuned BM25)\n", df_b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
