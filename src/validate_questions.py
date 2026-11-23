"""Sanity check: every gold-evidence phrase must occur in the cleaned handbook."""
import json
from pathlib import Path

from .chunking import pages_to_words
from .extract import load_pages
from .text_utils import normalize

ROOT = Path(__file__).resolve().parents[1]


def validate(verbose=True):
    words, _ = pages_to_words(load_pages())
    corpus = normalize(" ".join(words))
    questions = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    bad = []
    for q in questions:
        hits = {e: corpus.count(normalize(e)) for e in q["evidence"]}
        q["n_matches"] = sum(hits.values())
        if q["n_matches"] == 0:
            bad.append(q)
        if verbose and q["n_matches"] != 1:
            print(q["id"], q["n_matches"], hits)
    print(f"{len(questions)} questions, {len(bad)} with NO evidence match")
    return bad


if __name__ == "__main__":
    validate()
