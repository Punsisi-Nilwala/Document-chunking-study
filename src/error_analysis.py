"""Step 6 - error analysis: look at every question whose top-1 chunk is wrong.

    python -m src.error_analysis                 # default config below
    python -m src.error_analysis bm25 500 0.25   # method, chunk size, overlap

Each failure gets an automatic *first-guess* tag; the README then discusses the
patterns by hand after reading the rows.
"""
import sys

import pandas as pd

from .chunking import chunk_words, pages_to_words
from .evaluate import first_relevant_rank, load_questions, relevant_chunks
from .extract import load_pages
from .retrieval import BM25Retriever
from .run_experiments import RES
from .text_utils import normalize, tokenize


def best_sentence(text, target_tokens):
    """Sentence of ``text`` sharing most tokens with ``target_tokens`` (for human reading)."""
    import re
    best, score = "", -1
    for sent in re.split(r"(?<=[.;:])\s+", text):
        sc = len(set(tokenize(sent, True)) & set(target_tokens))
        if sc > score:
            best, score = sent, sc
    return best[:260]


def overlap_ratio(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    return len(a & b) / len(a) if a else 0.0


def analyse(size=500, overlap=0.25, use_stem=True, k1=1.5, b=0.75):
    pages = load_pages()
    words, word_pages = pages_to_words(pages)
    chunks = chunk_words(words, word_pages, size, overlap)
    ret = BM25Retriever(k1, b, use_stem).fit([c.text for c in chunks])
    rows = []
    for q in load_questions():
        rel = relevant_chunks(chunks, q["evidence"])
        order = ret.rank(q["question"])
        rank = first_relevant_rank(order, rel) if rel else None
        if rank == 1:
            continue
        top = chunks[order[0]]
        evid = q["evidence"][0]
        q_tok = tokenize(q["question"], use_stem)
        e_tok = tokenize(evid, use_stem)
        top_has_e = overlap_ratio(e_tok, tokenize(top.text, use_stem))  # evidence words in wrong top-1
        top_q_cov = overlap_ratio(q_tok, tokenize(top.text, use_stem))  # question words found in wrong top-1
        gold_q_cov = max((overlap_ratio(q_tok, tokenize(chunks[i].text, use_stem)) for i in rel),
                         default=0.0)                                    # ... and in the best gold chunk
        if not rel:
            tag = "evidence_split_by_chunk_boundary"
        elif gold_q_cov < 0.6:
            tag = "vocabulary_gap (gold chunk lacks most question words)"
        elif top_has_e >= 0.8:
            tag = "sibling_clause (same wording in another programme/section)"
        else:
            tag = "competing_chunk (other chunk matches query terms better)"
        rows.append({
            "qid": q["id"], "category": q["category"], "question": q["question"],
            "gold_evidence": evid, "rank_of_first_relevant": rank,
            "gold_pages": sorted({chunks[i].page_start for i in rel})[:4],
            "top1_pages": f"{top.page_start}-{top.page_end}",
            "q_words_in_gold_chunk": round(gold_q_cov, 2),
            "q_words_in_top1": round(top_q_cov, 2),
            "evidence_words_in_top1": round(top_has_e, 2),
            "auto_tag": tag,
            "top1_best_sentence": best_sentence(top.text, e_tok),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    method = args[0] if args else "bm25"
    size = int(args[1]) if len(args) > 1 else 500
    ov = float(args[2]) if len(args) > 2 else 0.25
    df = analyse(size, ov)
    out = RES / f"error_analysis_{method}_stem_{size}w_ov{int(ov*100)}.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} wrong top-1 predictions -> {out}")
    print(df["auto_tag"].value_counts().to_string())
    print(df.groupby("category").size().to_string())
