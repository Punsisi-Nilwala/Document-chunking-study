"""Step 8 - write README.md from the CSV files in results/, so the numbers in the
README can never disagree with the run that produced them.

    python -m src.make_report
"""
import pandas as pd

from .run_experiments import RES, ROOT

PRIMARY = [200, 500, 1000]
HAND = {   # hand-written diagnoses (the rank shown is read from the CSV of THIS run)
    "Q025": "Paraphrase: handbook says *effective date* ... last date of final end-semester exam; question says *officially take effect*. My toy stemmer does not map effect ~ effective.",
    "Q003": "Synonyms: *repeat* = resit, *best* = highest.",
    "Q011": "Wrong sense of *language*: top-1 is the award rule mentioning *English Language I and II*; the handbook phrase is *medium of instruction*.",
    "Q012": "Handbook: \"attendance outside the examination hall at least 15 minutes before ...\"; almost no shared content words.",
    "Q038": "Paraphrase (*retake / missed* vs *resit / unable to sit*); top-1 is a syllabus evaluation table repeating *In-Course Assessments* and *End-Semester Examination*.",
    "Q027": "Generic words (*hours, course, credit*) matched the Health Centre sentence about *office hours*; the right sentence is \"15 hours of lectures is considered as one credit\".",
    "Q028": "Generic-word distractor: the paragraph \"Opting for Bachelors Degree\" is dense in *degree, programme, honours* and outranks the specific sentence.",
    "Q029": "Same distractor as Q028 (\"Opting for Bachelors Degree\" paragraph wins).",
    "Q034": "Same distractor as Q028; the answer sentence is short (\"two months full-time considered as one credit\") and shares few query words.",
    "Q009": "Top-1 is the grade-point table (matches *grade, marks, course*); the weighting sentence (40% / 60%) does not use those words.",
    "Q008": "Partial false negative: top-1 is a course syllabus with the same 30% / 70% split, not the general regulation.",
    "Q036": "Table-header words (*Level, Semester, Credits*) in the IT curriculum table outrank the Environmental Science elective rule.",
    "Q039": "Partial false negative: top-1 holds the *Mathematics* line of the same entry-requirement list; the *English* line is in the neighbouring chunk.",
    "Q059": "Sibling text: top-1 is the Physical Science training split (daily diary 30 / report 40 / presentation 30); the Environmental Science split is in a different table.",
    "Q073": "The answer is a short table row (\"English Language I Level 1 Semester I 3\"); an award-rule paragraph full of *credits, core, level* wins.",
    "Q100": "Related but different rule: top-1 is 4.4 *Record Book as Identity*, which also mentions the Admission Card; the gold sentence is in the services list.",
    "Q072": "Tokenisation + numbers: grade symbols like \"A+\" / \"A-\" reduce to the stop-word *a*; the score range cannot be matched arithmetically.",
    "Q070": "Numbers: \"32 marks\" can never lexically match the range \"30-34\".",
    "Q032": "Sibling text: Physical Science chapter says \"sixteen weeks\", Bio-science says \"15 weeks\"; sentence shape identical, programme name is a weak signal.",
    "Q061": "Same words (\"English Language I and II\", \"grade C\") appear in the award-of-degree rules; right vocabulary, wrong section.",
    "Q098": "Query words *faculty, applied, science* occur everywhere; the plagiarism paragraph repeats them more densely.",
    "Q005": "Topical distractor: class-award paragraphs repeat \"within three academic years\" and out-score the \"maximum duration ... five academic years\" sentence.",
    "Q006": "Same distractor as Q005 (\"academic years\" in the award-of-class paragraphs).",
    "Q026": "Metric false negative: top-1 states the same fact (\"One credit shall constitute 50 notional learning hours\") in another section.",
    "Q055": "Metric false negative: top-1 (IT industrial-training syllabus) gives the same 30/40/30 split.",
    "Q074": "Metric false negative: top-1 is the course's own syllabus page, which states the credits and hours too.",
}


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v)
                                      for v in r) + " |")
    return "\n".join(out)


def main():
    A = pd.read_csv(RES / "table_A_default_all_questions.csv")
    B = pd.read_csv(RES / "table_B_tuned_cv_and_test.csv")
    G = pd.read_csv(RES / "paired_bootstrap_vs_baseline.csv")
    P = pd.read_csv(RES / "paired_bootstrap_chunk_sizes.csv")
    C = pd.read_csv(RES / "by_category.csv")
    F = pd.read_csv(RES / "flips_200_vs_1000.csv")
    E = pd.read_csv(RES / "error_analysis_bm25_stem_500w_ov25.csv")

    best = "bm25+stem+ov25"
    a = A[A.chunk_size.isin(PRIMARY) & (A.method != "random")].copy()
    a = a[["method", "chunk_size", "hit@1", "hit@3", "mrr@10", "hit@1000w"]]
    a.columns = ["method", "size", "Hit@1", "Hit@3", "MRR@10", "Hit @1000-word budget"]
    rnd = A[(A.method == "random") & (A.chunk_size == 500)].iloc[0]
    with open(ROOT / "data" / "handbook_pages.json", encoding="utf-8") as fh:   # utf-8: Windows default is cp1252
        pages = __import__("json").load(fh)
    n_words = sum(len(p["text"].split()) for p in pages)

    b = B[(B.method == best) & B.chunk_size.isin(PRIMARY)]
    b = pd.DataFrame({"size": b.chunk_size,
                      "CV MRR (mean +- sd, 5 folds)": [f"{m:.3f} +- {s:.3f}" for m, s in zip(b["cv_mrr@10"], b["cv_mrr@10_std"])],
                      "TEST MRR": b["test_mrr@10"], "TEST Hit@1": b["test_hit@1"]})

    gains = G[(G.method == best) & (G.metric == "mrr@10")].sort_values("chunk_size")
    gain_lines = "\n".join(
        f"   * {int(r.chunk_size)} words: {r.mean_gain_over_tfidf:+.3f} MRR, 95 % CI [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] - "
        f"{'significant' if r.significant else 'NOT significant (CI contains 0)'}"
        for _, r in gains.iterrows())

    def pdiff(method, a_, b_, metric="mrr@10"):
        r = P[(P.method == method) & (P.size_a == a_) & (P.size_b == b_) & (P.metric == metric)].iloc[0]
        return (f"{r['mean_diff(b-a)']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                f"{'' if r.significant else ' (not significant)'}")

    ab = A[(A.method == best) & A.chunk_size.isin(PRIMARY)].set_index("chunk_size")
    cov = A[A.method == "bm25+stem"].set_index("chunk_size")["evidence_covered"]
    cat = C[(C.method == best)].pivot(index="category", columns="chunk_size", values="hit@1")[PRIMARY]
    cat.insert(0, "n questions", C[(C.method == best) & (C.chunk_size == 200)].set_index("category")["n"])
    cat = cat.reset_index()
    cat.columns = ["category", "n questions", "Hit@1 @200", "Hit@1 @500", "Hit@1 @1000"]
    only200, only1000 = int(F.only_200_correct.sum()), int(F.only_1000_correct.sum())
    top_hit1 = ab["hit@1"].idxmax()
    n_wrong = len(E)
    tags = E.auto_tag.str.split(" ").str[0].value_counts()
    cats = E.category.value_counts()

    rows = []
    for qid, why in HAND.items():
        m = E[E.qid == qid]
        if len(m):
            rows.append((qid, m.iloc[0].question, int(m.iloc[0].rank_of_first_relevant), why))
    hand = "| qid | question | rank of right chunk | diagnosis |\n|---|---|---|---|\n" + "\n".join(
        f"| {q} | {t} | {r} | {w} |" for q, t, r, w in rows)

    text = f"""# Document chunking study - how big should a chunk be?

**Question (one sentence):** *When we split the Faculty of Applied Science student handbook into fixed-size word chunks and retrieve with a keyword search, does a chunk size of 200, 500 or 1000 words find the passage that answers a student's question most reliably?*

Everything is reproducible with one command (`python -m src.run_all`, about a minute, CPU only, no internet, no LLM or external API). **All numbers below are generated from `results/*.csv` by `src/make_report.py`.**

## 1. Data
* `data/Hand-Book-Fas-2021-2022.pdf` - FAS, University of Vavuniya handbook (252 pages: regulations, curricula, exam rules, services, detailed syllabi). Public document; no student or staff records.
* Pages 1-8 (cover, preface, contents) are dropped; running headers/page numbers removed; words broken by LaTeX line-hyphenation ("continua- tion") are repaired -> {len(pages)} pages, {n_words:,} words in this run.
* **103 hand-written questions** (`data/questions.json`) in 5 categories (fact 39, rule 28, facility 14, syllabus 12, table 10). Each has 1-2 **gold evidence phrases** copied from the handbook. A chunk is *relevant* if it contains a gold phrase, so the label depends only on the text and every chunk size is judged by the same rule. `python -m src.validate_questions` checks that each phrase exists in the corpus.

## 2. Baseline and methods
| name | what it is |
|---|---|
| `random` | random chunk order (sanity floor, 50 repetitions) |
| **`tfidf` (BASELINE)** | TF-IDF + cosine similarity (scikit-learn), fixed chunks, no overlap |
| `bm25` | Okapi BM25 written from scratch (`src/retrieval.py`) |
| `bm25+stem` | + tiny suffix stemmer (resitting ~ resit, credits ~ credit) |
| `{best}` | + 25 % overlap between consecutive chunks |

Chunk sizes: **200 / 500 / 1000 words (required)**, plus 100 and 300 as extra points in the CSVs and figure.

## 3. Evaluation protocol
* Metrics: **Hit@1, Hit@3, Hit@5, MRR@10** and **Hit within a 1000-word budget** (top-k with k = ceil(1000 / size): top-5 for 200-word chunks, top-2 for 500, top-1 for 1000). Plain top-k is unfair across sizes because three 1000-word chunks hand the reader 5x more text than three 200-word chunks.
* **Table A** (cleanest chunk-size comparison): default parameters, no tuning, all 103 questions, bootstrap CIs, paired bootstrap for differences.
* **Table B**: stratified 70/30 split -> DEV 72 / **held-out TEST 31**. BM25 `k1, b` chosen by **5-fold CV inside DEV** (select on training folds, score on the held-out fold), refit on DEV, scored once on TEST.
* **Error analysis:** every question whose top-1 chunk is wrong is saved with an automatic tag (`results/error_analysis_*.csv`) and discussed by hand in Section 5.

## 4. Results
**Table A** - all 103 questions, default parameters (`results/table_A_default_all_questions.csv`). Random order at 500 words scores Hit@1 {rnd['hit@1']:.3f}, Hit@3 {rnd['hit@3']:.3f}.

{md(a)}

![chunk size effect](results/fig_chunk_size_effect.png)

**Table B** - `{best}`, tuned by CV on DEV, one-shot TEST (n = 31) (`results/table_B_tuned_cv_and_test.csv`):

{md(b)}

**What the numbers say**
1. **Baseline vs improvement** (`{best}` minus TF-IDF, same chunk size, paired bootstrap on MRR@10, `results/paired_bootstrap_vs_baseline.csv`):
{gain_lines}
2. **The answer to "which chunk size is best" depends on how you count.** For `{best}`, Hit@3 goes {ab.loc[200,'hit@3']:.3f} -> {ab.loc[500,'hit@3']:.3f} -> {ab.loc[1000,'hit@3']:.3f} for 200 -> 500 -> 1000 words (bigger looks better), but Hit within a 1000-word budget goes {ab.loc[200,'hit@1000w']:.3f} -> {ab.loc[500,'hit@1000w']:.3f} -> {ab.loc[1000,'hit@1000w']:.3f} (smaller looks better). Large chunks win mostly because they cover more text per rank position, not because they are ranked more precisely. Best Hit@1 for this system is at **{top_hit1} words** ({ab.loc[top_hit1,'hit@1']:.3f}).
3. **Are the chunk-size differences real?** Paired bootstrap on MRR@10 for `{best}`: 200 vs 500 {pdiff(best,200,500)}; 500 vs 1000 {pdiff(best,500,1000)}; 200 vs 1000 {pdiff(best,200,1000)}. For the TF-IDF baseline: 200 vs 1000 {pdiff('tfidf',200,1000)}. Intervals that contain 0 mean I cannot claim a winner. TEST has only 31 questions, so its sampling noise is large (about +-0.15).
4. **Question type matters** (Hit@1 of `{best}`, `results/by_category.csv`):

{md(cat)}

   {only200} questions are answered at rank 1 only by 200-word chunks and {only1000} only by 1000-word chunks (`results/flips_200_vs_1000.csv`).
5. **Overlap fixes boundary cuts.** Gold phrases that survive intact in at least one chunk without overlap (`bm25+stem`): {cov[100]} at 100 words, {cov[200]} at 200, {cov[500]} at 500, {cov[1000]} at 1000. With 25 % overlap every phrase is intact at every size (`evidence_covered` column in Table A).

**Practical reading (for this handbook, keyword retrieval):** differences between 200, 500 and 1000 words are small for the best system and mostly not statistically reliable; the robust effects are (a) BM25+stem+overlap beats the TF-IDF baseline at small/medium sizes and (b) the direction of the trade-off between top-k recall and context size (point 2). 300-500 words with 25 % overlap is a sensible compromise: good Hit@1 while keeping retrieved context short.

## 5. Error analysis ({n_wrong} wrong top-1 predictions of 103 for `{best}` at 500 words)
Full list with the offending text: `results/error_analysis_bm25_stem_500w_ov25.csv` (also 200 and 1000 words). Automatic first-guess tags: {', '.join(f'{k} {v}' for k, v in tags.items())}. By category: {', '.join(f'{k} {v}' for k, v in cats.items())}.

Hand-read diagnosis of the failures that appear in this run (ranks are from this run):

{hand}

**Patterns.** (1) *Paraphrase / vocabulary gaps* - the dominant problem for a lexical retriever (resit~retake, medium of instruction~language, effective date~take effect). (2) *Sibling text* - the handbook repeats the same regulation wording for Bio-science and Physical Science programmes and for many course syllabi, so BM25 often returns the right sentence from the wrong programme. (3) *Numbers, tables and tokenisation* - grade symbols such as A+ become stop-words and a mark like 32 cannot match the range 30-34. (4) *Generic-word distractors* - paragraphs dense in common words ("academic years", "credits") outrank the specific sentence. (5) *Metric false negatives* - some wrong top-1 chunks really do contain a correct answer (the same fact is stated elsewhere), so the strict gold-phrase metric slightly under-estimates every system.
**What would fix them:** query expansion or dense embeddings for paraphrases, structure-aware chunks that keep the programme heading with each paragraph, a tokeniser that keeps grade symbols, and a table-aware parser.

## 6. Limitations
* 103 questions written by one person -> wide confidence intervals; TEST has only 31 questions. Chunk-size conclusions are suggestive, not definitive.
* Relevance = "chunk contains a gold phrase"; other chunks may also answer the question. Questions were written after reading the handbook, so they are somewhat vocabulary-aligned; real students would paraphrase more and scores would be lower. (While checking the error file I also found that five grade-table questions only listed one of the handbook's two grade tables as correct; this is fixed in `questions.json`.)
* Retrieval only: no LLM answer generation and no dense embeddings were evaluated (no internet in the build environment). Dense retrieval may shift the best chunk size.
* One document, one language, one retrieval family. Fixed word windows ignore headings and sentence boundaries.
* The stemmer is a 20-line toy, not Porter; the BM25 tuning grid is small (12 settings).
* Extracted text depends slightly on the pypdf version, so exact numbers can differ by a few hundredths between machines.

## 7. Privacy
Only the public handbook is used. Nothing is sent to any external service; all computation is local. Do not add real student or staff records to `data/`.

## 8. Reproduce
```bash
pip install -r requirements.txt     # or: python -m pip install --user -r requirements.txt
python -m src.run_all               # extract -> validate -> experiments -> figures -> error analysis -> this README
python -m src.error_analysis bm25 500 0.25   # error table for any (method, size, overlap)
python -m pytest -q tests           # unit tests (chunking, metrics, retrievers, split)
```
Layout: `src/extract.py` - `chunking.py` - `retrieval.py` (TF-IDF, BM25) - `evaluate.py` - `run_experiments.py` - `error_analysis.py` - `make_figures.py` - `make_report.py`; `data/questions.json`; `results/`. Seed = 42 everywhere.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")
    print("README.md written from results/")


if __name__ == "__main__":
    main()
