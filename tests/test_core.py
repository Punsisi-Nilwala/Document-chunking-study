import numpy as np

from src.chunking import chunk_words
from src.evaluate import first_relevant_rank, metrics_from_ranks, split_dev_test
from src.retrieval import BM25Retriever, TfidfRetriever
from src.text_utils import normalize, stem, tokenize


def test_chunk_sizes_and_coverage():
    words = [f"w{i}" for i in range(1050)]
    chunks = chunk_words(words, [1] * 1050, size=500)
    assert [c.n_words for c in chunks] == [500, 500, 50]
    assert " ".join(c.text for c in chunks).split() == words      # nothing lost


def test_overlap_stride():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_words(words, [1] * 100, size=40, overlap=0.5)
    assert chunks[1].start_word == 20                             # stride = 20
    assert chunks[-1].end_word == 100


def test_normalize_and_stem():
    assert normalize("Grade \u2018C\u2013\u2019  is   enough") == "grade c- is enough"
    assert stem("resitting") == "resit" and stem("credits") == "credit"
    assert "the" not in tokenize("the minimum grade")


def test_retrievers_find_obvious_chunk():
    texts = ["cats and dogs", "minimum attendance is eighty percent", "weather is nice"]
    for r in (TfidfRetriever(), BM25Retriever()):
        r.fit(texts)
        assert r.rank("what attendance is required")[0] == 1


def test_metrics():
    assert first_relevant_rank([3, 1, 2], {2}) == 3
    m = metrics_from_ranks([1, 2, None, 10], chunk_size=500)
    assert m["hit@1"] == 0.25 and m["hit@3"] == 0.5


def test_split_is_disjoint():
    qs = [{"category": "a"}] * 10 + [{"category": "b"}] * 10
    dev, test = split_dev_test(qs)
    assert not set(dev) & set(test) and len(dev) + len(test) == 20
