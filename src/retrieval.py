"""Step 3 - retrievers.  All share one interface:

    r = Retriever(...).fit(list_of_chunk_texts)
    order = r.rank(query)      # numpy array of chunk indices, best first
"""
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .text_utils import tokenize


class TfidfRetriever:
    """BASELINE: cosine similarity between TF-IDF vectors (scikit-learn)."""

    def __init__(self, use_stem: bool = False):
        self.use_stem = use_stem

    def fit(self, texts):
        self.vec = TfidfVectorizer(analyzer=lambda t: tokenize(t, self.use_stem))
        self.matrix = self.vec.fit_transform(texts)
        return self

    def scores(self, query):
        return linear_kernel(self.vec.transform([query]), self.matrix).ravel()

    def rank(self, query):
        # stable sort -> ties resolved by earlier chunk, deterministic
        return np.argsort(-self.scores(query), kind="stable")


class BM25Retriever:
    """Okapi BM25 written from scratch (so every line can be explained in the viva)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_stem: bool = False):
        self.k1, self.b, self.use_stem = k1, b, use_stem

    def fit(self, texts):
        docs = [tokenize(t, self.use_stem) for t in texts]
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avg_len = self.doc_len.mean()
        self.postings = defaultdict(list)          # term -> [(doc_id, tf), ...]
        for doc_id, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((doc_id, tf))
        self.idf = {t: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}
        return self

    def scores(self, query):
        s = np.zeros(self.n_docs)
        for term in set(tokenize(query, self.use_stem)):
            if term not in self.postings:
                continue
            for doc_id, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len)
                s[doc_id] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return s

    def rank(self, query):
        return np.argsort(-self.scores(query), kind="stable")
