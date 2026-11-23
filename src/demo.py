"""Try the retriever yourself:  python -m src.demo "how many times can I resit a course unit"
Prints the top-3 chunks (page numbers + the sentence that best matches the question)."""
import sys

from .chunking import chunk_words, pages_to_words
from .error_analysis import best_sentence
from .extract import load_pages
from .retrieval import BM25Retriever
from .text_utils import tokenize

SIZE, OVERLAP = 500, 0.25


def main():
    question = " ".join(sys.argv[1:]) or "What is the minimum attendance required for exams?"
    words, word_pages = pages_to_words(load_pages())
    chunks = chunk_words(words, word_pages, SIZE, OVERLAP)
    ret = BM25Retriever(1.5, 0.75, use_stem=True).fit([c.text for c in chunks])
    print(f"Question: {question}\n({len(chunks)} chunks of ~{SIZE} words, 25% overlap)\n")
    q_tokens = tokenize(question, True)
    for rank, idx in enumerate(ret.rank(question)[:3], start=1):
        c = chunks[idx]
        print(f"#{rank}  PDF pages {c.page_start}-{c.page_end}  (chunk {c.idx}, {c.n_words} words)")
        print("   best sentence:", best_sentence(c.text, q_tokens), "\n")


if __name__ == "__main__":
    main()
