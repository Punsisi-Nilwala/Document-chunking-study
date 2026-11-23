"""Step 2 - fixed-size word chunking with optional overlap."""
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    idx: int
    text: str
    start_word: int
    end_word: int
    page_start: int
    page_end: int

    @property
    def n_words(self) -> int:
        return self.end_word - self.start_word


def pages_to_words(pages):
    """Flatten pages to one word list plus a parallel list of page numbers."""
    words, word_pages = [], []
    for p in pages:
        toks = p["text"].split()
        words.extend(toks)
        word_pages.extend([p["page"]] * len(toks))
    return words, word_pages


def chunk_words(words: List[str], word_pages: List[int], size: int,
                overlap: float = 0.0) -> List[Chunk]:
    """Slide a window of ``size`` words with stride ``size * (1 - overlap)``.

    The last window is kept even if shorter than ``size`` (so no text is lost).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(round(size * (1 - overlap))))
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(Chunk(len(chunks), " ".join(words[start:end]), start, end,
                            word_pages[start], word_pages[end - 1]))
        if end == len(words):
            break
        start += stride
    return chunks
