"""Small text helpers shared by every other module.

Two kinds of normalisation are used in this project:

* ``normalize``  - for *matching* gold evidence phrases against chunk text.
* ``tokenize``   - for *retrieval* (bag-of-words used by TF-IDF / BM25).
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Typographic dashes -> "-", typographic and plain quotes -> removed.
_CHAR_MAP = {ord(c): "-" for c in "\u2013\u2014\u2212\u2010\u2011"}
_CHAR_MAP.update({ord(c): None for c in "\u2018\u2019\u201c\u201d'\""})
# Control / private-use glyphs (LaTeX bullets such as \x88) -> space
_PUA = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\ue000-\uf8ff\u2022\u25cf\u25aa]")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


def normalize(text: str) -> str:
    """Lower-case, unify dashes/quotes, drop bullet glyphs, collapse whitespace."""
    text = text.translate(_CHAR_MAP)
    text = _PUA.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def stem(token: str) -> str:
    """A deliberately tiny suffix stripper (we have no internet for NLTK).

    It only needs to map e.g. ``resitting``/``resit`` or ``credits``/``credit``
    onto the same term; it is NOT a full Porter stemmer.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    for suffix, replacement in (("ies", "y"), ("sses", "ss"), ("ing", ""),
                                ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and not (suffix == "s" and token.endswith("ss")):
            base = token[: len(token) - len(suffix)] + replacement
            if len(base) >= 3:
                # undo doubled consonant: "resitt" -> "resit"
                if suffix in ("ing", "ed") and len(base) > 3 and base[-1] == base[-2]:
                    base = base[:-1]
                return base
    return token


def tokenize(text: str, use_stem: bool = False) -> list:
    """Lower-cased alphanumeric tokens, English stop-words removed."""
    tokens = [t for t in _TOKEN.findall(normalize(text)) if t not in STOPWORDS]
    return [stem(t) for t in tokens] if use_stem else tokens
