"""Step 1 - turn the handbook PDF into clean per-page text.

Decisions (documented in README):
* Pages 1-8 (cover, preface, table of contents) are skipped: the contents page
  lists every heading and would act as an unrealistic "keyword magnet".
* Running headers ("CHAPTER 2. ...", "2.3. EVALUATION SYSTEM ... 19") and bare
  page numbers are removed, but only at the very top/bottom of a page so that
  numbers inside tables are never touched.
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

FIRST_CONTENT_PAGE = 9          # 1-based PDF page where "1. General Information" starts
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "Hand-Book-Fas-2021-2022.pdf"
PAGES_JSON = ROOT / "data" / "handbook_pages.json"

_HEADER = re.compile(
    r"^(\d+\s+)?(CHAPTER \d+\.|APPENDIX [A-Z]\.).*$"       # chapter / appendix running header
    r"|^\d+\.\d+\.\s+[A-Z0-9 ,&/()\-\u2013]+\s*\d*$"        # "2.3. EVALUATION SYSTEM ... 19"
    r"|^\d{1,3}$"                                            # bare page number
)


def clean_page(raw: str) -> str:
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    # strip header/footer candidates only from the top and bottom of the page
    while lines and _HEADER.match(lines[0]):
        lines.pop(0)
    while lines and _HEADER.match(lines[-1]):
        lines.pop()
    return " ".join(lines)


_BROKEN = re.compile(r"^[A-Za-z]+-$")


def dehyphenate(pages):
    """Repair words that LaTeX hyphenated at a line break ("continua- tion").

    A token ending in "-" followed by a lower-case token is re-joined, unless the
    hyphenated form is a genuine compound found elsewhere in the handbook
    (e.g. "in-course") and the fused form is not.
    """
    vocab = set()
    for p in pages:
        vocab.update(t.lower() for t in p["text"].split())
    out = []
    for p in pages:
        toks, res, i = p["text"].split(), [], 0
        while i < len(toks):
            t = toks[i]
            if i + 1 < len(toks) and _BROKEN.match(t) and toks[i + 1][:1].islower():
                nxt = toks[i + 1]
                fused, hyph = (t[:-1] + nxt).lower(), (t + nxt).lower()
                if hyph in vocab and fused not in vocab:
                    res.append(t + nxt)          # real compound word, keep the hyphen
                else:
                    res.append(t[:-1] + nxt)     # soft hyphen from line wrapping
                i += 2
            else:
                res.append(t)
                i += 1
        out.append({"page": p["page"], "text": " ".join(res)})
    return out


def extract_pages(pdf_path: Path = PDF_PATH, first_page: int = FIRST_CONTENT_PAGE):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        if i < first_page:
            continue
        text = clean_page(page.extract_text() or "")
        pages.append({"page": i, "text": text})
    return dehyphenate(pages)


def load_pages(rebuild: bool = False):
    """Return cached pages, extracting from the PDF if needed."""
    if rebuild or not PAGES_JSON.exists():
        pages = extract_pages()
        PAGES_JSON.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return json.loads(PAGES_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    pages = load_pages(rebuild=True)
    n_words = sum(len(p["text"].split()) for p in pages)
    print(f"{len(pages)} pages, {n_words} words -> {PAGES_JSON}")
