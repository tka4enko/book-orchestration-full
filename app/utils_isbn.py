import re
from isbnlib import canonical as isbn_canonical, to_isbn10

ISBN_RX = re.compile(r"(?i)\b(97[89][\s\-]?(?:\d[\s\-]?){9}\d|\d{9}[\dXx])\b")

def normalize_isbn(raw: str):
    if not raw: return None
    s = re.sub(r"[^0-9Xx]", "", raw)
    try:
        i13 = isbn_canonical(s)
        i10 = to_isbn10(i13) if i13 else None
        return {"isbn13": i13, "isbn10": i10}
    except Exception:
        return None

def extract_first_isbn(text: str):
    m = ISBN_RX.search(text or "")
    if not m: return None
    return normalize_isbn(m.group(0))
