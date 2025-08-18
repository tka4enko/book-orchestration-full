import re, unicodedata, json
from langdetect import detect as lang_detect, DetectorFactory
DetectorFactory.seed = 42

def strip_diacritics(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s or '') if unicodedata.category(c) != 'Mn')

def canon(s: str) -> str:
    s = strip_diacritics(s or '').lower()
    return re.sub(r'\s+', ' ', s).strip()

def to_list(x):
    if x is None: return []
    if isinstance(x, str):
        try:
            j = json.loads(x)
            return j if isinstance(j, list) else ([x] if x else [])
        except Exception:
            return [x] if x else []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def detect_lang(text: str, fallback: str = "en") -> str:
    try:
        l = lang_detect((text or "")[:2000])
        return (l or fallback)[:2]
    except Exception:
        return (fallback or "en")[:2]

def cheap_summary(text: str, limit: int = 600) -> str:
    if not text: return ""
    parts = [p.strip() for p in text.split("\n") if len(p.strip()) > 60]
    return " ".join(parts[:2])[:limit]

def build_master_meta(base: dict) -> dict:
    title = (base.get("title") or "").strip()
    author = (base.get("author") or None)
    language = base.get("language") or None
    if not language:
        language = detect_lang(f"{title}\n{base.get('summary') or ''}", fallback="en")

    secondary = to_list(base.get("secondary_genres"))
    main_topics = to_list(base.get("main_topics"))
    mentioned = to_list(base.get("mentioned_topics"))

    return {
        "document_id": base.get("document_id") or base.get("id") or "",
        "title": title,
        "author": author,
        "year": base.get("year"),
        "isbn13": base.get("isbn13") or base.get("isbn"),
        "isbn10": base.get("isbn10"),
        "language": language[:2] if language else None,
        "summary": base.get("summary") or cheap_summary(base.get("full_text") or ""),
        "file_type": base.get("file_type") or "txt",
        "primary_genre": base.get("primary_genre"),
        "secondary_genres": secondary,
        "main_topics": main_topics,
        "mentioned_topics": mentioned,
        "is_master_chunk": True,
        "title_canonical": canon(title) if title else None,
        "author_canonical": canon(author) if author else None,
        "primary_genre_canonical": canon(base.get("primary_genre")) if base.get("primary_genre") else None,
    }
