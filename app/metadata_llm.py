from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .utils_isbn import normalize_isbn
from .metadata import detect_lang, canon

class LLMBookMeta(BaseModel):
    title: Optional[str] = None
    title_aliases: List[str] = Field(default_factory=list)  # Alternative titles in other languages
    author: Optional[str] = None
    year: Optional[int] = None
    isbn13: Optional[str] = None
    isbn10: Optional[str] = None
    language: Optional[str] = None
    summary: Optional[str] = None

    primary_genre: Optional[str] = None
    secondary_genres: List[str] = Field(default_factory=list)

    main_topics: List[str] = Field(default_factory=list)
    mentioned_topics: List[str] = Field(default_factory=list)

def _sample_text(text: str, max_chars: int = 9000) -> str:
    if not text: return ""
    text = text.replace("\r", " ").replace("\t", " ")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[- max_chars // 2 :]
    return head + "\n...\n" + tail

PROMPT = """Extract bibliographic metadata from book text using the provided schema.
If unknown, return null/[] and DO NOT guess. Prefer explicit mentions (title/copyright/ISBN).
For title_aliases: include alternative titles in other languages if mentioned (e.g. original vs translated titles).
Return concise summary (3-5 sentences). Normalize genres/topics to lowercase tokens. Language code: 2 letters.
"""

def extract_metadata_llm(full_text: str) -> dict:
    if not OPENAI_API_KEY:
        return {}
    text = _sample_text(full_text or "", max_chars=9000)

    llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0, api_key=OPENAI_API_KEY)
    Structured = llm.with_structured_output(LLMBookMeta)
    out = Structured.invoke([("system", PROMPT), ("user", f"TEXT:\n{text}")])
    data = out.dict()

    raw_isbn = data.get("isbn13") or data.get("isbn10")
    if raw_isbn:
        norm = normalize_isbn(raw_isbn) or {}
        data["isbn13"] = norm.get("isbn13")
        data["isbn10"] = norm.get("isbn10")

    if not data.get("language"):
        data["language"] = detect_lang(full_text or "", fallback="en")

    if data.get("primary_genre"):
        data["primary_genre"] = canon(data["primary_genre"])
    data["secondary_genres"] = [canon(x) for x in (data.get("secondary_genres") or []) if x]

    return data
