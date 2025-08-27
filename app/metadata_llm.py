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
    isbn: Optional[str] = None
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

    raw_isbn = data.get("isbn")
    if raw_isbn:
        norm = normalize_isbn(raw_isbn) or {}
        # Use ISBN13 as primary, fallback to ISBN10, then original
        data["isbn"] = norm.get("isbn13") or norm.get("isbn10") or raw_isbn

    if not data.get("language"):
        data["language"] = detect_lang(full_text or "", fallback="en")

    if data.get("primary_genre"):
        data["primary_genre"] = canon(data["primary_genre"])
    data["secondary_genres"] = [canon(x) for x in (data.get("secondary_genres") or []) if x]

    return data

# NEW: Stage 1 - Basic metadata extraction without detailed content analysis
class LLMBasicMeta(BaseModel):
    title: Optional[str] = None
    title_aliases: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    year: Optional[int] = None
    isbn: Optional[str] = None
    language: Optional[str] = None
    summary: Optional[str] = None  # Basic summary from start/end
    primary_genre: Optional[str] = None
    secondary_genres: List[str] = Field(default_factory=list)
    main_topics: List[str] = Field(default_factory=list)
    mentioned_topics: List[str] = Field(default_factory=list)

def extract_basic_metadata(full_text: str) -> dict:
    """Stage 1: Extract basic metadata from start+end of long books"""
    if not OPENAI_API_KEY:
        return {}
    
    text = _sample_text(full_text or "", max_chars=9000)
    
    basic_prompt = """Extract bibliographic metadata from book text (start and end sections).
Focus on explicit mentions: title page, copyright, ISBN, author bio.
Create basic summary from available text. Don't guess genres/topics if unclear.
Normalize to lowercase tokens. Language code: 2 letters."""

    llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0, api_key=OPENAI_API_KEY)
    Structured = llm.with_structured_output(LLMBasicMeta)
    out = Structured.invoke([("system", basic_prompt), ("user", f"TEXT:\n{text}")])
    data = out.dict()

    # Process ISBN
    raw_isbn = data.get("isbn")
    if raw_isbn:
        norm = normalize_isbn(raw_isbn) or {}
        # Use ISBN13 as primary, fallback to ISBN10, then original
        data["isbn"] = norm.get("isbn13") or norm.get("isbn10") or raw_isbn

    if not data.get("language"):
        data["language"] = detect_lang(full_text or "", fallback="en")

    if data.get("primary_genre"):
        data["primary_genre"] = canon(data["primary_genre"])
    data["secondary_genres"] = [canon(x) for x in (data.get("secondary_genres") or []) if x]

    return data

# NEW: Stage 3 - Enhanced content analysis from chunks
class LLMContentAnalysis(BaseModel):
    summary: Optional[str] = None  # Comprehensive summary
    primary_genre: Optional[str] = None
    secondary_genres: List[str] = Field(default_factory=list)
    main_topics: List[str] = Field(default_factory=list)
    mentioned_topics: List[str] = Field(default_factory=list)

def enhance_metadata_from_chunks(chunks_text: str) -> dict:
    """Stage 3: Create comprehensive summary and extract themes from all content"""
    if not OPENAI_API_KEY or not chunks_text.strip():
        return {}
    
    # Limit chunks text to avoid token limits
    if len(chunks_text) > 8000:
        chunks_text = chunks_text[:8000] + "..."
    
    content_prompt = """Analyze this book content and create:
1. Comprehensive summary (5-7 sentences) covering main themes and plot
2. Primary genre based on content style and themes
3. Secondary genres if applicable
4. Main topics (key themes, subjects discussed)
5. Mentioned topics (secondary themes, references)

Focus on actual content, not just bibliographic info. Normalize genres/topics to lowercase tokens."""

    llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0, api_key=OPENAI_API_KEY)
    Structured = llm.with_structured_output(LLMContentAnalysis)
    out = Structured.invoke([("system", content_prompt), ("user", f"CONTENT:\n{chunks_text}")])
    data = out.dict()

    if data.get("primary_genre"):
        data["primary_genre"] = canon(data["primary_genre"])
    data["secondary_genres"] = [canon(x) for x in (data.get("secondary_genres") or []) if x]

    return data
