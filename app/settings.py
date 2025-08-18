import os
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
OPENAI_MODEL_EMBED = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

BM25_K = 8
VEC_BOOKS_K = 8
VEC_CONTENT_K = 6
RRF_K = 60

# Legacy similarity threshold for filtering irrelevant results
MIN_SIMILARITY_THRESHOLD = 0.6

# JSON configuration loading function
def _load_json_config(env_var: str, default: dict) -> dict:
    try:
        config_str = os.getenv(env_var)
        if config_str:
            return json.loads(config_str)
        return default
    except json.JSONDecodeError:
        return default

# Language threshold modifiers (loaded from JSON)
LANGUAGE_THRESHOLD_MODIFIERS = _load_json_config("LANGUAGE_THRESHOLD_MODIFIERS", {
    "same_language": -0.1,
    "different_language": 0.0,
    "unknown_language": -0.05
})

# Intent-based similarity thresholds (loaded from JSON)
SIMILARITY_THRESHOLDS = _load_json_config("SIMILARITY_THRESHOLDS", {
    "author": 0.8,
    "author_title": 0.8,
    "isbn": 0.9,
    "title": 0.7,
    "topic": 0.4,
    "genre": 0.5,
    "free_text": 0.5,
    "clarify": 0.3
})

# Intent-based BM25 thresholds (loaded from JSON)
BM25_THRESHOLDS = _load_json_config("BM25_THRESHOLDS", {
    "isbn": 0.05,
    "author": 0.1,
    "title": 0.1,
    "author_title": 0.1,
    "genre": 0.1,
    "topic": 0.6,
    "free_text": 0.4,
    "clarify": 0.3
})

# safe chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
MAX_CHUNKS = 80

# Deduplication settings
MAX_CHUNKS_PER_BOOK = int(os.getenv("MAX_CHUNKS_PER_BOOK", "2"))

# Content search diversification settings
CHUNKS_PER_BOOK_IN_CONTENT = int(os.getenv("CHUNKS_PER_BOOK_IN_CONTENT", "2"))
CONTENT_SEARCH_EXPAND_K = int(os.getenv("CONTENT_SEARCH_EXPAND_K", "20"))
