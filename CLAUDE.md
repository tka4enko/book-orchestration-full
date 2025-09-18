# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BookBot Full is an intelligent book search system with hybrid search algorithms, chat interface, and multi-level duplicate detection. It combines BM25 keyword search with semantic vector search, uses LLM for intent detection, and provides both REST API and WebSocket chat interfaces.

## Core Architecture

### Key Components
- **FastAPI server** (`main.py`) - HTTP/WebSocket endpoints with health checks
- **LangGraph orchestrators** - Two separate pipelines for search vs chat
  - `orchestrator.py` - Classic search pipeline (detect intent → route search → answer)
  - `orchestrator_with_chat.py` - Chat pipeline with context and history
- **Hybrid retrievers** (`retrievers.py`) - BM25 + vector search with adaptive thresholds
- **Document ingestion** (`ingest.py`) - PDF/DOCX/TXT processing with duplicate detection
- **Duplicate detection** (`duplicate_detection.py`) - 4-level system (file hash, ISBN, metadata, content)
- **Vector stores** - ChromaDB with separate "books" (master records) and "content" (chunks) collections

### Search Strategy
The system uses a multi-stage approach:
1. **BM25 search** - Fast keyword matching with priority
2. **Vector search** - Semantic search excluding BM25 results  
3. **Content search** - Diversified chunk search with per-book limits
4. **Smart deduplication** - Merge master records with best chunks

## Common Commands

### Development
```bash
# Start the server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Install dependencies
pip install -r requirements.txt
```

### Testing Search
```bash
# Test search endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"найди книги Орвелла"}'

# Test chat WebSocket (see chat_test.html)
# WebSocket endpoint: ws://localhost:8000/ws/chat_test
```

### Document Management
```bash
# Upload document
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@book.pdf" \
  -F "meta={\"title\":\"Title\",\"author\":\"Author\"}" \
  -F "prefer_llm=true"

# View collections
curl "http://localhost:8000/vector-store/collection/books?limit=5"
curl "http://localhost:8000/vector-store/collection/content?limit=5"
```

## Configuration

### Required Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_CHAT=gpt-4o-mini  
OPENAI_MODEL_EMBED=text-embedding-3-small
CHROMA_DIR=.chroma
```

### Optional Performance Settings
```bash
# Search parameters
MAX_CHUNKS_PER_BOOK=2
CHUNKS_PER_BOOK_IN_CONTENT=2
CONTENT_SEARCH_EXPAND_K=20

# Dynamic thresholds (JSON format)
SIMILARITY_THRESHOLDS='{"author": 0.8, "title": 0.7, "topic": 0.4, "free_text": 0.5}'
BM25_THRESHOLDS='{"author": 0.1, "title": 0.1, "topic": 0.6, "free_text": 0.4}'
LANGUAGE_THRESHOLD_MODIFIERS='{"same_language": -0.1, "different_language": 0.0}'
```

### LangSmith Tracing (Optional)
```bash
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bookbot
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## Intent Detection System

The system recognizes these query intents:
- **isbn** - ISBN number (auto-detected via regex)
- **author_title** - Author + book title combination  
- **author** - Author search only
- **title** - Book title only
- **genre** - Genre/category search
- **topic** - Subject/theme search
- **free_text** - General queries
- **clarify** - Needs user clarification

## Data Structure

### Books Collection (Master Records)
Master records in ChromaDB with enriched metadata:
- `is_master_chunk: true` 
- Enhanced text: "Title — Description — Author: Name — Year: YYYY — Genre: Genre — Topics: topic1, topic2"
- Full metadata: title, author, ISBN, summary, genres, topics, language, year

### Content Collection (Text Chunks)
Document fragments with parent book references:
- `document_id` links to parent book
- Chunk index and content for semantic search
- Used for content-based queries

## Duplicate Detection

4-level system with configurable thresholds:
1. **File hash** (SHA-256) - Immediate stop if identical
2. **ISBN check** - Database lookup for ISBN matches  
3. **Metadata** - Author + title similarity with fuzzy matching
4. **Content similarity** - n-gram analysis with Jaccard similarity

Use `force_ingest=true` parameter to override duplicate detection.

## Chat vs Search Modes

### Classic Search (`/chat`)
- Uses `orchestrator.py` 
- Returns structured JSON with results array
- No conversation history

### Interactive Chat (`/ws/chat_test`)
- Uses `orchestrator_with_chat.py`
- Maintains conversation history per session
- Returns chat responses with action chips
- Context-aware intent detection

## Key Files to Understand

- `settings.py` - All configuration constants and thresholds
- `retrievers.py` - Core search algorithms (OptimizedThresholdRetriever)
- `metadata.py` + `metadata_llm.py` - Document analysis and enrichment
- `utils_isbn.py` - ISBN extraction and normalization
- `file_hash_store.py` - Persistent hash storage for duplicates

## Debugging

Enable detailed logging by checking:
- LangSmith tracing for LLM calls
- Console logs with emoji prefixes (🔍, 🧠, 💬, etc.)
- Collection endpoints for data inspection
- Health check: `GET /health`

The system uses extensive logging throughout the pipeline to track intent detection, search steps, and result formatting.
- all code comments promtp should be in English
- не нужно пропт подгонять под запрос
- не нужно пропт подгонять под запрос нужен умнаый универсальный промпт