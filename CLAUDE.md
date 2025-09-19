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
# Upload single document
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@book.pdf" \
  -F "meta={\"title\":\"Title\",\"author\":\"Author\"}" \
  -F "prefer_llm=true"

# Upload multiple documents (batch processing with duplicate detection)
curl -X POST "http://localhost:8000/ingest-batch" \
  -F "files=@book1.pdf" \
  -F "files=@book2.pdf" \
  -F "files=@book3.pdf" \
  -F "prefer_llm=true"

# Force ingest (skip duplicate detection)
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@book1.pdf" \
  -F "files=@book2.pdf" \
  -F "force_ingest=true"

# View collections
curl "http://localhost:8000/vector-store/collection/books?limit=5"
curl "http://localhost:8000/vector-store/collection/content?limit=5"

# Get collection statistics
curl "http://localhost:8000/vector-store/stats"
```

### Data Quality & Cleanup
```bash
# Find empty/invalid records
curl "http://localhost:8000/vector-store/collection/books/empty-records?limit=100"

# Clean up empty records from both collections
curl -X DELETE "http://localhost:8000/vector-store/cleanup-empty-records"

# Delete specific record by ID
curl -X DELETE "http://localhost:8000/vector-store/collection/books/delete/{record_id}"

# Bulk delete multiple records
curl -X DELETE "http://localhost:8000/vector-store/collection/books/bulk-delete" \
  -H "Content-Type: application/json" \
  -d '{"ids": ["id1", "id2", "id3"]}'

# DANGER: Clear all records from both collections
curl -X DELETE "http://localhost:8000/vector-store/clear-all-collections"

# DANGER: Clear specific collection
curl -X DELETE "http://localhost:8000/vector-store/collection/books/clear"

# EXTREME DANGER: Total database reset (clears everything including file hashes)
curl -X DELETE "http://localhost:8000/vector-store/total-reset"
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

## Document Ingestion & Quality Control

### Duplicate Detection
4-level system with configurable thresholds:
1. **File hash** (SHA-256) - Immediate stop if identical
2. **ISBN check** - Database lookup for ISBN matches
3. **Metadata** - Author + title similarity with fuzzy matching
4. **Content similarity** - n-gram analysis with Jaccard similarity

Use `force_ingest=true` parameter to override duplicate detection.

### Quality Validation
The system validates record quality before saving to prevent empty/invalid records:
- **Title OR Author required** - At least one meaningful identifier
- **Content validation** - Minimum 10 characters of actual content
- **Metadata filtering** - Rejects "unknown", "untitled", empty values
- **Automatic rejection** - Poor quality records are blocked with detailed error messages

### Bulk Upload Support
Single `/ingest` endpoint handles both scenarios:
- **Single file**: Use `file` parameter
- **Multiple files**: Use `files` parameter (multiple values)
- **Robust parsing**: Handles malformed multipart requests with raw body extraction
- **Detailed logging**: Request tracking with unique IDs and comprehensive error reporting

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

## API Reference

### Core Endpoints
- **`GET /health`** - Health check
- **`POST /chat`** - Classic search with structured JSON response
- **`POST /simple_chat`** - Alternative search implementation
- **`GET /chat_test`** - Chat interface HTML page
- **`WS /ws/chat_test`** - WebSocket chat interface
- **`WS /ws/chat_agent`** - Smart chat agent WebSocket

### Document Management
- **`POST /ingest`** - Universal endpoint for single/batch document upload with duplicate detection
  - Single file: Use `file` parameter
  - Batch upload: Use `files` parameter (multiple files)
  - Supports: `meta`, `prefer_llm`, `max_chunks`, `force_ingest` parameters
  - Returns comprehensive batch summary with success/duplicate/error counts
- **`GET /vector-store/collection/{name}`** - View collection contents
- **`GET /vector-store/stats`** - Get statistics for all collections

### Data Quality & Cleanup
- **`GET /vector-store/collection/{name}/empty-records`** - Find invalid records
- **`DELETE /vector-store/cleanup-empty-records`** - Clean both collections
- **`DELETE /vector-store/collection/{name}/delete/{id}`** - Delete specific record
- **`DELETE /vector-store/collection/{name}/bulk-delete`** - Delete multiple records
- **`DELETE /vector-store/collection/{name}/clear`** - Clear specific collection (DANGER)
- **`DELETE /vector-store/clear-all-collections`** - Clear all collections (DANGER)
- **`DELETE /vector-store/total-reset`** - Complete database reset including file hashes (EXTREME DANGER)

## Key Files to Understand

- `settings.py` - All configuration constants and thresholds
- `retrievers.py` - Core search algorithms (OptimizedThresholdRetriever)
- `metadata.py` + `metadata_llm.py` - Document analysis and enrichment
- `utils_isbn.py` - ISBN extraction and normalization
- `file_hash_store.py` - Persistent hash storage for duplicates
- `ingest.py` - Document ingestion with quality validation and multipart handling
- `main.py` - FastAPI server with comprehensive error handling and logging

## Debugging & Monitoring

### Logging System
- **Request tracking**: Unique request IDs for all operations
- **Emoji prefixes**: Visual categorization (🔍 search, 🧠 LLM, 💬 chat, 🗑️ delete, etc.)
- **Comprehensive error handling**: Full stack traces and detailed error context
- **Quality validation logs**: Shows exactly why records are accepted/rejected

### Debug Tools
- **LangSmith tracing** - LLM call monitoring
- **Collection endpoints** - Data inspection and statistics
- **Health checks** - System status verification
- **Request ID tracking** - End-to-end request monitoring

### Performance Monitoring
- **Execution timers** - Track processing times
- **Statistics endpoints** - Monitor collection health and data quality
- **Error categorization** - Structured error responses with suggestions
## Development Guidelines

### Code Standards
- **All code comments and prompts should be in English**
- **Universal intelligence approach**: Use smart, universal prompts instead of hardcoded rules or specific examples
- **No prompt customization**: Don't adapt prompts to specific requests - create intelligent, universal prompts that work across all scenarios
- **Error handling**: Comprehensive error handling with detailed logging and structured responses
- **Request tracking**: All operations should include unique request IDs for debugging

### Prompt Engineering Principles
- **ПРАВИЛА (Universal Intelligence)**: Always use universal, intelligent approaches rather than rule-based systems
- **Smart LLM integration**: Let LLMs use their intelligence to understand context and intent
- **Avoid hardcoded examples**: Create flexible prompts that adapt to any scenario
- **Context awareness**: Design prompts that can handle varying contexts without modification

### Quality Assurance
- **Data validation**: Always validate data quality before saving to collections
- **Comprehensive logging**: Track all operations with detailed context and error information
- **Graceful degradation**: Handle errors elegantly with helpful error messages and recovery suggestions
- **Performance monitoring**: Track execution times and system health metrics