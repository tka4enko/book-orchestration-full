# 📚 BookBot - Intelligent Book Search System with Chat Interface

**BookBot** is an advanced AI-powered book search system that combines vector search, intelligent intent detection, and a modern chat interface with WebSocket support for real-time communication.

## 🎯 Key Features

### 🔍 Intelligent Search

- **Vector Search** - semantic search using OpenAI embeddings
- **Smart Intent Detection** - automatic recognition of query types (ISBN, author, genre, topic)
- **LLM Filtering** - AI-powered result filtering and analysis
- **Real-time Chat** - interactive conversation with context awareness

### 🧠 AI Capabilities

- **LangGraph Orchestration** - structured request processing pipelines
- **Contextual Chat** - conversation history and contextual responses
- **LLM Metadata Analysis** - automatic book information extraction
- **Language Detection** - multilingual support

### 💬 Chat Interface

- **Interactive Mode** - conversational book discovery
- **WebSocket Support** - real-time communication
- **Smart Chips** - suggested actions and conversation continuations
- **Contextual Guidance** - directing users to book search

## 🏗️ System Architecture

```
bookbot_full/
├── app/
│   ├── main.py                      # FastAPI server + WebSocket
│   ├── simple_orchestrator.py       # Simple search pipeline
│   ├── orchestrator_chat_agent.py   # Chat agent with thread persistence
│   ├── simple_retriever.py          # Vector search algorithms
│   ├── simple_llm_filter.py         # LLM result filtering
│   ├── ingest.py                    # Document loading and processing
│   ├── loaders.py                   # File loaders (PDF, DOCX, TXT)
│   ├── metadata.py                  # Metadata processing
│   ├── metadata_llm.py              # LLM content analysis
│   ├── duplicate_detection.py       # Multi-level duplicate detection
│   ├── file_hash_store.py           # File hash storage
│   ├── utils_isbn.py                # ISBN normalization and validation
│   ├── settings.py                  # System configuration
│   └── static/
│       ├── index.html               # Main web interface
│       ├── chat_agent.html          # Chat interface
│       └── simple_chat.html         # Simple chat test interface
└── requirements.txt                 # Python dependencies
```

## 💻 Technology Stack

### Backend

- **FastAPI** - high-performance web framework
- **LangChain** - LLM application framework
- **LangGraph** - graph orchestration for AI pipelines
- **ChromaDB** - vector database with persistence
- **WebSocket** - real-time chat support

### AI Models

- **OpenAI GPT-4o-mini** - language model for chat and analysis
- **text-embedding-3-small** - vector embeddings
- **LangSmith** - LLM monitoring and tracing

### Data Processing

- **PyPDF2** - PDF document processing
- **python-docx** - Word document processing
- **isbnlib** - ISBN number handling
- **langdetect** - language detection

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd bookbot_full
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_CHAT=gpt-4o-mini
OPENAI_MODEL_EMBED=text-embedding-3-small

# Database
CHROMA_DIR=.chroma

# Server
HOST=127.0.0.1
PORT=8000

# LangSmith (optional)
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bookbot
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Search settings (optional)
MAX_CHUNKS_PER_BOOK=2
CHUNKS_PER_BOOK_IN_CONTENT=2
CONTENT_SEARCH_EXPAND_K=20
```

### 3. Launch

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Access

- **Web Interface**: http://127.0.0.1:8000
- **Chat Interface**: http://127.0.0.1:8000/chat_agent
- **Simple Chat Test**: http://127.0.0.1:8000/simple_chat
- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## 📖 Usage

### Web Interfaces

#### Main Interface (`/`)

- Standard search with formatted results
- Support for all query types
- JSON responses with detailed information

#### Chat Interface (`/chat_agent`)

- Conversational mode with history
- WebSocket real-time connection
- Contextual hints and chips
- Smart switching between chat and search

### API Endpoints

#### Book Search

```bash
POST /chat
Content-Type: application/json

{
  "session_id": "unique_session_id",
  "message": "find books by Orwell"
}
```

Response:

```json
{
  "session_id": "unique_session_id",
  "intent": "author",
  "results": [
    {
      "message": "1. «1984» by George Orwell - Dystopian novel about totalitarian control",
    "intent": "author"
    }
  ]
}
```

#### Chat via WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/chat_agent");
ws.send(
  JSON.stringify({
    session_id: "test_session",
    message: "hello, recommend something to read",
  })
);
```

Response:

```json
{
  "reply": "Hello! What interests you?",
  "cards": [],
  "chips": [
    { "text": "Detective", "action": "search" },
    { "text": "Fantasy", "action": "search" }
  ],
  "intent": "chat"
}
```

#### Book Upload

```bash
POST /ingest
Content-Type: multipart/form-data

file: [PDF/DOCX/TXT file]
meta: {"title": "Title", "author": "Author", "prefer_llm": true}
max_chunks: 50
force_ingest: false
```

With duplicate detection:

```json
{
  "status": "duplicate_detected",
  "message": "Document rejected: ISBN 9781234567890 already exists",
  "duplicate_level": "isbn",
  "duplicate_confidence": 1.0,
  "existing_document_id": "existing-doc-123"
}
```

#### Collection View

```bash
# Books (master records)
GET /vector-store/collection/books?limit=10

# Content (chunks)
GET /vector-store/collection/content?limit=10

# Specific book
GET /vector-store/collection/books/chunks?document_id=book-id-123

# Search in collection
GET /vector-store/collection/books/search?q=Orwell&k=5

# Individual chunk
GET /vector-store/chunk/books/book-id-123
```

## 🔍 Search Algorithms

### Simple Vector Search (SimpleVectorRetriever)

Two-stage pipeline with optimization:

```
1. Vector Search in Books Collection
   ├─ Semantic search in master records
   ├─ Score-based filtering
   └─ Top results selection

2. Vector Search in Content Collection
   ├─ Semantic search in text chunks
   ├─ Grouping by books
   └─ Best chunk selection per book

3. Smart Merging
   ├─ Combine master + best chunk
   ├─ Remove duplicates
   └─ Final ranking
```

### Intent Detection

LLM analyzes queries and context to determine type:

```python
# Exact search
"isbn": "978-1234567890"        # ISBN number
"author": "Stephen King"         # Author
"title": "1984"                 # Title
"author_title": "King Shining"   # Author + title

# Semantic search
"genre": "detective"             # Genre
"topic": "psychology"            # Topic
"free_text": "interesting book"  # Free text

# Chat
"greeting": "hello"              # Greeting
"clarify": "recommend genre"     # Need advice
"chat": "how are you?"           # Conversation
```

### Contextual Analysis in Chat

System analyzes conversation history:

```python
# Agreement handling
History: "assistant: How about fantasy or detective?"
User: "yes"
→ Offers choice between options

History: "assistant: Try Stephen King"
User: "okay"
→ Searches for King's books

# Book direction
After 4+ messages without book topics
→ Gently suggests books
→ Switches to recommendations on agreement
```

## 📊 Data Structure

### "books" Collection (Master Records)

```python
{
  "document_id": "unique-uuid",
  "title": "Book Title",
  "author": "Author Name",
  "isbn13": "9781234567890",
  "isbn10": "1234567890", 
  "summary": "Brief description...",
  "primary_genre": "Science Fiction",
  "secondary_genres": ["Dystopian", "Classic"],
  "main_topics": ["Totalitarianism", "Control"],
  "mentioned_topics": ["Politics", "Society"],
  "language": "en",
  "year": 1949,
  "is_master_chunk": true,
  "title_canonical": "book title",
  "author_canonical": "author name",
  "file_type": "pdf"
}
```

### "content" Collection (Chunks)

```python
{
  "document_id": "parent-book-uuid",
  "title": "Book Title",
  "author": "Author Name",
  "language": "en",
  "idx": 0  // chunk index
}
```

### Master Text Format

Enriched text for semantic search:

```
Book Title — Full description — Author: Author Name — Year: 1949 — Genre: Science Fiction, Dystopian — Topics: Totalitarianism, Control, Politics
```

## ⚙️ Configuration

### Search Parameters (settings.py)

```python
# Chunk sizes
CHUNK_SIZE = 1200          # Text chunk size
CHUNK_OVERLAP = 120        # Overlap between chunks
MAX_CHUNKS = 80           # Maximum chunks per book

# Algorithm parameters
VEC_BOOKS_K = 8          # Results from books
VEC_CONTENT_K = 6        # Results from content
MAX_CHUNKS_PER_BOOK = 2             # Max chunks per book
CHUNKS_PER_BOOK_IN_CONTENT = 2      # Chunks per book in content
CONTENT_SEARCH_EXPAND_K = 20        # Expanded search for selection
```

### Environment Variables

| Variable             | Description        | Default                  |
| -------------------- | ------------------ | ------------------------ |
| `OPENAI_API_KEY`     | OpenAI API key     | **required**             |
| `OPENAI_MODEL_CHAT`  | Chat model         | `gpt-4o-mini`            |
| `OPENAI_MODEL_EMBED` | Embedding model    | `text-embedding-3-small` |
| `CHROMA_DIR`         | ChromaDB folder    | `.chroma`                |
| `HOST`               | Server host        | `127.0.0.1`              |
| `PORT`               | Server port        | `8000`                   |
| `LANGSMITH_TRACING`  | LangSmith tracing  | `false`                  |
| `LANGSMITH_PROJECT`  | LangSmith project  | -                        |
| `LANGSMITH_API_KEY`  | LangSmith key      | -                        |
| `LANGSMITH_ENDPOINT` | LangSmith endpoint | -                        |

## 🎨 User Interface

### Main Components

#### 1. Main Search (`index.html`)

- Search form with autofocus
- Results with detailed information
- File upload form with settings
- Vector database collection view

#### 2. Chat Interface (`chat_agent.html`)

- WebSocket connection
- Message history with scrolling
- Smart chip buttons for actions
- Connection status indication
- Debug information

### Interactive Elements

```javascript
// Chips for quick actions
chips: [
  { text: "Detective", action: "search" },
  { text: "Fantasy", action: "search" },
  { text: "Let's talk about something else", action: "chat" },
];

// Action types
action: "search"; // Book search
action: "chat"; // Continue conversation
```

## 🔧 API Reference

### Main Endpoints

#### `GET /` - Main Page

Returns HTML interface for search

#### `GET /chat_agent` - Chat Interface

Returns HTML chat interface with WebSocket

#### `GET /simple_chat` - Simple Chat Test

Returns simple HTML chat test interface

#### `GET /health` - Health Check

```json
{ "ok": true }
```

#### `POST /chat` - Book Search

Main endpoint for search via REST API

Request:

```json
{
  "session_id": "string",
  "message": "string"  
}
```

Response:

```json
{
  "session_id": "string",
  "message": "string", 
  "intent": "author|title|isbn|genre|topic|free_text|clarify",
  "filters": {},
  "results": [
    {
      "title": "string",
      "author": "string", 
      "isbn13": "string",
      "summary": "string",
      "message": "string", // Formatted response
      "intent": "string"
    }
  ],
  "need_clarify": false,
  "clarify_question": "string"
}
```

#### `POST /ingest` - Document Upload

Book upload with automatic processing

Parameters (multipart/form-data):

- `file`: PDF/DOCX/TXT file
- `meta`: JSON with metadata (optional)
- `prefer_llm`: "true"/"false" - use LLM for analysis
- `max_chunks`: number - maximum chunks
- `force_ingest`: "true"/"false" - forced upload

Success response:

```json
{
  "status": "success",
  "message": "Document 'Title' ingested successfully", 
  "book_id": "book:hash16",
  "document_id": "uuid",
  "chunks": 25,
  "metadata": {
    "title": "string",
    "author": "string",
    "language": "en",
    "primary_genre": "string",
    "isbn13": "string"
  },
  "duplicate_checks": []
}
```

Duplicate detection:

```json
{
  "status": "duplicate_detected",
  "message": "Document rejected: reason",
  "duplicate_level": "file_hash|isbn|metadata|content_similarity",
  "duplicate_confidence": 0.95,
  "existing_document_id": "uuid",
  "duplicate_checks": [
    {
      "level": "string",
      "is_duplicate": true,
      "reason": "string", 
      "confidence": 0.95
    }
  ]
}
```

#### `WebSocket /ws/chat_agent` - Real-time Chat

Client message:

```json
{
  "session_id": "string",
  "message": "string"
}
```

Server response:

```json
{
  "reply": "string", // Bot response
  "cards": [], // Cards (future feature)
  "chips": [
    // Suggested actions
    {
      "text": "string", 
      "action": "search|chat"
    }
  ],
  "intent": "search|chat", // Processing type
  "debug": {
    // Debug information
    "intent": "string",
    "mode": "chat_agent",
    "processing_time": "string"
  }
}
```

### Vector Database Collections

#### `GET /vector-store/collection/{name}` - Collection Overview

Parameters:

- `name`: "books" | "content"
- `limit`: number of results (1-200, default 3)

#### `GET /vector-store/collection/{name}/chunks` - Document Chunks

Parameters:

- `name`: collection name
- `document_id`: document ID (optional)
- `offset`: offset (default 0)
- `limit`: limit (1-200, default 50)

#### `GET /vector-store/collection/{name}/search` - Collection Search

Parameters:

- `name`: collection name
- `q`: search query
- `k`: number of results (1-50, default 5)

#### `GET /vector-store/chunk/{name}/{chunk_id}` - Specific Chunk

Returns complete chunk information

## 🚀 Performance and Optimizations

### Algorithmic Optimizations

```python
# 1. Async vector search
# Parallel queries to books and content collections
books_task = asyncio.create_task(search_books())
content_task = asyncio.create_task(search_content())
results = await asyncio.gather(books_task, content_task)

# 2. Content diversification
# Limit chunks per book for variety
for book_id, chunks in grouped_chunks.items():
    best_chunks = sorted(chunks, key=score)[:CHUNKS_PER_BOOK_IN_CONTENT]

# 3. Smart merging
# Combine master record with best chunk
merged_doc.content = master.content + "\n\n" + best_chunk.content
```

### Caching and Reuse

- **Embedding Reuse** - vector search uses pre-computed embeddings
- **Persistent Database** - ChromaDB saves data between restarts
- **Hash Storage** - fast file duplicate checking

### Scaling

- **Async Loading** - `asyncio.to_thread` for blocking operations
- **WebSocket Pool** - support for multiple chat sessions
- **Modular Architecture** - easy component extension
- **Configurable Limits** - tune for data volume

## 🔒 Security

### Data Validation

- **Pydantic Models** - strict API typing
- **File Size Limits** - upload restrictions
- **File Formats** - only allowed types supported
- **SQL Injection** - parameterized ChromaDB queries

### Anomaly Detection

- **Duplication** - prevent spam uploads
- **Empty Files** - content verification
- **Invalid ISBN** - validation via isbnlib
- **Suspicious Requests** - logging and monitoring

### Logging and Audit

```python
# Detailed logging of all operations
logger.info("🔍 [retrievers.py] Starting vector search...")
logger.warning("🚨 DUPLICATE DETECTED: ISBN already exists")
logger.error("❌ [main.py] Chat request failed")

# Tracing via LangSmith
LANGSMITH_TRACING=true  # Track LLM calls
```

## 🧪 Development and Debugging

### Log Structure

```
🔥 CHAT REQUEST STARTED          # Request processing start
🧠 node_detect_intent            # Intent detection
🔍 node_route_search             # Document search
💬 node_answer                   # Response formation
🎉 CHAT REQUEST COMPLETED        # Completion
```

### Detail Levels

```python
# Search pipeline logging
🎯 SimpleVectorRetriever         # Main algorithm
  🔤 Step 1: Books search        # Books collection search
  📊 Step 2: Content search      # Content collection search
  🧹 Smart merging              # Result merging

# Duplicate detection
🔍 Level 1: File hash check      # Hash verification
🔍 Level 2: ISBN check           # ISBN verification
🔍 Level 3: Metadata check       # Metadata verification
🔍 Level 4: Content similarity   # Content verification
```

### Debug Endpoints

```bash
# Collection status
GET /vector-store/collection/books
GET /vector-store/collection/content

# Collection search
GET /vector-store/collection/books/search?q=test&k=5

# Individual documents
GET /vector-store/chunk/books/book-id-123
```

### Debug Configuration

```env
# Enable detailed logging
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bookbot_debug
```

## 🤝 System Extension

### Adding New File Formats

```python
# In loaders.py
def load_text_from_file(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    
    # Add new format
    if ext == ".epub":
        return load_epub(path), "epub"
    
    # Existing formats
    if ext in (".txt", ".md", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), ext.lstrip(".")
```

### New Search Algorithms

```python
# In simple_retriever.py
class CustomRetriever:
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        # Custom search logic
        return documents
```

### Additional Duplicate Detectors

```python
# In duplicate_detection.py
def level5_semantic_similarity(text: str) -> DuplicateDetectionResult:
    """Level 5: Semantic similarity via embeddings"""
    # Vector content comparison
    return DuplicateDetectionResult(...)

# Update main function
def detect_duplicates(...):
    # Existing levels 1-4
    result5 = level5_semantic_similarity(full_text) 
    results.append(result5)
```

### Custom Intents

```python
# In simple_orchestrator.py
INTENT_SYS = """
Add new intents:
- series: search for book series
- publisher: search by publisher
- year_range: search by years
"""

# Add processing in search_step
if state.intent == "series":
    # Series search logic
```

## 📈 Monitoring and Metrics

### LangSmith Integration

```env
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bookbot_production
LANGSMITH_API_KEY=your_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Tracked metrics:

- LLM call execution time
- Tokens used for analysis
- Intent detection success rate
- Search result quality

### Performance Logs

```python
# Step execution time
logger.info("🔍 Vector search completed in 0.32s")
logger.info("🧹 Merging completed in 0.08s")
logger.info("🎯 Total search time: 0.40s")

# Result statistics
logger.info("📊 Books found: 12 → filtered: 3")
logger.info("📚 Content found: 24 → kept: 7")
logger.info("🎯 FINAL RESULTS: 8 documents after merging")
```

### Key KPIs

- **Search Accuracy** - result relevance to query
- **Response Time** - request processing speed
- **Collection Coverage** - percentage of findable books
- **Duplication Rate** - duplicate detection effectiveness
- **Resource Usage** - OpenAI API consumption

## ❓ Troubleshooting

### Common Issues

#### 1. No Search Results

```bash
# Check collections
GET /vector-store/collection/books

# Lower thresholds
SIMILARITY_THRESHOLDS='{"free_text": 0.2}'
```

#### 2. Slow Performance

```python
# Check collection size
collection.count()  # If > 10000 - optimize

# Reduce search parameters
VEC_BOOKS_K = 5
CONTENT_SEARCH_EXPAND_K = 10
```

#### 3. Upload Issues

```python
# Forced upload
force_ingest = True

# Debug duplicate detection
logger.info("Duplicate checks: %s", duplicate_results)
```

#### 4. WebSocket Disconnections

```javascript
// Reconnection
ws.onclose = function () {
    setTimeout(connectWebSocket, 1000);
};
```

### Diagnostic Commands

```bash
# Health check
curl http://localhost:8000/health

# Test search
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"test"}'

# Collection status
curl "http://localhost:8000/vector-store/collection/books?limit=1"
```

## 📄 License

MIT License - free use and modification.

## 🆘 Support

1. **Check Configuration** - `.env` file and API keys
2. **Review Logs** - detailed logging of all operations
3. **Test Components** - use diagnostic endpoints
4. **Create Issue** - for bug reports and suggestions

---

**BookBot** - modern AI system for intelligent book search with advanced chat capabilities! 📚✨🤖
