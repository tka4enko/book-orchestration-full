from typing import Optional, Dict, Any, List
import logging
import os
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY, MIN_SIMILARITY_THRESHOLD
from .retrievers import isbn_exact, hybrid_with_rerank
from .utils_isbn import extract_first_isbn

logger = logging.getLogger(__name__)

# Ensure LangSmith tracing is enabled
if os.getenv("LANGSMITH_TRACING") == "true":
    logger.info("🔍 LangSmith tracing enabled")
else:
    logger.warning("⚠️ LangSmith tracing not enabled")

class ChatState(BaseModel):
    session_id: str
    message: str
    intent: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    need_clarify: bool = False
    clarify_question: Optional[str] = None

llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0, api_key=OPENAI_API_KEY)

def _filter_results_by_relevance(query: str, docs: List, threshold: float = MIN_SIMILARITY_THRESHOLD) -> List:
    """Filter documents by semantic similarity to query"""
    if not docs:
        return []
    
    try:
        from langchain_openai import OpenAIEmbeddings
        from .settings import OPENAI_MODEL_EMBED
        import math
        
        embeddings_func = OpenAIEmbeddings(model=OPENAI_MODEL_EMBED, api_key=OPENAI_API_KEY)
        query_embedding = embeddings_func.embed_query(query)
        
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            return dot_product / (magnitude1 * magnitude2)
        
        filtered_docs = []
        for doc in docs:
            # Create text for embedding from document content/metadata
            if hasattr(doc, 'page_content'):
                text = doc.page_content
            elif hasattr(doc, 'metadata') and doc.metadata:
                # Create searchable text from metadata
                meta = doc.metadata
                text_parts = []
                if meta.get('title'): text_parts.append(meta['title'])
                if meta.get('author'): text_parts.append(meta['author'])
                if meta.get('summary'): text_parts.append(meta['summary'])
                if meta.get('primary_genre'): text_parts.append(meta['primary_genre'])
                text = ' '.join(text_parts)
            else:
                text = str(doc)
            
            if not text.strip():
                continue
                
            doc_embedding = embeddings_func.embed_query(text)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= threshold:
                filtered_docs.append(doc)
                
        return filtered_docs
    except Exception:
        # If filtering fails, return original docs
        return docs

INTENT_SYS = """You are an intent router for a book search assistant.
Decide one intent: isbn | author_title | author | title | genre | topic | free_text | clarify.
Return compact JSON: {"intent":"...", "filters":{...}, "clarify":""}.
Be strict: do not invent values."""

def node_detect_intent(state: ChatState) -> ChatState:
    logger.info("🧠 [orchestrator.py] node_detect_intent - Analyzing user intent...")
    logger.info(f"    Purpose: Determine what user wants (ISBN, author/title, genre, etc.)")
    logger.info(f"    Input: '{state.message}'")
    
    # Check for ISBN first
    isbn = extract_first_isbn(state.message)
    if isbn:
        logger.info(f"📖 [orchestrator.py] ISBN detected: {isbn}")
        state.intent = "isbn"
        state.filters = {"isbn13": isbn["isbn13"], "isbn10": isbn.get("isbn10")}
        logger.info(f"    Result: Intent='{state.intent}', Filters={state.filters}")
        return state
    
    # Use LLM for intent detection
    logger.info("🤖 [orchestrator.py] No ISBN found, using LLM for intent detection...")
    logger.info("    LLM Call: gpt-4o-mini to analyze query intent")
    
    out = llm.invoke([("system", INTENT_SYS), ("user", state.message)]).content
    
    logger.info(f"    LLM Response: {out}")
    
    import json, re
    try:
        j = json.loads(out)
    except Exception:
        logger.warning("⚠️  [orchestrator.py] LLM response not valid JSON, trying regex...")
        m = re.search(r"\{.*\}", out, re.S)
        j = json.loads(m.group(0)) if m else {"intent":"free_text","filters":{}}
    
    state.intent = j.get("intent") or "free_text"
    state.filters = j.get("filters") or {}
    clar = j.get("clarify") or ""
    
    if state.intent == "clarify" and clar.strip():
        state.need_clarify = True
        state.clarify_question = clar.strip()
        logger.info(f"❓ [orchestrator.py] Need clarification: {clar}")
    
    logger.info(f"✅ [orchestrator.py] Intent detected:")
    logger.info(f"    Intent: '{state.intent}'")
    logger.info(f"    Filters: {state.filters}")
    logger.info(f"    Need clarify: {state.need_clarify}")
    
    return state

def _docs_to_payload(docs):
    out = []
    for d in docs:
        m = d.metadata or {}
        out.append({
            "title": m.get("title"),
            "author": m.get("author"),
            "isbn13": m.get("isbn13"),
            "isbn10": m.get("isbn10"),
            "language": m.get("language"),
            "year": m.get("year"),
            "primary_genre": m.get("primary_genre"),
            "secondary_genres": m.get("secondary_genres"),
            "summary": m.get("summary"),
            "document_id": m.get("document_id"),
        })
    return out

def node_route_search(state: ChatState) -> ChatState:
    logger.info("🔍 [orchestrator.py] node_route_search - Searching for relevant documents...")
    logger.info(f"    Purpose: Find books based on detected intent '{state.intent}'")
    logger.info(f"    Filters: {state.filters}")
    
    if state.intent == "isbn":
        logger.info("📚 [orchestrator.py] ISBN search path:")
        logger.info("    Step 1: Trying exact ISBN match...")
        
        docs = isbn_exact(state.message)
        if docs:
            logger.info(f"    ✅ Found {len(docs)} exact ISBN matches")
        else:
            logger.info("    ⚠️  No exact ISBN match, trying semantic search...")
            retr = hybrid_with_rerank(intent=state.intent)
            docs = retr.invoke(state.message, intent=state.intent)
            logger.info(f"    📊 Semantic search returned {len(docs)} documents")
            
        state.results = _docs_to_payload(docs[:1])
        logger.info(f"    📋 Final ISBN results: {len(state.results)} books")
        return state

    logger.info("🔎 [orchestrator.py] Semantic search path:")
    logger.info("    Using OptimizedThresholdRetriever with similarity filtering")
    
    # Use hybrid search with threshold filtering for all other intents
    retr = hybrid_with_rerank(intent=state.intent) 
    docs = retr.invoke(state.message, intent=state.intent)
    
    logger.info(f"    📊 Retrieved {len(docs)} documents after similarity filtering")
    
    payload = _docs_to_payload(docs)

    logger.info("📋 [orchestrator.py] Converting documents to response format:")
    for i, doc in enumerate(payload):
        logger.info(f"    {i+1}. '{doc.get('title', 'Unknown')}' by {doc.get('author', 'Unknown')}")
        if doc.get('summary'):
            logger.info(f"       Summary: {doc['summary'][:80]}...")
        logger.info(f"       Genre: {doc.get('primary_genre', 'Unknown')}")

    if state.intent == "author_title":
        logger.info("👤📖 [orchestrator.py] Author-Title specific filtering:")
        
        f = state.filters or {}
        a = (f.get("author") or f.get("author_canonical") or "").lower()
        t = (f.get("title") or f.get("title_canonical") or "").lower()
        
        logger.info(f"    Looking for author: '{a}', title: '{t}'")
        
        # Filter exact matches from already filtered results
        exact = []
        for p in payload:
            author_match = (p.get("author") or "").lower().find(a) != -1 if a else True
            title_match = (p.get("title") or "").lower().find(t) != -1 if t else True
            
            if author_match and title_match:
                logger.info(f"    ✅ MATCH: '{p.get('title')}' by {p.get('author')}")
                exact.append(p)
            else:
                logger.info(f"    ❌ NO MATCH: '{p.get('title')}' by {p.get('author')}")
                logger.info(f"       Author match: {author_match}, Title match: {title_match}")
        
        logger.info(f"    📋 Found {len(exact)} exact author-title matches")
        
        state.results = exact[:1] if exact else []
        return state

    logger.info(f"📋 [orchestrator.py] General search results: {len(payload)} documents")
    state.results = payload[:10]
    return state

ANSWER_SYS = """You are a book assistant. Format responses as clear numbered lists.

For book recommendations, return a simple numbered list:
1. "Title" by Author - Brief description (1-2 sentences about why it matches the query)
2. "Title" by Author - Brief description

Guidelines:
- Use ONLY provided metadata, never invent information
- For genre/topic searches: focus on why each book matches the requested genre/topic  
- For author/title searches: list all matches found
- For ISBN searches: return exactly one book or "not found"
- Keep descriptions concise and relevant
- Always use Russian language for responses"""

def node_answer(state: ChatState) -> ChatState:
    logger.info("💬 [orchestrator.py] node_answer - Generating final response...")
    logger.info(f"    Purpose: Format search results into user-friendly answer")
    logger.info(f"    Input: {len(state.results)} results")
    
    if state.need_clarify and not state.results:
        logger.info("❓ [orchestrator.py] Returning clarification question")
        state.results = [{"clarify": state.clarify_question}]
        return state
    
    # Handle empty results
    if not state.results:
        logger.info("🚫 [orchestrator.py] No results found - generating 'not found' message")
        no_results_msg = f"К сожалению, не найдено книг, соответствующих вашему запросу '{state.message}'. Попробуйте изменить запрос или загрузить больше книг."
        state.results = [{"message": no_results_msg, "intent": state.intent}]
        logger.info("    📝 Generated: 'No results found' message")
        return state
    
    logger.info("🤖 [orchestrator.py] Using LLM to format response...")
    logger.info("    LLM Call: gpt-4o-mini to format search results")
    
    content = {"intent": state.intent, "filters": state.filters, "results": state.results}
    logger.info(f"    Sending to LLM: intent={state.intent}, {len(state.results)} results")
    
    # Log what we're sending to LLM for formatting
    logger.info("    📤 Data being sent to LLM for formatting:")
    for i, result in enumerate(state.results):
        if isinstance(result, dict):
            title = result.get('title', 'Unknown')
            author = result.get('author', 'Unknown')
            logger.info(f"      {i+1}. '{title}' by {author}")
    
    msg = llm.invoke([("system", ANSWER_SYS), ("user", f"{content}")]).content
    
    logger.info("    📝 LLM formatted the response successfully")
    logger.info(f"    📋 Final formatted response: {msg[:200]}..." if len(msg) > 200 else f"    📋 Final formatted response: {msg}")
    
    state.results = [{"message": msg, "intent": state.intent}]
    return state

builder = StateGraph(ChatState)
builder.add_node("detect_intent", node_detect_intent)
builder.add_node("route_search", node_route_search)
builder.add_node("answer", node_answer)
builder.set_entry_point("detect_intent")
builder.add_edge("detect_intent", "route_search")
builder.add_edge("route_search", "answer")
builder.add_edge("answer", END)
graph = builder.compile()
