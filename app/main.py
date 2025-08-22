import os, json, logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from .ingest import ingest_one, books_store, content_store
from .orchestrator import graph, ChatState
from .orchestrator_with_chat import graph_with_chat, ChatStateWithChat
from .orchestrator_chat_agent import chat_agent_graph, ChatAgentState
from .simple_orchestrator import process_simple_search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-load NLTK data at startup for better performance
logger.info("🔧 Pre-loading NLTK data...")
try:
    from .retrievers import _ensure_nltk
    _ensure_nltk()
    logger.info("✅ NLTK data loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ NLTK pre-loading failed: {e}")

# Clear BM25 cache on startup (Railway sleep/restart recovery)
logger.info("🔧 Clearing BM25 cache for Railway restart recovery...")
try:
    from .retrievers import clear_bm25_cache
    clear_bm25_cache()
    logger.info("✅ BM25 cache cleared - will rebuild on first search")
except Exception as e:
    logger.warning(f"⚠️ BM25 cache clearing failed: {e}")

# Debug LangSmith configuration
import os
logger.info(f"🔧 LangSmith config:")
logger.info(f"   LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
logger.info(f"   LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
logger.info(f"   LANGSMITH_API_KEY: {'*' * 10}...{os.getenv('LANGSMITH_API_KEY', '')[-4:]}")
logger.info(f"   LANGSMITH_ENDPOINT: {os.getenv('LANGSMITH_ENDPOINT')}")

# Ensure data directories exist (Railway Volume mount support)
from .settings import CHROMA_DIR, FILE_HASH_STORE_PATH
import os

# Create Chroma directory
os.makedirs(CHROMA_DIR, exist_ok=True)
logger.info(f"🗂️ Chroma directory ensured: {CHROMA_DIR}")

# Ensure parent directory exists for file hash store
hash_store_dir = os.path.dirname(FILE_HASH_STORE_PATH)
if hash_store_dir:
    os.makedirs(hash_store_dir, exist_ok=True)
    logger.info(f"📝 File hash store directory ensured: {hash_store_dir}")

logger.info(f"📝 File hash store path: {FILE_HASH_STORE_PATH}")

app = FastAPI(title="BookBot Final 4")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

class ChatIn(BaseModel):
    session_id: str
    message: str

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/chat_test", response_class=HTMLResponse)
def chat_test():
    with open(os.path.join(os.path.dirname(__file__), "static", "chat_test.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/simple-chat", response_class=HTMLResponse)
def simple_chat_page():
    """Web interface for testing simple search"""
    with open(os.path.join(os.path.dirname(__file__), "static", "simple_chat.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/chat_agent", response_class=HTMLResponse)
def chat_agent_page():
    """Web interface for testing chat-agent"""
    with open(os.path.join(os.path.dirname(__file__), "static", "chat_agent.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/chat")
def chat(body: ChatIn):
    import time
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info(f"🔥 CHAT REQUEST STARTED")
    logger.info(f"   Session: {body.session_id}")
    logger.info(f"   Query: '{body.message}'")
    logger.info("=" * 80)
    
    try:
        state = ChatState(session_id=body.session_id, message=body.message)
        logger.info("📋 [main.py] Creating ChatState and invoking LangGraph...")
        
        out = graph.invoke(state)
        
        logger.info("✅ [main.py] LangGraph execution completed")
        
        # Finalize debug session here - at the very end of request
        from .debug_reporter import finalize_debug
        try:
            # Handle both dict and object types
            if isinstance(out, dict) and 'results' in out:
                results = out['results']
            elif hasattr(out, 'results'):
                results = out.results
            else:
                results = []
            
            final_count = len([r for r in results if isinstance(r, dict) and r.get('title')])
            execution_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
            finalize_debug(final_count, execution_time_ms)
        except Exception as debug_error:
            logger.warning(f"Debug finalization failed: {debug_error}")
            import traceback
            traceback.print_exc()
        
        try:
            if hasattr(out, "model_dump"):
                payload = out.model_dump()
            elif isinstance(out, dict):
                payload = out
            else:
                try:
                    payload = dict(out)  # type: ignore[arg-type]
                except Exception:
                    payload = {"result": str(out)}
            
            # Add performance metrics to response
            if hasattr(out, 'performance_metrics') and out.performance_metrics:
                payload["performance_metrics"] = out.performance_metrics
                total_time = sum(out.performance_metrics.values())
                payload["total_execution_time_ms"] = total_time
                logger.info(f"📊 [main.py] Performance metrics: {out.performance_metrics}")
                logger.info(f"⏱️ [main.py] Total execution time: {total_time:.1f}ms")
            
            # Add detailed search metrics to response
            if hasattr(out, 'search_metrics') and out.search_metrics:
                payload["search_metrics"] = out.search_metrics
                logger.info(f"📊 [main.py] Search metrics: {out.search_metrics}")
                    
            logger.info(f"📤 [main.py] Response prepared with {len(payload.get('results', []))} results")
            logger.info("=" * 80)
            logger.info("🎉 CHAT REQUEST COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            return JSONResponse(payload)
            
        except Exception as e:
            logger.error(f"❌ [main.py] Serialization failed: {e}")
            return JSONResponse({"error": f"chat serialization failed: {e.__class__.__name__}: {e}"}, status_code=500)
            
    except Exception as e:
        logger.error(f"💥 [main.py] Chat request failed: {e}")
        logger.error("=" * 80)
        return JSONResponse({"error": f"Chat failed: {e}"}, status_code=500)

@app.post("/simple_chat")
async def simple_chat(body: ChatIn):
    """Simple search with LLM filter - alternative implementation without BM25"""
    import time
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info(f"🔍 SIMPLE CHAT REQUEST STARTED")
    logger.info(f"   Session: {body.session_id}")
    logger.info(f"   Query: '{body.message}'")
    logger.info("=" * 80)
    
    try:
        # Use new simple orchestrator
        result = await process_simple_search(body.session_id, body.message)
        
        # Add total execution time
        total_time = time.time() - start_time
        result["total_execution_time_seconds"] = total_time
        
        logger.info(f"📤 [simple_chat] Response prepared with {len(result.get('results', []))} results")
        logger.info(f"⏱️ [simple_chat] Total execution time: {total_time:.2f}s")
        logger.info("=" * 80)
        logger.info("🎉 SIMPLE CHAT REQUEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"💥 [simple_chat] Simple chat request failed: {e}")
        logger.error("=" * 80)
        return JSONResponse({
            "error": f"Simple chat failed: {e}",
            "response": "Sorry, an error occurred while processing the request.",
            "results": [],
            "intent": "error",
            "total_execution_time_seconds": time.time() - start_time
        }, status_code=500)

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...), meta: str = Form(None), prefer_llm: str = Form(None), max_chunks: str = Form(None), force_ingest: str = Form(None)):
    import asyncio
    
    logger.info(f"📁 Starting ingest process for file: {file.filename}")
    
    try:
        logger.info("💾 Saving uploaded file...")
        path = os.path.join("/tmp", file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        logger.info(f"✅ File saved to: {path}")
        
        logger.info("⚙️ Processing metadata parameters...")
        meta_json = json.loads(meta) if meta else None
        if prefer_llm is not None:
            v = str(prefer_llm).strip().lower()
            flag = v in ("1","true","yes","on")
            meta_json = (meta_json or {})
            meta_json["prefer_llm"] = flag
        if max_chunks is not None:
            try:
                mc = int(str(max_chunks).strip())
                meta_json = (meta_json or {})
                meta_json["max_chunks"] = mc
            except Exception:
                pass
        
        # Process force_ingest flag
        force_flag = False
        if force_ingest is not None:
            v = str(force_ingest).strip().lower()
            force_flag = v in ("1","true","yes","on")
        
        logger.info(f"✅ Metadata processed: {meta_json}")
        logger.info(f"🛡️ Force ingest: {force_flag}")
        
        logger.info("🤖 Starting document processing with duplicate detection...")
        res = await asyncio.to_thread(ingest_one, path, meta_json, force_flag)
        
        # Handle different response types
        if res.get("status") == "duplicate_detected":
            logger.warning(f"🚨 Ingestion blocked: {res.get('message')}")
            return JSONResponse(res, status_code=409)  # Conflict status for duplicates
        else:
            logger.info(f"🎉 Ingest completed successfully! Book ID: {res.get('book_id')}, Chunks: {res.get('chunks')}")
            return JSONResponse(res)
        
    except Exception as e:
        logger.error(f"❌ Ingest failed: {e.__class__.__name__}: {e}")
        return JSONResponse({"error": f"Ingest failed: {e}"}, status_code=500)

def _get_store(name: str):
    if name == "books": return books_store()
    if name == "content": return content_store()
    return None

@app.get("/vector-store/collection/{name}")
def collection_meta(name: str, limit: int = Query(3, ge=1, le=200)):
    store = _get_store(name)
    if store is None:
        return JSONResponse({"error":"unknown collection"}, status_code=404)
    coll = store._collection
    total = coll.count()
    
    # For books collection, only show master chunks
    if name == "books":
        sample = coll.get(
            where={"is_master_chunk": True}, 
            limit=limit, 
            include=["documents","metadatas"]
        )
        # Get count of only master chunks for books
        try:
            master_count = coll.count(where={"is_master_chunk": True})
        except:
            master_count = total  # Fallback to total count if filtering fails
    else:
        sample = coll.get(limit=limit, include=["documents","metadatas"])
        master_count = total
    
    docs = sample.get("documents", []) or []
    metas = sample.get("metadatas", []) or []
    items = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        # For books collection, show full content without truncation to see all enriched metadata
        # For other collections, keep 120 char limit
        if name == "books":
            content_preview = d or ""  # Full content for books
        else:
            content_preview = (d or "")[:120]  # Truncated for content/other collections
            
        items.append({
            "id": m.get("document_id") if isinstance(m, dict) else None,
            "content_preview": content_preview,
            "content_length": len(d or ""),
            "metadata": m,
        })
    
    # Use master_count for books, total for other collections
    actual_total = master_count if name == "books" else total
    
    return JSONResponse({
        "collection_name": name,
        "stats": {"total_documents": actual_total, "returned_count": len(docs), "limit": limit},
        "documents": items,
        "available_endpoints": {
            "view_chunks": f"/vector-store/collection/{name}/chunks",
            "search": f"/vector-store/collection/{name}/search",
            "specific_chunk": f"/vector-store/chunk/{name}" + "/{chunk_id}"
        }
    })

@app.get("/vector-store/collection/{name}/chunks")
def collection_chunks(name: str, document_id: Optional[str] = Query(default=None), offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    store = _get_store(name)
    if store is None:
        return JSONResponse({"error":"unknown collection"}, status_code=404)
    coll = store._collection
    # Chroma doesn't accept empty where clause, use None instead
    where = {"document_id": document_id} if document_id else None
    
    if where:
        data = coll.get(where=where, include=["documents","metadatas"], limit=limit, offset=offset)
    else:
        data = coll.get(include=["documents","metadatas"], limit=limit, offset=offset)
    ids = data.get("ids", []) or []
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    items = []
    for _id, doc, meta in zip(ids, docs, metas):
        # For books collection, show full content without truncation to see all enriched metadata
        # For other collections, keep 200 char limit
        if name == "books":
            content_preview = doc or ""  # Full content for books
        else:
            content_preview = (doc or "")[:200]  # Truncated for content/other collections
            
        items.append({
            "id": _id,
            "content_preview": content_preview,
            "content_length": len(doc or ""),
            "metadata": meta
        })
    return JSONResponse({"collection_name": name, "offset": offset, "limit": limit, "count": len(items), "items": items})

@app.get("/vector-store/chunk/{name}/{chunk_id}")
def get_chunk(name: str, chunk_id: str):
    store = _get_store(name)
    if store is None:
        return JSONResponse({"error":"unknown collection"}, status_code=404)
    coll = store._collection
    data = coll.get(ids=[chunk_id], include=["documents","metadatas"])
    if not data.get("ids"):
        return JSONResponse({"error":"not found"}, status_code=404)
    return JSONResponse({"id": data["ids"][0], "content": (data["documents"][0] or ""), "metadata": data["metadatas"][0]})

@app.get("/vector-store/collection/{name}/search")
def search_collection(name: str, q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50)):
    store = _get_store(name)
    if store is None:
        return JSONResponse({"error":"unknown collection"}, status_code=404)
    docs = store.similarity_search(q, k=k)
    results = [{"content_preview": (d.page_content or "")[:200], "content_length": len(d.page_content or ""), "metadata": d.metadata} for d in docs]
    return JSONResponse({"collection_name": name, "k": k, "query": q, "results": results})

# Chat history storage for sessions
chat_histories = {}

@app.websocket("/ws/chat_agent")
async def chat_agent_websocket(websocket: WebSocket):
    """New WebSocket endpoint for chat-agent"""
    await websocket.accept()
    
    try:
        while True:
            raw_message = await websocket.receive_json()
            session_id = raw_message.get("session_id", "default")
            user_message = raw_message.get("message", "")
            
            logger.info(f"🤖 [ChatAgent] Message from {session_id}: '{user_message}'")
            
            # Get history for session
            chat_history = chat_histories.get(session_id, [])
            
            # Use new chat-agent orchestrator
            state = ChatAgentState(
                session_id=session_id, 
                message=user_message, 
                chat_history=chat_history
            )
            # Use thread_id to preserve state between messages
            config = {"configurable": {"thread_id": session_id}}
            result = await chat_agent_graph.ainvoke(state, config=config)
            
            # Format response
            if hasattr(result, "model_dump"):
                response_data = result.model_dump()
            else:
                response_data = result
            
            # Update chat history
            if hasattr(result, "chat_history"):
                chat_histories[session_id] = result.chat_history
            elif "chat_history" in response_data:
                chat_histories[session_id] = response_data["chat_history"]
            
            # Create response for frontend
            results = response_data.get("results", [{}])
            first_result = results[0] if results else {}
            
            # Determine operation mode based on result type
            result_intent = first_result.get("intent", "chat")
            mode_type = "unknown"
            if first_result.get("chat_mode"):
                mode_type = "chat"
            elif first_result.get("search_mode"):
                mode_type = "search"
            elif first_result.get("recommendation_mode"):
                mode_type = "recommend"
            elif first_result.get("clarify_mode"):
                mode_type = "clarify"
            
            # Get performance metrics
            performance_metrics = response_data.get("performance_metrics", {})
            total_time = performance_metrics.get("total_time", 0)
            
            response = {
                "reply": first_result.get("message", ""),
                "cards": [],  # For future expansion
                "chips": first_result.get("chips", []),
                "intent": result_intent,
                "debug": {
                    "intent": result_intent,
                    "mode": f"chat_agent_{mode_type}",
                    "route_taken": mode_type,
                    "performance_metrics": performance_metrics,
                    "processing_time": f"{total_time:.1f}ms" if total_time > 0 else "N/A"
                }
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info("🤖 [ChatAgent] Client disconnected")

@app.websocket("/ws/chat_test")
async def chat_test_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            raw_message = await websocket.receive_json()
            session_id = raw_message.get("session_id", "default")
            user_message = raw_message.get("message", "")
            
            logger.info(f"🧪 [ChatTest] Message from {session_id}: '{user_message}'")
            
            # Get history for session
            chat_history = chat_histories.get(session_id, [])
            
            # Use new orchestrator with chat
            state = ChatStateWithChat(session_id=session_id, message=user_message, chat_history=chat_history)
            result = graph_with_chat.invoke(state)
            
            # Format response
            if hasattr(result, "model_dump"):
                response_data = result.model_dump()
            else:
                response_data = result
            
            # Update chat history
            if hasattr(result, "chat_history"):
                chat_histories[session_id] = result.chat_history
            elif "chat_history" in response_data:
                chat_histories[session_id] = response_data["chat_history"]
            
            # Create response for frontend
            response = {
                "reply": response_data.get("results", [{}])[0].get("message", ""),
                "cards": [],  # Will add later
                "chips": response_data.get("results", [{}])[0].get("chips", []),
                "intent": response_data.get("intent", "chat"),
                "debug": {
                    "intent": response_data.get("intent"),
                    "mode": "chat_test",
                    "processing_time": "N/A"
                }
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info("🧪 [ChatTest] Client disconnected")
