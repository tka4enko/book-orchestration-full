import os, json, logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .ingest import ingest_one, books_store, content_store
from .orchestrator import graph, ChatState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug LangSmith configuration
import os
logger.info(f"🔧 LangSmith config:")
logger.info(f"   LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
logger.info(f"   LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
logger.info(f"   LANGSMITH_API_KEY: {'*' * 10}...{os.getenv('LANGSMITH_API_KEY', '')[-4:]}")
logger.info(f"   LANGSMITH_ENDPOINT: {os.getenv('LANGSMITH_ENDPOINT')}")

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

@app.post("/chat")
def chat(body: ChatIn):
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
    sample = coll.get(limit=limit, include=["documents","metadatas"])
    docs = sample.get("documents", []) or []
    metas = sample.get("metadatas", []) or []
    items = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        items.append({
            "id": m.get("document_id") if isinstance(m, dict) else None,
            "content_preview": (d or "")[:120],
            "content_length": len(d or ""),
            "metadata": m,
        })
    return JSONResponse({
        "collection_name": name,
        "stats": {"total_documents": total, "returned_count": len(docs), "limit": limit},
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
        items.append({
            "id": _id,
            "content_preview": (doc or "")[:200],
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
