import os, json, logging
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from ..core.ingestion import ingest_one, ingest_batch, books_store, content_store
from ..agents.search_agent import graph, ChatState
from ..agents.chat_orchestrator import graph_with_chat, ChatStateWithChat
from ..agents.chat_agent import chat_agent_graph, ChatAgentState
from ..services.search_service import process_simple_search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-load NLTK data at startup for better performance
logger.info("🔧 Pre-loading NLTK data...")
try:
    from ..services.hybrid_retriever import _ensure_nltk
    _ensure_nltk()
    logger.info("✅ NLTK data loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ NLTK pre-loading failed: {e}")

# Clear BM25 cache on startup (Railway sleep/restart recovery)
logger.info("🔧 Clearing BM25 cache for Railway restart recovery...")
try:
    from ..services.hybrid_retriever import clear_bm25_cache
    clear_bm25_cache()
    logger.info("✅ BM25 cache cleared - will rebuild on first search")
except Exception as e:
    logger.warning(f"⚠️ BM25 cache clearing failed: {e}")

# Clear ChromaDB lock files on startup (SQLite WAL/SHM cleanup)
logger.info("🔧 Cleaning ChromaDB lock files for startup recovery...")
import glob
from ..infra.settings import CHROMA_DIR
try:
    lock_patterns = [
        f"{CHROMA_DIR}/*.lock",
        f"{CHROMA_DIR}/*-wal", 
        f"{CHROMA_DIR}/*-shm",
        f"{CHROMA_DIR}/**/*.lock",
        f"{CHROMA_DIR}/**/*-wal",
        f"{CHROMA_DIR}/**/*-shm"
    ]
    cleaned = 0
    for pattern in lock_patterns:
        for lock_file in glob.glob(pattern, recursive=True):
            try:
                os.remove(lock_file)
                cleaned += 1
                logger.info(f"🗑️ Removed lock file: {lock_file}")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {lock_file}: {e}")
    logger.info(f"✅ ChromaDB cleanup: removed {cleaned} lock files")
except Exception as e:
    logger.warning(f"⚠️ ChromaDB lock cleanup failed: {e}")

# Debug LangSmith configuration
import os
logger.info(f"🔧 LangSmith config:")
logger.info(f"   LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
logger.info(f"   LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
logger.info(f"   LANGSMITH_API_KEY: {'*' * 10}...{os.getenv('LANGSMITH_API_KEY', '')[-4:]}")
logger.info(f"   LANGSMITH_ENDPOINT: {os.getenv('LANGSMITH_ENDPOINT')}")

# Ensure data directories exist (Railway Volume mount support)
from ..infra.settings import CHROMA_DIR, FILE_HASH_STORE_PATH
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
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "..", "static")), name="static")

class ChatIn(BaseModel):
    session_id: str
    message: str

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "static", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/chat_test", response_class=HTMLResponse)
def chat_test():
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "static", "chat_test.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/simple-chat", response_class=HTMLResponse)
def simple_chat_page():
    """Web interface for testing simple search"""
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "static", "simple_chat.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/chat_agent", response_class=HTMLResponse)
def chat_agent_page():
    """Web interface for testing chat-agent"""
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "static", "chat_agent.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/chat")
def chat(body: ChatIn):
    import time
    import uuid
    start_time = time.time()
    request_id = uuid.uuid4().hex[:8]

    logger.info("=" * 80)
    logger.info(f"🔥 CHAT REQUEST STARTED [ID: {request_id}]")
    logger.info(f"   Session: {body.session_id}")
    logger.info(f"   Query: '{body.message}'")
    logger.info("=" * 80)
    
    try:
        state = ChatState(session_id=body.session_id, message=body.message)
        logger.info(f"📋 [main.py] [{request_id}] Creating ChatState and invoking LangGraph...")

        out = graph.invoke(state)

        logger.info(f"✅ [main.py] [{request_id}] LangGraph execution completed")
        
        # Finalize debug session here - at the very end of request
        from ..infra.debug import finalize_debug
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
                    
            logger.info(f"📤 [main.py] [{request_id}] Response prepared with {len(payload.get('results', []))} results")
            logger.info("=" * 80)
            logger.info(f"🎉 CHAT REQUEST COMPLETED SUCCESSFULLY [ID: {request_id}]")
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
        
        # Simplify response for clean presentation - remove technical details
        simple_result = {
            "response": result.get("response", "Sorry, no results found."),
            "results": result.get("results", [])
        }

        logger.info(f"📤 [simple_chat] Simple response prepared with {len(simple_result.get('results', []))} results")
        logger.info("=" * 80)
        logger.info("🎉 SIMPLE CHAT REQUEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return JSONResponse(simple_result)
        
    except Exception as e:
        logger.error(f"💥 [simple_chat] Simple chat request failed: {e}")
        logger.error("=" * 80)
        return JSONResponse({
            "response": "Sorry, an error occurred while processing the request.",
            "results": []
        }, status_code=500)

@app.post("/ingest")
async def ingest_single_endpoint(
    file: UploadFile = File(...),
    meta: Optional[str] = Form(None),
    prefer_llm: Optional[str] = Form(None),
    max_chunks: Optional[str] = Form(None),
    force_ingest: Optional[str] = Form(None)
):
    """Single file upload endpoint"""
    import uuid
    import asyncio

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"📁 [{request_id}] Single file ingest: {file.filename}")

    # Parse parameters
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

    force_flag = False
    if force_ingest is not None:
        v = str(force_ingest).strip().lower()
        force_flag = v in ("1","true","yes","on")

    try:
        # Save file to temp location
        temp_path = os.path.join("/tmp", f"{request_id}_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Process with existing ingest_one function
        result = await asyncio.to_thread(ingest_one, temp_path, meta_json, force_flag)

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        # Handle response
        if result.get("status") == "duplicate_detected":
            logger.warning(f"🚨 [{request_id}] Single ingestion blocked: {result.get('message')}")
            return JSONResponse(result, status_code=409)
        else:
            logger.info(f"🎉 [{request_id}] Single ingest completed! Book ID: {result.get('book_id')}")
            return JSONResponse(result)

    except Exception as e:
        # Clean up temp file on error
        try:
            temp_path = os.path.join("/tmp", f"{request_id}_{file.filename}")
            os.remove(temp_path)
        except Exception:
            pass

        logger.error(f"❌ [{request_id}] Single ingest failed: {e}")
        return JSONResponse({"error": f"Single ingest failed: {e}"}, status_code=500)

@app.post("/ingest-batch")
async def ingest_batch_endpoint(request: Request):
    """
    Batch files upload with manual form parsing to handle multiple files with same key 'files'
    """
    import uuid
    import asyncio

    request_id = uuid.uuid4().hex[:8]
    logger.info(f"📦 [{request_id}] Batch ingest request received")

    try:
        # Get content type and check if it's multipart
        content_type = request.headers.get("content-type", "")
        logger.info(f"📥 [{request_id}] Content-Type: {content_type}")

        if not content_type.startswith("multipart/form-data"):
            return JSONResponse({"error": "Content-Type must be multipart/form-data"}, status_code=400)

        # Try different parsing approaches
        files = []
        meta = None
        prefer_llm = None
        max_chunks = None
        force_ingest = None

        try:
            # First attempt: Standard FastAPI form parsing
            form = await request.form()
            logger.info(f"📥 [{request_id}] Standard parsing successful, keys: {list(form.keys())}")

            # NEW APPROACH: Iterate through all form items to find ALL files
            logger.info(f"📁 [{request_id}] Checking all form items...")
            for key, value in form.items():
                logger.info(f"📁 [{request_id}] Form item: {key} = {type(value)} {getattr(value, 'filename', 'no filename')}")

                # Check if it's a file upload (has filename attribute)
                if hasattr(value, 'filename') and value.filename:
                    files.append(value)
                    logger.info(f"📁 [{request_id}] Found file: {value.filename}")
                elif key == "files" and hasattr(value, 'filename'):
                    files.append(value)
                    logger.info(f"📁 [{request_id}] Found 'files' key file: {value.filename}")

            # Also try getlist as backup
            if "files" in form:
                form_files = form.getlist("files")
                logger.info(f"📁 [{request_id}] form.getlist('files') returned {len(form_files)} items")

                for i, value in enumerate(form_files):
                    if hasattr(value, 'filename') and value.filename:
                        # Check if not already added
                        if value not in files:
                            files.append(value)
                            logger.info(f"📁 [{request_id}] Added from getlist {i+1}: {value.filename}")

            logger.info(f"📦 [{request_id}] Total files found: {len(files)}")

            # Extract other form parameters
            meta = form.get("meta")
            prefer_llm = form.get("prefer_llm")
            max_chunks = form.get("max_chunks")
            force_ingest = form.get("force_ingest")

        except Exception as parse_error:
            logger.warning(f"⚠️ [{request_id}] Standard parsing failed: {parse_error}")

            # Fallback: Try to parse multipart with different approach
            try:
                # Get raw body
                body = await request.body()
                logger.info(f"📥 [{request_id}] Raw body size: {len(body)} bytes")

                # Try email.message approach for multipart parsing
                import email.message
                import io
                from email.mime.multipart import MIMEMultipart

                # Extract boundary properly
                boundary = None
                if "boundary=" in content_type:
                    boundary_part = content_type.split("boundary=")[1]
                    # Remove quotes if present
                    boundary = boundary_part.split(";")[0].strip().strip('"').strip("'")
                    logger.info(f"📥 [{request_id}] Extracted boundary: '{boundary}'")

                if not boundary:
                    return JSONResponse({"error": "No boundary found in Content-Type header"}, status_code=400)

                # Create a simple file counter by splitting on boundary
                parts = body.split(f'--{boundary}'.encode())
                valid_parts = []

                for part in parts:
                    part_str = part.decode('utf-8', errors='ignore')
                    if 'filename=' in part_str and 'Content-Type:' in part_str:
                        valid_parts.append(part_str)
                        logger.info(f"📁 [{request_id}] Found file part in boundary")

                if not valid_parts:
                    return JSONResponse({
                        "error": "No files found in multipart data",
                        "boundary": boundary,
                        "parts_found": len(parts),
                        "instruction": "Use curl: curl -X POST 'http://localhost:8000/ingest-batch' -F 'files=@file1.pdf' -F 'files=@file2.pdf'"
                    }, status_code=400)

                # Return helpful info about what we found
                return JSONResponse({
                    "error": "Multipart parsing failed but files detected",
                    "detected_files": len(valid_parts),
                    "boundary": boundary,
                    "recommendation": "This appears to be a FastAPI/python-multipart compatibility issue with Postman",
                    "solution": "Use curl command instead",
                    "curl_example": f"curl -X POST 'http://localhost:8000/ingest-batch' -F 'files=@file1.pdf' -F 'files=@file2.pdf' -F 'prefer_llm=true'"
                }, status_code=422)

            except Exception as manual_error:
                logger.error(f"❌ [{request_id}] Manual parsing also failed: {manual_error}")
                return JSONResponse({
                    "error": "Complete multipart parsing failure",
                    "details": str(manual_error),
                    "workaround": "Use /ingest endpoint for single files"
                }, status_code=500)

        if not files:
            return JSONResponse({"error": "No files found. Add files with key 'files' in Postman."}, status_code=400)

        logger.info(f"📦 [{request_id}] Extracted {len(files)} files")

        # Parse parameters
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

        force_flag = False
        if force_ingest is not None:
            v = str(force_ingest).strip().lower()
            force_flag = v in ("1","true","yes","on")

        logger.info(f"✅ [{request_id}] Parameters: force={force_flag}, meta={meta_json}")

        # Process each file individually
        results = []
        success_count = 0
        duplicate_count = 0
        error_count = 0

        for i, file in enumerate(files):
            file_num = i + 1
            logger.info(f"📄 [{request_id}] [{file_num}/{len(files)}] Processing: {file.filename}")

            try:
                # Save file to temp location
                temp_path = os.path.join("/tmp", f"{request_id}_{file_num}_{file.filename}")
                with open(temp_path, "wb") as f:
                    f.write(await file.read())

                # Process with existing ingest_one function
                logger.info(f"🔧 [{request_id}] [{file_num}/{len(files)}] Calling ingest_one with: path={temp_path}, meta={meta_json}, force={force_flag}")
                result = await asyncio.to_thread(ingest_one, temp_path, meta_json, force_flag)
                logger.info(f"🔧 [{request_id}] [{file_num}/{len(files)}] ingest_one returned: {result.get('status') if result else 'None'}")

                # Clean up temp file
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

                # Count results by status
                if result.get("status") == "success":
                    success_count += 1
                    logger.info(f"✅ [{request_id}] [{file_num}/{len(files)}] SUCCESS: {result.get('metadata', {}).get('title', 'Unknown')}")
                elif result.get("status") == "duplicate_detected":
                    duplicate_count += 1
                    logger.warning(f"🚨 [{request_id}] [{file_num}/{len(files)}] DUPLICATE: {result.get('detected_title', 'Unknown')} - {result.get('message')}")
                else:
                    error_count += 1
                    logger.error(f"❌ [{request_id}] [{file_num}/{len(files)}] ERROR: {result.get('message', 'Unknown error')}")

                results.append(result)

            except Exception as e:
                error_count += 1

                # Log full traceback for debugging
                import traceback
                logger.error(f"💥 [{request_id}] [{file_num}/{len(files)}] EXCEPTION: {file.filename}")
                logger.error(f"💥 [{request_id}] Error: {e}")
                logger.error(f"💥 [{request_id}] Full traceback: {traceback.format_exc()}")

                error_result = {
                    "status": "error",
                    "message": f"Processing failed: {str(e)}",
                    "file_path": file.filename,
                    "error_type": e.__class__.__name__,
                    "traceback": traceback.format_exc()
                }
                results.append(error_result)
                logger.error(f"💥 [{request_id}] [{file_num}/{len(files)}] EXCEPTION: {file.filename} - {e}")

                # Clean up temp file if it exists
                try:
                    temp_path = os.path.join("/tmp", f"{request_id}_{file_num}_{file.filename}")
                    os.remove(temp_path)
                except Exception:
                    pass

        # Summary
        total_files = len(files)
        logger.info("=" * 50)
        logger.info(f"📦 [{request_id}] BATCH INGESTION COMPLETED")
        logger.info(f"   Total files: {total_files}")
        logger.info(f"   ✅ Successful: {success_count}")
        logger.info(f"   🚨 Duplicates: {duplicate_count}")
        logger.info(f"   ❌ Errors: {error_count}")
        logger.info("=" * 50)

        return JSONResponse({
            "status": "batch_completed",
            "summary": {
                "total_files": total_files,
                "successful": success_count,
                "duplicates": duplicate_count,
                "errors": error_count
            },
            "results": results,
            "force_ingest": force_flag,
            "request_id": request_id
        })

    except Exception as e:
        logger.error(f"❌ [{request_id}] Batch endpoint failed: {e}")
        return JSONResponse({"error": f"Batch processing failed: {str(e)}"}, status_code=500)


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
        # Show full content for all collections - no truncation
        content_preview = d or ""
            
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
        # Show full content for all collections - no truncation
        content_preview = doc or ""
            
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
    results = [{"content_preview": (d.page_content or ""), "content_length": len(d.page_content or ""), "metadata": d.metadata} for d in docs]
    return JSONResponse({"collection_name": name, "k": k, "query": q, "results": results})

@app.delete("/vector-store/total-reset")
def total_database_reset():
    """
    DANGER: Complete database and hash store reset

    This endpoint will:
    1. Clear ALL records from books collection
    2. Clear ALL records from content collection
    3. Clear ALL file hashes from duplicate detection store
    4. Clear BM25 cache

    USE WITH EXTREME CAUTION - ALL DATA WILL BE LOST!
    """
    import uuid
    reset_id = uuid.uuid4().hex[:8]

    logger.warning("=" * 80)
    logger.warning(f"🚨 [{reset_id}] TOTAL DATABASE RESET INITIATED")
    logger.warning("🚨 THIS WILL DELETE ALL DATA - BOOKS, CONTENT & DUPLICATE DETECTION HASHES")
    logger.warning("=" * 80)

    results = {
        "reset_id": reset_id,
        "timestamp": import_timestamp(),
        "cleared_collections": [],
        "cleared_hashes": 0,
        "cleared_cache": False,
        "errors": []
    }

    try:
        # Step 1: Clear books collection
        logger.warning(f"🗑️ [{reset_id}] Step 1: Clearing books collection...")
        books = books_store()
        books_collection = books._collection
        books_count = books_collection.count()

        # Clear all documents from books collection
        if books_count > 0:
            # Get all IDs first, then delete them
            all_data = books_collection.get(include=[])
            if all_data and all_data.get("ids"):
                books_collection.delete(ids=all_data["ids"])
                logger.warning(f"🗑️ [{reset_id}] Books collection cleared: {len(all_data['ids'])} records deleted")
                results["cleared_collections"].append({"name": "books", "count": len(all_data["ids"])})
            else:
                logger.info(f"ℹ️ [{reset_id}] Books collection has no documents to delete")
                results["cleared_collections"].append({"name": "books", "count": 0})
        else:
            logger.info(f"ℹ️ [{reset_id}] Books collection was already empty")
            results["cleared_collections"].append({"name": "books", "count": 0})

    except Exception as e:
        error_msg = f"Failed to clear books collection: {e}"
        logger.error(f"❌ [{reset_id}] {error_msg}")
        results["errors"].append(error_msg)

    try:
        # Step 2: Clear content collection
        logger.warning(f"🗑️ [{reset_id}] Step 2: Clearing content collection...")
        content = content_store()
        content_collection = content._collection
        content_count = content_collection.count()

        # Clear all documents from content collection
        if content_count > 0:
            # Get all IDs first, then delete them
            all_data = content_collection.get(include=[])
            if all_data and all_data.get("ids"):
                content_collection.delete(ids=all_data["ids"])
                logger.warning(f"🗑️ [{reset_id}] Content collection cleared: {len(all_data['ids'])} records deleted")
                results["cleared_collections"].append({"name": "content", "count": len(all_data["ids"])})
            else:
                logger.info(f"ℹ️ [{reset_id}] Content collection has no documents to delete")
                results["cleared_collections"].append({"name": "content", "count": 0})
        else:
            logger.info(f"ℹ️ [{reset_id}] Content collection was already empty")
            results["cleared_collections"].append({"name": "content", "count": 0})

    except Exception as e:
        error_msg = f"Failed to clear content collection: {e}"
        logger.error(f"❌ [{reset_id}] {error_msg}")
        results["errors"].append(error_msg)

    try:
        # Step 3: Clear file hash store
        logger.warning(f"🗑️ [{reset_id}] Step 3: Clearing file hash store...")
        from ..infra.hash_store import get_file_hash_store
        hash_store = get_file_hash_store()
        hash_count = hash_store.clear_all_hashes()
        results["cleared_hashes"] = hash_count
        logger.warning(f"🗑️ [{reset_id}] File hash store cleared: {hash_count} hashes deleted")

    except Exception as e:
        error_msg = f"Failed to clear file hash store: {e}"
        logger.error(f"❌ [{reset_id}] {error_msg}")
        results["errors"].append(error_msg)

    try:
        # Step 4: Clear BM25 cache
        logger.warning(f"🗑️ [{reset_id}] Step 4: Clearing BM25 cache...")
        from ..services.hybrid_retriever import clear_bm25_cache
        clear_bm25_cache()
        results["cleared_cache"] = True
        logger.warning(f"🗑️ [{reset_id}] BM25 cache cleared")

    except Exception as e:
        error_msg = f"Failed to clear BM25 cache: {e}"
        logger.error(f"❌ [{reset_id}] {error_msg}")
        results["errors"].append(error_msg)

    # Final summary
    total_cleared = sum(c["count"] for c in results["cleared_collections"])

    if results["errors"]:
        logger.error("=" * 80)
        logger.error(f"⚠️ [{reset_id}] RESET COMPLETED WITH ERRORS")
        logger.error(f"   Total records cleared: {total_cleared}")
        logger.error(f"   Hash store cleared: {results['cleared_hashes']} hashes")
        logger.error(f"   BM25 cache cleared: {results['cleared_cache']}")
        logger.error(f"   Errors encountered: {len(results['errors'])}")
        for error in results["errors"]:
            logger.error(f"   - {error}")
        logger.error("=" * 80)

        results["status"] = "completed_with_errors"
        results["message"] = f"Reset completed with {len(results['errors'])} errors. {total_cleared} records and {results['cleared_hashes']} hashes cleared."
        return JSONResponse(results, status_code=207)  # Multi-status for partial success

    else:
        logger.warning("=" * 80)
        logger.warning(f"✅ [{reset_id}] TOTAL RESET COMPLETED SUCCESSFULLY")
        logger.warning(f"   Total records cleared: {total_cleared}")
        logger.warning(f"   Hash store cleared: {results['cleared_hashes']} hashes")
        logger.warning(f"   BM25 cache cleared: {results['cleared_cache']}")
        logger.warning("   Database is now completely empty!")
        logger.warning("=" * 80)

        results["status"] = "success"
        results["message"] = f"Total reset completed successfully. {total_cleared} records and {results['cleared_hashes']} hashes cleared."
        return JSONResponse(results)

def import_timestamp():
    """Get current timestamp for import tracking"""
    import datetime
    return datetime.datetime.now().isoformat()

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
            
            # Create HumanMessage from user input
            from langchain_core.messages import HumanMessage

            # Use thread_id to preserve state between messages
            config = {"configurable": {"thread_id": session_id}}

            # Get existing state from checkpoint to preserve seen_books and other data
            try:
                snapshot = await chat_agent_graph.aget_state(config)
                if snapshot and snapshot.values:
                    # Update existing state with new message
                    current_state = snapshot.values
                    current_state["messages"] = [HumanMessage(content=user_message)]
                    current_state["chat_history"] = chat_history
                else:
                    # First message - create minimal state
                    current_state = ChatAgentState(
                        session_id=session_id,
                        messages=[HumanMessage(content=user_message)],
                        chat_history=chat_history
                    )
            except Exception as e:
                logger.warning(f"⚠️ Could not get checkpoint state: {e}")
                # Fallback to new state
                current_state = ChatAgentState(
                    session_id=session_id,
                    messages=[HumanMessage(content=user_message)],
                    chat_history=chat_history
                )

            result = await chat_agent_graph.ainvoke(current_state, config=config)
            
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
