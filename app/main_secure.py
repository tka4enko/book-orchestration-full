import os, json, logging, asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Security imports
from .security import security_validator, security_logger, get_client_ip
from .ingest import ingest_one, books_store, content_store
from .orchestrator import graph, ChatState
from .orchestrator_with_chat import graph_with_chat, ChatStateWithChat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced logging configuration
security_handler = logging.FileHandler('security.log')
security_handler.setLevel(logging.WARNING)
security_formatter = logging.Formatter('%(asctime)s - SECURITY - %(levelname)s - %(message)s')
security_handler.setFormatter(security_formatter)
logging.getLogger('bookbot.security').addHandler(security_handler)

# Debug LangSmith configuration
logger.info(f"🔧 LangSmith config:")
logger.info(f"   LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
logger.info(f"   LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
logger.info(f"   LANGSMITH_API_KEY: {'*' * 10}...{os.getenv('LANGSMITH_API_KEY', '')[-4:]}")
logger.info(f"   LANGSMITH_ENDPOINT: {os.getenv('LANGSMITH_ENDPOINT')}")

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="BookBot Final 4 - Secure",
    description="Secure intelligent book search system with hybrid algorithms",
    version="2.0.0"
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

class SecureChatIn(BaseModel):
    session_id: str
    message: str
    
    def validate_and_sanitize(self):
        """Validate and sanitize input fields"""
        self.session_id = security_validator.sanitize_session_id(self.session_id)
        self.message = security_validator.sanitize_user_input(self.message)
        return self

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for logging and basic protection"""
    start_time = asyncio.get_event_loop().time()
    client_ip = get_client_ip(request)
    
    # Log request
        # CLEANED: logger.debug(f"🔒 Request from {client_ip}: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Log successful requests
        process_time = asyncio.get_event_loop().time() - start_time
        if process_time > 10.0:  # Log slow requests
            logger.warning(f"🐌 Slow request from {client_ip}: {process_time:.2f}s")
            
        return response
        
    except Exception as e:
        # Log errors
        security_logger.log_security_event(
            "request_error", 
            {"error": str(e), "path": request.url.path, "method": request.method},
            client_ip
        )
        logger.error(f"❌ Request error from {client_ip}: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
@limiter.limit("30/minute")  # Generous limit for UI access
async def index(request: Request):
    """Serve main UI with rate limiting"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception as e:
        logger.error(f"Failed to serve index.html: {e}")
        raise HTTPException(status_code=500, detail="Failed to load interface")

@app.get("/health")
@limiter.limit("100/minute")  # Higher limit for health checks
async def health(request: Request):
    """Health check endpoint with rate limiting"""
    return {"ok": True, "version": "2.0.0", "security": "enabled"}

@app.get("/chat_test", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def chat_test(request: Request):
    """Serve chat test UI with rate limiting"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "static", "chat_test.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except Exception as e:
        logger.error(f"Failed to serve chat_test.html: {e}")
        raise HTTPException(status_code=500, detail="Failed to load chat interface")

@app.post("/chat")
@limiter.limit("20/minute")  # Reasonable limit for chat requests
async def chat(request: Request, body: SecureChatIn):
    """Secure chat endpoint with input validation and rate limiting"""
    client_ip = get_client_ip(request)
    
    try:
        # Validate and sanitize input
        body = body.validate_and_sanitize()
        
        logger.info("=" * 80)
        logger.info(f"🔥 SECURE CHAT REQUEST STARTED")
        logger.info(f"   Client IP: {client_ip}")
        logger.info(f"   Session: {body.session_id}")
        logger.info(f"   Query: '{body.message[:100]}{'...' if len(body.message) > 100 else ''}'")
        logger.info("=" * 80)
        
        # Create secure state
        state = ChatState(session_id=body.session_id, message=body.message)
        logger.info("📋 [main_secure.py] Creating ChatState and invoking LangGraph...")
        
        # Execute with timeout protection
        try:
            out = await asyncio.wait_for(
                asyncio.to_thread(graph.invoke, state),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("⏱️ LangGraph execution timeout")
            security_logger.log_security_event(
                "request_timeout",
                {"session_id": body.session_id, "message_length": len(body.message)},
                client_ip
            )
            raise HTTPException(status_code=504, detail="Request timeout")
        
        logger.info("✅ [main_secure.py] LangGraph execution completed")
        
        # Secure serialization
        try:
            if hasattr(out, "model_dump"):
                payload = out.model_dump()
            elif isinstance(out, dict):
                payload = out
            else:
                try:
                    payload = dict(out)
                except Exception:
                    payload = {"result": str(out)}
            
            # Sanitize output (remove any potentially harmful content)
            payload = _sanitize_response_payload(payload)
            
            logger.info(f"📤 [main_secure.py] Response prepared with {len(payload.get('results', []))} results")
            logger.info("=" * 80)
            logger.info("🎉 SECURE CHAT REQUEST COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            return JSONResponse(payload)
            
        except Exception as e:
            logger.error(f"❌ [main_secure.py] Serialization failed: {e}")
            return JSONResponse({"error": "Response processing failed"}, status_code=500)
            
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise
    except Exception as e:
        logger.error(f"💥 [main_secure.py] Chat request failed: {e}")
        security_logger.log_security_event(
            "chat_error",
            {"error": str(e), "session_id": getattr(body, 'session_id', 'unknown')},
            client_ip
        )
        return JSONResponse({"error": "Chat request failed"}, status_code=500)

@app.post("/ingest")
@limiter.limit("10/hour")  # Strict limit for file uploads
async def secure_ingest_endpoint(
    request: Request,
    file: UploadFile = File(...), 
    meta: str = Form(None), 
    prefer_llm: str = Form(None), 
    max_chunks: str = Form(None), 
    force_ingest: str = Form(None)
):
    """Secure file ingestion endpoint with comprehensive validation"""
    client_ip = get_client_ip(request)
    
    logger.info(f"📁 Starting secure ingest process for file: {file.filename} from {client_ip}")
    
    try:
        # 1. Secure file validation
        safe_path, content = security_validator.validate_file_upload(file)
        
        # 2. Save file securely
        logger.info("💾 Saving file securely...")
        with open(safe_path, "wb") as f:
            f.write(content)
        logger.info(f"✅ File saved securely to: {safe_path}")
        
        # 3. Process and validate metadata
        logger.info("⚙️ Processing metadata parameters...")
        meta_json = None
        if meta:
            try:
                meta_parsed = json.loads(meta)
                meta_json = security_validator.validate_metadata(meta_parsed)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Process other parameters securely
        if prefer_llm is not None:
            v = str(prefer_llm).strip().lower()
            flag = v in ("1", "true", "yes", "on")
            meta_json = (meta_json or {})
            meta_json["prefer_llm"] = flag
            
        if max_chunks is not None:
            try:
                mc = int(str(max_chunks).strip())
                if mc < 1 or mc > 200:  # Reasonable limits
                    raise HTTPException(status_code=400, detail="max_chunks must be between 1 and 200")
                meta_json = (meta_json or {})
                meta_json["max_chunks"] = mc
            except ValueError:
                raise HTTPException(status_code=400, detail="max_chunks must be a valid integer")
        
        # Process force_ingest flag
        force_flag = False
        if force_ingest is not None:
            v = str(force_ingest).strip().lower()
            force_flag = v in ("1", "true", "yes", "on")
        
        logger.info(f"✅ Metadata processed: {meta_json}")
        logger.info(f"🛡️ Force ingest: {force_flag}")
        
        # 4. Ingest with timeout protection
        logger.info("🤖 Starting secure document processing...")
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(ingest_one, str(safe_path), meta_json, force_flag),
                timeout=120.0  # 2 minute timeout for ingestion
            )
        except asyncio.TimeoutError:
            logger.error("⏱️ Document ingestion timeout")
            # Clean up uploaded file
            try:
                safe_path.unlink()
            except:
                pass
            raise HTTPException(status_code=504, detail="Document processing timeout")
        
        # 5. Clean up uploaded file
        try:
            safe_path.unlink()
            logger.info("🧹 Temporary file cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")
        
        # Handle different response types
        if res.get("status") == "duplicate_detected":
            logger.warning(f"🚨 Ingestion blocked: {res.get('message')}")
            security_logger.log_security_event(
                "duplicate_upload_blocked",
                {"filename": file.filename, "reason": res.get('message')},
                client_ip
            )
            return JSONResponse(res, status_code=409)
        else:
            logger.info(f"🎉 Secure ingest completed! Book ID: {res.get('book_id')}, Chunks: {res.get('chunks')}")
            security_logger.log_security_event(
                "successful_upload",
                {"filename": file.filename, "book_id": res.get('book_id'), "chunks": res.get('chunks')},
                client_ip
            )
            return JSONResponse(res)
        
    except HTTPException:
        # Clean up on validation failure
        try:
            if 'safe_path' in locals():
                safe_path.unlink()
        except:
            pass
        raise
    except Exception as e:
        logger.error(f"❌ Secure ingest failed: {e.__class__.__name__}: {e}")
        security_logger.log_security_event(
            "ingest_error",
            {"filename": file.filename, "error": str(e)},
            client_ip
        )
        # Clean up on error
        try:
            if 'safe_path' in locals():
                safe_path.unlink()
        except:
            pass
        return JSONResponse({"error": "Ingest failed"}, status_code=500)

# Vector store endpoints with rate limiting
def _get_store(name: str):
    """Get vector store with validation"""
    if name not in ["books", "content"]:
        raise HTTPException(status_code=404, detail="Unknown collection")
    
    if name == "books": 
        return books_store()
    if name == "content": 
        return content_store()
    return None

@app.get("/vector-store/collection/{name}")
@limiter.limit("60/minute")  # Generous limit for data browsing
async def collection_meta(request: Request, name: str, limit: int = Query(3, ge=1, le=50)):  # Reduced max limit
    """Get collection metadata with rate limiting and validation"""
    store = _get_store(name)  # This will raise 404 for invalid names
    
    try:
        coll = store._collection
        total = coll.count()
        
        # For books collection, only show master chunks
        if name == "books":
            sample = coll.get(
                where={"is_master_chunk": True}, 
                limit=limit, 
                include=["documents","metadatas"]
            )
            try:
                master_count = coll.count(where={"is_master_chunk": True})
            except:
                master_count = total
        else:
            sample = coll.get(limit=limit, include=["documents","metadatas"])
            master_count = total
        
        docs = sample.get("documents", []) or []
        metas = sample.get("metadatas", []) or []
        items = []
        
        for i, (d, m) in enumerate(zip(docs, metas)):
            # Sanitize content for display
            if name == "books":
                content_preview = (d or "")[:500]  # Limit preview length
            else:
                content_preview = (d or "")[:120]
                
            items.append({
                "id": m.get("document_id") if isinstance(m, dict) else None,
                "content_preview": content_preview,
                "content_length": len(d or ""),
                "metadata": _sanitize_metadata_for_display(m),
            })
        
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
        
    except Exception as e:
        logger.error(f"Collection metadata error: {e}")
        return JSONResponse({"error": "Failed to retrieve collection data"}, status_code=500)

# WebSocket with enhanced security
chat_histories = {}
MAX_CHAT_HISTORY_SIZE = 50  # Limit chat history size

@app.websocket("/ws/chat_test")
async def secure_chat_test_websocket(websocket: WebSocket):
    """Secure WebSocket endpoint for chat testing"""
    client_ip = websocket.client.host if websocket.client else "unknown"
    
    await websocket.accept()
    logger.info(f"🧪 [ChatTest] WebSocket connection from {client_ip}")
    
    try:
        message_count = 0
        max_messages_per_session = 100  # Prevent spam
        
        while True:
            # Rate limiting for WebSocket messages
            message_count += 1
            if message_count > max_messages_per_session:
                await websocket.send_json({
                    "reply": "Session limit reached. Please refresh to continue.",
                    "error": "rate_limit_exceeded"
                })
                break
            
            raw_message = await websocket.receive_json()
            
            # Validate and sanitize WebSocket message
            try:
                session_id = security_validator.sanitize_session_id(
                    raw_message.get("session_id", "default")
                )
                user_message = security_validator.sanitize_user_input(
                    raw_message.get("message", "")
                )
            except HTTPException as e:
                await websocket.send_json({
                    "reply": "Invalid input. Please check your message.",
                    "error": "validation_error"
                })
                continue
            
            logger.info(f"🧪 [ChatTest] Message from {client_ip} ({session_id}): '{user_message[:50]}{'...' if len(user_message) > 50 else ''}'")
            
            # Manage chat history size
            chat_history = chat_histories.get(session_id, [])
            if len(chat_history) > MAX_CHAT_HISTORY_SIZE:
                chat_history = chat_history[-MAX_CHAT_HISTORY_SIZE:]  # Keep last N messages
            
            try:
                # Use secure chat orchestrator
                state = ChatStateWithChat(
                    session_id=session_id, 
                    message=user_message, 
                    chat_history=chat_history
                )
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(graph_with_chat.invoke, state),
                    timeout=20.0  # Shorter timeout for WebSocket
                )
                
                # Format response
                if hasattr(result, "model_dump"):
                    response_data = result.model_dump()
                else:
                    response_data = result
                
                # Update chat history with size limit
                if hasattr(result, "chat_history"):
                    chat_histories[session_id] = result.chat_history[-MAX_CHAT_HISTORY_SIZE:]
                elif "chat_history" in response_data:
                    chat_histories[session_id] = response_data["chat_history"][-MAX_CHAT_HISTORY_SIZE:]
                
                # Create secure response
                response = {
                    "reply": response_data.get("results", [{}])[0].get("message", ""),
                    "cards": [],
                    "chips": response_data.get("results", [{}])[0].get("chips", []),
                    "intent": response_data.get("intent", "chat"),
                    "debug": {
                        "intent": response_data.get("intent"),
                        "mode": "secure_chat_test",
                        "processing_time": "N/A"
                    }
                }
                
                await websocket.send_json(response)
                
            except asyncio.TimeoutError:
                logger.warning(f"🧪 [ChatTest] Timeout for {client_ip}")
                await websocket.send_json({
                    "reply": "Processing timeout. Please try a simpler query.",
                    "error": "timeout"
                })
            except Exception as e:
                logger.error(f"🧪 [ChatTest] Processing error for {client_ip}: {e}")
                await websocket.send_json({
                    "reply": "Sorry, something went wrong. Please try again.",
                    "error": "processing_error"
                })
                
    except WebSocketDisconnect:
        logger.info(f"🧪 [ChatTest] Client {client_ip} disconnected")
    except Exception as e:
        logger.error(f"🧪 [ChatTest] WebSocket error from {client_ip}: {e}")
        try:
            await websocket.close()
        except:
            pass

def _sanitize_response_payload(payload: dict) -> dict:
    """Sanitize response payload to remove sensitive information"""
    if not isinstance(payload, dict):
        return payload
    
    # Remove potentially sensitive keys
    sensitive_keys = ['api_key', 'password', 'token', 'secret', 'private_key']
    
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items() 
                   if k.lower() not in sensitive_keys}
        elif isinstance(d, list):
            return [clean_dict(item) for item in d]
        elif isinstance(d, str):
            # Truncate very long strings to prevent response bloat
            return d[:2000] + "..." if len(d) > 2000 else d
        else:
            return d
    
    return clean_dict(payload)

def _sanitize_metadata_for_display(metadata: dict) -> dict:
    """Sanitize metadata for public display"""
    if not isinstance(metadata, dict):
        return {}
    
    # Only show safe metadata fields
    safe_fields = {
        'title', 'author', 'year', 'language', 'primary_genre', 
        'secondary_genres', 'isbn13', 'isbn10', 'document_id',
        'is_master_chunk', 'summary'
    }
    
    return {k: v for k, v in metadata.items() if k in safe_fields}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 BookBot Secure server starting up...")
    logger.info("🔒 Security features: ENABLED")
    logger.info("⚡ Rate limiting: ENABLED")
    logger.info("🛡️ Input validation: ENABLED")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 BookBot Secure server shutting down...")
    # Clean up any resources if needed