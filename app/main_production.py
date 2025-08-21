"""
Production-ready secure main.py for BookBot
Integrates all security improvements and error handling
"""

import asyncio
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Import our secure modules
from .secure_config import get_config, OPENAI_API_KEY
from .security import security_validator, get_client_ip
from .error_handler import (
    error_logger, health_monitor, handle_errors, CircuitBreaker,
    BookBotError, ValidationError, SecurityError, LLMError, 
    SearchError, TimeoutError, ErrorType, ErrorSeverity,
    log_audit_event, log_security_event
)
from .secure_orchestrator import secure_graph, SecureChatState

# Legacy imports for compatibility
from .ingest import ingest_one, books_store, content_store
from .orchestrator_with_chat import graph_with_chat, ChatStateWithChat

import logging

# Load secure configuration
config = get_config()

# Setup enhanced logging
logging.basicConfig(level=getattr(logging, config.observability.log_level))
logger = logging.getLogger(__name__)

# LangSmith configuration
if config.observability.langsmith_tracing and config.observability.langsmith_api_key:
    import os
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = config.observability.langsmith_project or "bookbot-production"
    os.environ["LANGSMITH_API_KEY"] = config.observability.langsmith_api_key
    if config.observability.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = config.observability.langsmith_endpoint
    
    logger.info("🔍 LangSmith tracing enabled for production monitoring")

# Circuit breakers for external services
openai_circuit_breaker = CircuitBreaker(
    failure_threshold=config.security.circuit_breaker_failure_threshold,
    recovery_timeout=config.security.circuit_breaker_recovery_timeout,
    expected_exception=Exception
)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="BookBot Production",
    description="Production-ready secure intelligent book search system",
    version="2.0.0",
    docs_url="/docs" if config.server.debug else None,  # Disable docs in production
    redoc_url="/redoc" if config.server.debug else None
)

# Add middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class SecureChatIn(BaseModel):
    """Secure chat input model"""
    session_id: str
    message: str

@app.middleware("http")
async def security_and_monitoring_middleware(request: Request, call_next):
    """Comprehensive security and monitoring middleware"""
    start_time = time.time()
    client_ip = get_client_ip(request)
    
    # Log request start
    logger.debug(f"🔒 Request from {client_ip}: {request.method} {request.url.path}")
    
    # Record active request
    health_monitor.metrics["active_sessions"] += 1
    
    try:
        response = await call_next(request)
        
        # Record successful request
        processing_time = time.time() - start_time
        health_monitor.record_request(success=True, response_time=processing_time)
        
        # Log slow requests
        if processing_time > 5.0:
            logger.warning(f"🐌 Slow request from {client_ip}: {processing_time:.2f}s for {request.url.path}")
            log_security_event(
                "slow_request",
                {
                    "path": request.url.path,
                    "method": request.method,
                    "processing_time": processing_time
                },
                client_ip
            )
        
        return response
        
    except Exception as e:
        # Record failed request
        processing_time = time.time() - start_time
        health_monitor.record_request(success=False, response_time=processing_time)
        
        # Log and handle error
        if isinstance(e, BookBotError):
            health_monitor.record_error(e)
            error_logger.log_error(e, context={"path": request.url.path, "method": request.method}, client_ip=client_ip)
        else:
            # Convert unknown errors to BookBotError
            bookbot_error = BookBotError(
                message=str(e),
                error_type=ErrorType.UNKNOWN_ERROR,
                severity=ErrorSeverity.MEDIUM,
                details={"path": request.url.path, "method": request.method, "exception_type": type(e).__name__}
            )
            health_monitor.record_error(bookbot_error)
            error_logger.log_error(bookbot_error, client_ip=client_ip)
        
        raise
    finally:
        # Record request completion
        health_monitor.metrics["active_sessions"] -= 1

# Health and monitoring endpoints
@app.get("/health")
@limiter.limit(config.security.health_rate_limit)
async def health_check(request: Request):
    """Comprehensive health check"""
    try:
        health_status = health_monitor.get_health_status()
        
        # Add configuration status
        health_status.update({
            "version": "2.0.0",
            "security": "enabled",
            "openai_configured": bool(OPENAI_API_KEY and len(OPENAI_API_KEY) > 40),
            "langsmith_enabled": config.observability.langsmith_tracing
        })
        
        return JSONResponse(health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=500)

@app.get("/metrics")
@limiter.limit("60/minute")
async def metrics_endpoint(request: Request):
    """Metrics endpoint for monitoring (consider authentication in production)"""
    if not config.observability.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return JSONResponse({
        "metrics": health_monitor.metrics,
        "timestamp": time.time()
    })

# Main UI endpoints
@app.get("/")
@limiter.limit(config.security.ui_rate_limit)
@handle_errors(ErrorType.FILE_ERROR, ErrorSeverity.LOW)
async def index(request: Request):
    """Secure main interface"""
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        log_audit_event(
            action="ui_access",
            user_id=get_client_ip(request),
            resource="main_ui",
            result="success"
        )
        
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Interface not found")

@app.get("/chat_test")
@limiter.limit(config.security.ui_rate_limit)
@handle_errors(ErrorType.FILE_ERROR, ErrorSeverity.LOW)
async def chat_test(request: Request):
    """Secure chat test interface"""
    try:
        with open("app/static/chat_test.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        log_audit_event(
            action="chat_ui_access",
            user_id=get_client_ip(request),
            resource="chat_ui",
            result="success"
        )
        
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Chat interface not found")

# Main chat endpoint
@app.post("/chat")
@limiter.limit(config.security.chat_rate_limit)
@openai_circuit_breaker
@handle_errors(ErrorType.LLM_ERROR, ErrorSeverity.MEDIUM)
async def secure_chat(request: Request, body: SecureChatIn):
    """Production-ready secure chat endpoint"""
    client_ip = get_client_ip(request)
    
    try:
        # Enhanced input validation
        session_id = security_validator.sanitize_session_id(body.session_id)
        message = security_validator.sanitize_user_input(body.message)
        
        logger.info("=" * 80)
        logger.info(f"🔥 SECURE CHAT REQUEST")
        logger.info(f"   Client: {client_ip}")
        logger.info(f"   Session: {session_id}")
        logger.info(f"   Query length: {len(message)} chars")
        logger.info("=" * 80)
        
        # Create secure state
        state = SecureChatState(session_id=session_id, message=message)
        
        # Execute with timeout protection
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(secure_graph.invoke, state),
                timeout=config.security.search_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                message="Chat request timeout",
                details={"session_id": session_id, "timeout": config.security.search_timeout}
            )
        
        # Process and sanitize response
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        else:
            payload = dict(result) if hasattr(result, '__dict__') else {"result": str(result)}
        
        # Remove sensitive information
        payload = _sanitize_response(payload)
        
        # Log successful completion
        log_audit_event(
            action="chat_request",
            user_id=session_id,
            resource="secure_chat",
            result="success",
            details={"results_count": len(payload.get('results', []))}
        )
        
        logger.info("🎉 SECURE CHAT COMPLETED")
        return JSONResponse(payload)
        
    except ValidationError:
        # Input validation failed
        log_security_event("input_validation_failed", {"session_id": body.session_id}, client_ip)
        raise
    except SecurityError:
        # Security threat detected
        log_security_event("security_threat_detected", {"session_id": body.session_id}, client_ip)
        raise
    except (LLMError, TimeoutError, SearchError):
        # Service errors - already logged by decorators
        raise
    except Exception as e:
        # Unexpected errors
        logger.error(f"💥 Unexpected chat error: {e}")
        raise BookBotError(
            message="Chat processing failed",
            error_type=ErrorType.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details={"session_id": body.session_id, "exception_type": type(e).__name__}
        )

# File upload endpoint  
@app.post("/ingest")
@limiter.limit(config.security.upload_rate_limit)
@handle_errors(ErrorType.FILE_ERROR, ErrorSeverity.HIGH)
async def secure_ingest(
    request: Request,
    file: UploadFile = File(...),
    meta: str = Form(None),
    prefer_llm: str = Form(None),
    max_chunks: str = Form(None),
    force_ingest: str = Form(None)
):
    """Production-ready secure file ingestion"""
    client_ip = get_client_ip(request)
    
    try:
        logger.info(f"📁 Secure ingestion started: {file.filename} from {client_ip}")
        
        # Secure file validation
        safe_path, content = security_validator.validate_file_upload(file)
        
        # Process metadata securely
        metadata = {}
        if meta:
            try:
                import json
                parsed_meta = json.loads(meta)
                metadata = security_validator.validate_metadata(parsed_meta)
            except json.JSONDecodeError:
                raise ValidationError("Invalid metadata format")
        
        # Process flags securely
        prefer_llm_flag = str(prefer_llm or "true").lower() in ("1", "true", "yes", "on")
        force_ingest_flag = str(force_ingest or "false").lower() in ("1", "true", "yes", "on")
        
        if max_chunks:
            try:
                max_chunks_val = int(str(max_chunks).strip())
                if not (1 <= max_chunks_val <= 200):
                    raise ValidationError("max_chunks must be between 1 and 200")
                metadata["max_chunks"] = max_chunks_val
            except ValueError:
                raise ValidationError("max_chunks must be a valid integer")
        
        metadata["prefer_llm"] = prefer_llm_flag
        
        # Save file securely
        with open(safe_path, "wb") as f:
            f.write(content)
        
        # Process with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(ingest_one, str(safe_path), metadata, force_ingest_flag),
                timeout=config.security.upload_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                message="Document processing timeout",
                details={"filename": file.filename, "timeout": config.security.upload_timeout}
            )
        finally:
            # Always clean up
            try:
                safe_path.unlink()
            except Exception:
                pass
        
        # Log result
        if result.get("status") == "duplicate_detected":
            log_audit_event(
                action="upload_blocked",
                user_id=client_ip,
                resource="document_ingest",
                result="duplicate_detected",
                details={"filename": file.filename, "reason": result.get("message")}
            )
            return JSONResponse(result, status_code=409)
        else:
            log_audit_event(
                action="document_uploaded",
                user_id=client_ip,
                resource="document_ingest", 
                result="success",
                details={
                    "filename": file.filename,
                    "book_id": result.get("book_id"),
                    "chunks": result.get("chunks")
                }
            )
            
        return JSONResponse(result)
        
    except ValidationError:
        # Clean up on validation failure
        try:
            if 'safe_path' in locals():
                safe_path.unlink()
        except Exception:
            pass
        raise
    except Exception as e:
        # Clean up on any error
        try:
            if 'safe_path' in locals():
                safe_path.unlink()
        except Exception:
            pass
        
        if not isinstance(e, BookBotError):
            raise BookBotError(
                message=f"File ingestion failed: {str(e)}",
                error_type=ErrorType.FILE_ERROR,
                severity=ErrorSeverity.HIGH,
                details={"filename": file.filename}
            )
        raise

# WebSocket with enhanced security
active_websocket_connections = {}

@app.websocket("/ws/chat_test")
async def secure_websocket_chat(websocket: WebSocket):
    """Production WebSocket with comprehensive security"""
    client_ip = websocket.client.host if websocket.client else "unknown"
    connection_id = f"{client_ip}_{time.time()}"
    
    await websocket.accept()
    active_websocket_connections[connection_id] = {
        "websocket": websocket,
        "client_ip": client_ip,
        "message_count": 0,
        "start_time": time.time()
    }
    
    logger.info(f"🔌 Secure WebSocket connection from {client_ip}")
    
    try:
        while True:
            # Connection timeout check
            if time.time() - active_websocket_connections[connection_id]["start_time"] > 3600:  # 1 hour
                await websocket.send_json({
                    "reply": "Connection timeout. Please refresh to continue.",
                    "error": "connection_timeout"
                })
                break
            
            # Receive and validate message
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=config.security.websocket_timeout
                )
            except asyncio.TimeoutError:
                continue
            
            # Rate limiting
            connection_info = active_websocket_connections[connection_id]
            connection_info["message_count"] += 1
            
            if connection_info["message_count"] > config.security.websocket_message_limit:
                await websocket.send_json({
                    "reply": "Message limit reached. Please refresh to continue.",
                    "error": "rate_limit_exceeded"
                })
                break
            
            # Validate and process message
            try:
                session_id = security_validator.sanitize_session_id(
                    raw_message.get("session_id", "default")
                )
                message = security_validator.sanitize_user_input(
                    raw_message.get("message", "")
                )
            except Exception as e:
                await websocket.send_json({
                    "reply": "Invalid message format. Please try again.",
                    "error": "validation_error"
                })
                continue
            
            logger.debug(f"🧪 WebSocket message from {client_ip}: {message[:50]}...")
            
            try:
                # Use legacy chat orchestrator for WebSocket (simpler for now)
                # In production, you might want to use the secure orchestrator here too
                chat_history = []  # Simplified for now
                state = ChatStateWithChat(
                    session_id=session_id,
                    message=message,
                    chat_history=chat_history
                )
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(graph_with_chat.invoke, state),
                    timeout=config.security.websocket_timeout
                )
                
                # Format secure response
                if hasattr(result, "model_dump"):
                    response_data = result.model_dump()
                else:
                    response_data = result
                
                response = {
                    "reply": response_data.get("results", [{}])[0].get("message", ""),
                    "cards": [],
                    "chips": response_data.get("results", [{}])[0].get("chips", []),
                    "intent": response_data.get("intent", "chat"),
                    "debug": {
                        "mode": "secure_websocket",
                        "connection_id": connection_id[:8],
                        "message_count": connection_info["message_count"]
                    }
                }
                
                # Sanitize response
                response = _sanitize_response(response)
                
                await websocket.send_json(response)
                
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "reply": "Request timeout. Please try a simpler query.",
                    "error": "timeout"
                })
            except Exception as e:
                logger.error(f"🧪 WebSocket processing error: {e}")
                await websocket.send_json({
                    "reply": "Something went wrong. Please try again.",
                    "error": "processing_error"
                })
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {client_ip}")
    except Exception as e:
        logger.error(f"🔌 WebSocket error: {e}")
    finally:
        # Clean up connection
        if connection_id in active_websocket_connections:
            del active_websocket_connections[connection_id]

# Vector store endpoints with security
@app.get("/vector-store/collection/{name}")
@limiter.limit("60/minute")
@handle_errors(ErrorType.DATABASE_ERROR, ErrorSeverity.MEDIUM)
async def secure_collection_info(
    request: Request, 
    name: str, 
    limit: int = Query(3, ge=1, le=20)  # Reduced max limit
):
    """Secure collection information endpoint"""
    if name not in ["books", "content"]:
        raise ValidationError(f"Invalid collection name: {name}")
    
    try:
        store = books_store() if name == "books" else content_store()
        coll = store._collection
        
        # Get limited data
        if name == "books":
            sample = coll.get(
                where={"is_master_chunk": True},
                limit=limit,
                include=["documents", "metadatas"]
            )
            total = coll.count(where={"is_master_chunk": True})
        else:
            sample = coll.get(limit=limit, include=["documents", "metadatas"])
            total = coll.count()
        
        docs = sample.get("documents", []) or []
        metas = sample.get("metadatas", []) or []
        
        items = []
        for doc, meta in zip(docs, metas):
            # Sanitize content
            content_preview = (doc or "")[:200]
            safe_metadata = _sanitize_metadata(meta)
            
            items.append({
                "content_preview": content_preview,
                "content_length": len(doc or ""),
                "metadata": safe_metadata
            })
        
        return JSONResponse({
            "collection_name": name,
            "total_documents": total,
            "returned_count": len(items),
            "documents": items
        })
        
    except Exception as e:
        raise BookBotError(
            message=f"Failed to retrieve collection info: {str(e)}",
            error_type=ErrorType.DATABASE_ERROR,
            severity=ErrorSeverity.MEDIUM
        )

# Utility functions
def _sanitize_response(payload: dict) -> dict:
    """Remove sensitive information from responses"""
    if not isinstance(payload, dict):
        return payload
    
    # Remove sensitive keys
    sensitive_keys = {'api_key', 'password', 'token', 'secret', 'key'}
    
    def clean_dict(obj):
        if isinstance(obj, dict):
            return {k: clean_dict(v) for k, v in obj.items() 
                   if k.lower() not in sensitive_keys}
        elif isinstance(obj, list):
            return [clean_dict(item) for item in obj]
        elif isinstance(obj, str):
            return obj[:1000] + "..." if len(obj) > 1000 else obj
        else:
            return obj
    
    return clean_dict(payload)

def _sanitize_metadata(metadata: dict) -> dict:
    """Sanitize metadata for public display"""
    if not isinstance(metadata, dict):
        return {}
    
    safe_fields = {
        'title', 'author', 'year', 'language', 'primary_genre',
        'secondary_genres', 'isbn13', 'isbn10', 'summary'
    }
    
    return {k: str(v)[:200] if isinstance(v, str) else v 
            for k, v in metadata.items() if k in safe_fields}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 BookBot Production Server Starting")
    logger.info(f"🔒 Security: ENABLED (Rate limits, Input validation, Error handling)")
    logger.info(f"📊 Monitoring: ENABLED (Health checks, Metrics, Audit logs)")  
    logger.info(f"🤖 LLM: {config.llm.chat_model}")
    logger.info(f"🗄️ Database: {config.database.chroma_dir}")
    logger.info(f"🌐 Server: {config.server.host}:{config.server.port}")
    
    # Initialize directories
    import os
    os.makedirs(config.database.temp_dir, exist_ok=True)
    os.makedirs(config.database.logs_dir, exist_ok=True)
    
    logger.info("✅ Production server ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 BookBot Production Server Shutting Down")
    
    # Close active WebSocket connections
    for conn_id, conn_info in active_websocket_connections.items():
        try:
            await conn_info["websocket"].close()
        except Exception:
            pass
    
    logger.info("👋 Shutdown complete")

# Error handlers for HTTP exceptions
@app.exception_handler(BookBotError)
async def bookbot_error_handler(request: Request, exc: BookBotError):
    """Handle BookBot custom errors"""
    client_ip = get_client_ip(request)
    
    # Log error
    error_logger.log_error(exc, client_ip=client_ip)
    
    # Determine HTTP status code
    status_codes = {
        ErrorType.VALIDATION_ERROR: 400,
        ErrorType.SECURITY_ERROR: 403,
        ErrorType.TIMEOUT_ERROR: 504,
        ErrorType.RATE_LIMIT_ERROR: 429,
        ErrorType.LLM_ERROR: 503,
        ErrorType.SEARCH_ERROR: 503,
        ErrorType.DATABASE_ERROR: 503,
        ErrorType.FILE_ERROR: 400,
        ErrorType.UNKNOWN_ERROR: 500
    }
    
    status_code = status_codes.get(exc.error_type, 500)
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.user_message,
            "error_type": exc.error_type,
            "timestamp": exc.timestamp.isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_production:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.observability.log_level.lower()
    )