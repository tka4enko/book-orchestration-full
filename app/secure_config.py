"""
Secure configuration management for BookBot
"""

import os
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv, find_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv(), override=False)

class SecurityConfig(BaseModel):
    """Security-related configuration"""
    
    # Rate limiting
    chat_rate_limit: str = Field(default="20/minute", description="Rate limit for chat endpoints")
    upload_rate_limit: str = Field(default="10/hour", description="Rate limit for file uploads")
    health_rate_limit: str = Field(default="100/minute", description="Rate limit for health checks")
    ui_rate_limit: str = Field(default="30/minute", description="Rate limit for UI endpoints")
    websocket_message_limit: int = Field(default=100, description="Max messages per WebSocket session")
    
    # File upload security
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Maximum file size in bytes")
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md", ".json"], 
        description="Allowed file extensions"
    )
    upload_timeout: float = Field(default=120.0, description="Upload processing timeout in seconds")
    
    # Input validation
    max_message_length: int = Field(default=2000, description="Maximum message length")
    max_session_id_length: int = Field(default=100, description="Maximum session ID length")
    max_chat_history_size: int = Field(default=50, description="Maximum chat history entries")
    
    # Request timeouts
    llm_timeout: float = Field(default=15.0, description="LLM request timeout")
    search_timeout: float = Field(default=30.0, description="Search operation timeout")
    websocket_timeout: float = Field(default=20.0, description="WebSocket message timeout")
    
    # Circuit breaker
    circuit_breaker_failure_threshold: int = Field(default=5, description="Failures before circuit opens")
    circuit_breaker_recovery_timeout: float = Field(default=60.0, description="Recovery timeout in seconds")
    
    @validator('max_file_size')
    def validate_file_size(cls, v):
        if v < 1024 * 1024 or v > 100 * 1024 * 1024:  # 1MB to 100MB
            raise ValueError("File size must be between 1MB and 100MB")
        return v

class DatabaseConfig(BaseModel):
    """Database and storage configuration"""
    
    chroma_dir: str = Field(default=".chroma", description="ChromaDB storage directory")
    temp_dir: str = Field(default="/tmp/bookbot_uploads", description="Temporary upload directory")
    logs_dir: str = Field(default="logs", description="Logs directory")
    
    # Chunk processing
    chunk_size: int = Field(default=1200, description="Text chunk size")
    chunk_overlap: int = Field(default=120, description="Text chunk overlap")
    max_chunks: int = Field(default=80, description="Maximum chunks per document")
    
    # Search parameters
    bm25_k: int = Field(default=8, description="BM25 results count")
    vec_books_k: int = Field(default=8, description="Vector books results count")  
    vec_content_k: int = Field(default=6, description="Vector content results count")
    rrf_k: int = Field(default=60, description="Reciprocal rank fusion parameter")
    
    # Performance limits
    max_chunks_per_book: int = Field(default=2, description="Max chunks per book in results")
    chunks_per_book_in_content: int = Field(default=2, description="Chunks per book in content search")
    content_search_expand_k: int = Field(default=20, description="Expanded search for diversity")

class LLMConfig(BaseModel):
    """LLM service configuration"""
    
    openai_api_key: str = Field(description="OpenAI API key")
    chat_model: str = Field(default="gpt-4o-mini", description="Chat model name")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    
    # LLM parameters
    chat_temperature: float = Field(default=0.0, description="Temperature for chat model")
    creative_temperature: float = Field(default=0.7, description="Temperature for creative responses")
    max_retries: int = Field(default=2, description="Maximum LLM retry attempts")
    
    # Thresholds
    min_similarity_threshold: float = Field(default=0.6, description="Legacy minimum similarity threshold")
    
    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v or len(v) < 40:
            raise ValueError("OpenAI API key is required and must be valid")
        return v
    
    @validator('chat_temperature', 'creative_temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

class ThresholdConfig(BaseModel):
    """Configurable thresholds for search and filtering"""
    
    # Intent-based similarity thresholds
    similarity_thresholds: Dict[str, float] = Field(default={
        "author": 0.8,
        "author_title": 0.8, 
        "isbn": 0.9,
        "title": 0.7,
        "topic": 0.4,
        "genre": 0.5,
        "free_text": 0.5,
        "clarify": 0.3
    })
    
    # Intent-based BM25 thresholds
    bm25_thresholds: Dict[str, float] = Field(default={
        "isbn": 0.05,
        "author": 0.1,
        "title": 0.1,
        "author_title": 0.1,
        "genre": 0.1,
        "topic": 0.6,
        "free_text": 0.4,
        "clarify": 0.3
    })
    
    # Language threshold modifiers
    language_threshold_modifiers: Dict[str, float] = Field(default={
        "same_language": -0.1,
        "different_language": 0.0,
        "unknown_language": -0.05
    })
    
    @validator('similarity_thresholds', 'bm25_thresholds')
    def validate_threshold_values(cls, v):
        for intent, threshold in v.items():
            if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
                raise ValueError(f"Threshold for {intent} must be between 0.0 and 1.0")
        return v

class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration"""
    
    # LangSmith configuration
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_project: Optional[str] = Field(default=None, description="LangSmith project name")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langsmith_endpoint: Optional[str] = Field(default=None, description="LangSmith endpoint URL")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    structured_logging: bool = Field(default=True, description="Enable structured JSON logging")
    log_to_files: bool = Field(default=True, description="Enable file logging")
    
    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    metrics_retention_days: int = Field(default=7, description="Metrics retention period")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

class ServerConfig(BaseModel):
    """Server configuration"""
    
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    
    @validator('port')
    def validate_port(cls, v):
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

class SecureAppConfig(BaseModel):
    """Main application configuration"""
    
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    class Config:
        env_prefix = ""
        case_sensitive = False

def _load_json_config(env_var: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Safely load JSON configuration from environment variable"""
    try:
        config_str = os.getenv(env_var)
        if config_str:
            parsed = json.loads(config_str)
            if isinstance(parsed, dict):
                return parsed
        return default
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {env_var}: {e}. Using default.")
        return default
    except Exception as e:
        logger.warning(f"Error loading {env_var}: {e}. Using default.")
        return default

def load_secure_config() -> SecureAppConfig:
    """Load secure application configuration from environment"""
    
    # Load LLM configuration
    llm_config = LLMConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-small")
    )
    
    # Load security configuration
    security_config = SecurityConfig(
        max_file_size=int(os.getenv("MAX_FILE_SIZE", "10485760")),  # 10MB
        upload_timeout=float(os.getenv("UPLOAD_TIMEOUT", "120.0")),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "15.0")),
        search_timeout=float(os.getenv("SEARCH_TIMEOUT", "30.0"))
    )
    
    # Load database configuration
    database_config = DatabaseConfig(
        chroma_dir=os.getenv("CHROMA_DIR", ".chroma"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        max_chunks=int(os.getenv("MAX_CHUNKS", "80")),
        max_chunks_per_book=int(os.getenv("MAX_CHUNKS_PER_BOOK", "2")),
        chunks_per_book_in_content=int(os.getenv("CHUNKS_PER_BOOK_IN_CONTENT", "2")),
        content_search_expand_k=int(os.getenv("CONTENT_SEARCH_EXPAND_K", "20"))
    )
    
    # Load threshold configuration from JSON environment variables
    thresholds_config = ThresholdConfig(
        similarity_thresholds=_load_json_config("SIMILARITY_THRESHOLDS", {
            "author": 0.8, "author_title": 0.8, "isbn": 0.9, "title": 0.7,
            "topic": 0.4, "genre": 0.5, "free_text": 0.5, "clarify": 0.3
        }),
        bm25_thresholds=_load_json_config("BM25_THRESHOLDS", {
            "isbn": 0.05, "author": 0.1, "title": 0.1, "author_title": 0.1,
            "genre": 0.1, "topic": 0.6, "free_text": 0.4, "clarify": 0.3
        }),
        language_threshold_modifiers=_load_json_config("LANGUAGE_THRESHOLD_MODIFIERS", {
            "same_language": -0.1, "different_language": 0.0, "unknown_language": -0.05
        })
    )
    
    # Load observability configuration
    observability_config = ObservabilityConfig(
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        langsmith_project=os.getenv("LANGSMITH_PROJECT"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT"),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )
    
    # Load server configuration
    server_config = ServerConfig(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=os.getenv("DEBUG", "false").lower() == "true"
    )
    
    # Combine all configurations
    app_config = SecureAppConfig(
        llm=llm_config,
        security=security_config,
        database=database_config,
        thresholds=thresholds_config,
        observability=observability_config,
        server=server_config
    )
    
    logger.info("🔧 Secure configuration loaded successfully")
    logger.info(f"   Security: File size limit {security_config.max_file_size // (1024*1024)}MB, Timeouts enabled")
    logger.info(f"   LLM: {llm_config.chat_model}, Retries: {llm_config.max_retries}")
    logger.info(f"   Database: {database_config.chroma_dir}, Chunks: {database_config.max_chunks}")
    logger.info(f"   Observability: LangSmith {'enabled' if observability_config.langsmith_tracing else 'disabled'}")
    
    return app_config

def validate_config(config: SecureAppConfig) -> bool:
    """Validate configuration for production readiness"""
    issues = []
    
    # Check critical security settings
    if config.llm.openai_api_key == "" or len(config.llm.openai_api_key) < 40:
        issues.append("OpenAI API key is missing or invalid")
    
    if config.server.debug and config.server.host != "127.0.0.1":
        issues.append("Debug mode should not be enabled in production")
    
    if config.security.max_file_size > 50 * 1024 * 1024:  # 50MB
        issues.append("File size limit is very high - consider reducing")
    
    # Check directory permissions
    try:
        Path(config.database.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(config.database.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(config.database.logs_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create required directories: {e}")
    
    # Validate thresholds
    for intent, threshold in config.thresholds.similarity_thresholds.items():
        if threshold < 0.0 or threshold > 1.0:
            issues.append(f"Invalid similarity threshold for {intent}: {threshold}")
    
    if issues:
        logger.error("❌ Configuration validation failed:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True

# Global configuration instance
def get_config() -> SecureAppConfig:
    """Get global configuration instance"""
    if not hasattr(get_config, '_config'):
        get_config._config = load_secure_config()
        
        # Validate configuration
        if not validate_config(get_config._config):
            raise ValueError("Configuration validation failed. Check logs for details.")
    
    return get_config._config

# Export commonly used values for backward compatibility
config = get_config()

# Expose key configuration values
OPENAI_API_KEY = config.llm.openai_api_key
OPENAI_MODEL_CHAT = config.llm.chat_model
OPENAI_MODEL_EMBED = config.llm.embedding_model
CHROMA_DIR = config.database.chroma_dir
HOST = config.server.host  
PORT = config.server.port

# Search parameters
BM25_K = config.database.bm25_k
VEC_BOOKS_K = config.database.vec_books_k
VEC_CONTENT_K = config.database.vec_content_k
RRF_K = config.database.rrf_k

# Chunking
CHUNK_SIZE = config.database.chunk_size
CHUNK_OVERLAP = config.database.chunk_overlap
MAX_CHUNKS = config.database.max_chunks

# Thresholds
MIN_SIMILARITY_THRESHOLD = config.llm.min_similarity_threshold
SIMILARITY_THRESHOLDS = config.thresholds.similarity_thresholds
BM25_THRESHOLDS = config.thresholds.bm25_thresholds
LANGUAGE_THRESHOLD_MODIFIERS = config.thresholds.language_threshold_modifiers

# Performance settings
MAX_CHUNKS_PER_BOOK = config.database.max_chunks_per_book
CHUNKS_PER_BOOK_IN_CONTENT = config.database.chunks_per_book_in_content
CONTENT_SEARCH_EXPAND_K = config.database.content_search_expand_k

logger.info("🚀 Secure configuration module initialized")