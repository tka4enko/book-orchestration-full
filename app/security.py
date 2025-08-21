"""
Security module for BookBot Full
Provides file upload validation, input sanitization, and security utilities
"""

import os
import re
import uuid
import mimetypes
import hashlib
import magic
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from fastapi import HTTPException, UploadFile
import logging

logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration constants"""
    
    # File upload constraints
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.json'}
    
    # MIME type mapping for double validation
    ALLOWED_MIME_TYPES = {
        '.pdf': ['application/pdf'],
        '.docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        '.txt': ['text/plain'],
        '.md': ['text/markdown', 'text/plain'],
        '.json': ['application/json', 'text/json']
    }
    
    # Input validation
    MAX_MESSAGE_LENGTH = 2000
    MAX_SESSION_ID_LENGTH = 100
    
    # Dangerous patterns for prompt injection detection
    PROMPT_INJECTION_PATTERNS = [
        r'ignore\s+(previous\s+|all\s+)?instructions?',
        r'forget\s+(everything|all|previous)',
        r'system\s*:',
        r'assistant\s*:',
        r'human\s*:',
        r'```\s*system',
        r'<\s*system\s*>',
        r'new\s+instructions?',
        r'act\s+as\s+',
        r'pretend\s+(to\s+be|you\s+are)',
        r'roleplay\s+as',
        r'simulate\s+',
        r'override\s+',
        r'jailbreak\s+(mode|prompt|hack|attack)',  # More specific pattern
        r'DAN\s+mode'
    ]
    
    # Educational/legitimate contexts that should not be blocked
    EDUCATIONAL_CONTEXTS = [
        r'что\s+такое\s+jailbreak',
        r'explain\s+jailbreak',
        r'define\s+jailbreak',
        r'about\s+jailbreak',
        r'jailbreak\s+(ios|iphone|android)',
        r'mobile\s+jailbreak',
        r'phone\s+jailbreak'
    ]
    
    # File path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\.[\\/]',
        r'[\\/]\.\.[\\/]',
        r'^[\\/]',
        r'[\\/]\.[\\/]'
    ]

class SecurityValidator:
    """Main security validation class"""
    
    def __init__(self):
        self.config = SecurityConfig()
        
    def validate_file_upload(self, file: UploadFile) -> Tuple[Path, bytes]:
        """
        Comprehensive file upload validation
        
        Returns:
            Tuple[Path, bytes]: Safe file path and content
            
        Raises:
            HTTPException: On validation failure
        """
        try:
            # 1. Basic checks
            if not file.filename:
                raise HTTPException(status_code=400, detail="Filename is required")
                
            # 2. Extension validation
            original_extension = Path(file.filename).suffix.lower()
            if original_extension not in self.config.ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {original_extension} not allowed. Allowed: {', '.join(self.config.ALLOWED_EXTENSIONS)}"
                )
            
            # 3. Filename security
            if self._has_path_traversal(file.filename):
                raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected")
                
            # 4. Read and validate size
            content = file.file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="File is empty")
                
            if len(content) > self.config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File too large. Max size: {self.config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            # 5. MIME type validation using python-magic
            try:
                detected_mime = magic.from_buffer(content, mime=True)
                allowed_mimes = self.config.ALLOWED_MIME_TYPES.get(original_extension, [])
                
                if detected_mime not in allowed_mimes:
                    logger.warning(f"MIME mismatch: detected {detected_mime}, expected {allowed_mimes}")
                    # For some cases, we might want to be more permissive
                    if not self._is_mime_compatible(detected_mime, original_extension):
                        raise HTTPException(
                            status_code=400, 
                            detail=f"File content doesn't match extension. Detected type: {detected_mime}"
                        )
            except Exception as e:
                logger.warning(f"MIME detection failed: {e}. Proceeding with extension validation only.")
            
            # 6. Generate safe filename
            safe_filename = f"{uuid.uuid4().hex}{original_extension}"
            safe_path = Path("/tmp") / "bookbot_uploads" / safe_filename
            
            # Ensure upload directory exists
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"File upload validated: {file.filename} -> {safe_filename} ({len(content)} bytes)")
            return safe_path, content
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File validation error: {e}")
            raise HTTPException(status_code=500, detail="File validation failed")
    
    def sanitize_user_input(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            text: User input text
            max_length: Maximum allowed length
            
        Returns:
            str: Sanitized text
            
        Raises:
            HTTPException: On validation failure
        """
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Input must be a string")
            
        # Length validation
        max_len = max_length or self.config.MAX_MESSAGE_LENGTH
        if len(text) > max_len:
            raise HTTPException(
                status_code=400, 
                detail=f"Input too long. Max length: {max_len} characters"
            )
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Check for prompt injection patterns
        if self._detect_prompt_injection(cleaned):
            logger.warning(f"Prompt injection attempt detected: {cleaned[:100]}...")
            raise HTTPException(
                status_code=400, 
                detail="Input contains potentially harmful content"
            )
        
        # Remove dangerous characters but preserve normal text
        # Keep Unicode characters for multilingual support
        cleaned = self._remove_dangerous_chars(cleaned)
        
        return cleaned
    
    def sanitize_session_id(self, session_id: str) -> str:
        """Sanitize session ID"""
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
            
        if len(session_id) > self.config.MAX_SESSION_ID_LENGTH:
            raise HTTPException(status_code=400, detail="Session ID too long")
            
        # Only allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9\-_]+$', session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID format")
            
        return session_id
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize metadata"""
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Metadata must be a dictionary")
        
        sanitized = {}
        allowed_keys = {'title', 'author', 'isbn', 'isbn13', 'isbn10', 'year', 'prefer_llm', 'max_chunks'}
        
        for key, value in metadata.items():
            if key not in allowed_keys:
                logger.warning(f"Unknown metadata key ignored: {key}")
                continue
                
            # Sanitize string values
            if isinstance(value, str):
                sanitized[key] = self.sanitize_user_input(value, max_length=500)
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            else:
                logger.warning(f"Invalid metadata value type for {key}: {type(value)}")
        
        return sanitized
    
    def _has_path_traversal(self, filename: str) -> bool:
        """Check for path traversal attempts"""
        for pattern in self.config.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, filename):
                return True
        return False
    
    def _detect_prompt_injection(self, text: str) -> bool:
        """Detect potential prompt injection attempts with educational context awareness"""
        text_lower = text.lower()
        
        # First check if this is an educational/legitimate context
        if self._is_educational_context(text_lower):
            return False
        
        for pattern in self.config.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                return True
                
        # Additional heuristics
        # Check for suspicious role-playing attempts
        role_keywords = ['system', 'assistant', 'user', 'human', 'ai', 'bot']
        colon_count = text.count(':')
        if colon_count > 2 and any(keyword in text_lower for keyword in role_keywords):
            return True
            
        # Check for excessive special characters (possible encoding attempts)
        special_char_ratio = len(re.findall(r'[^\w\s\-\.\,\!\?\:\"\'()]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            return True
            
        return False
    
    def _is_educational_context(self, text_lower: str) -> bool:
        """Check if text appears to be educational/legitimate rather than malicious"""
        for pattern in self.config.EDUCATIONAL_CONTEXTS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _remove_dangerous_chars(self, text: str) -> str:
        """Remove potentially dangerous characters while preserving readability"""
        # Remove control characters except for common whitespace
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        # Remove excessive repeated characters (potential DoS)
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)  # Max 3 repeated chars
        
        return text
    
    def _is_mime_compatible(self, detected_mime: str, extension: str) -> bool:
        """Check if detected MIME type is compatible with file extension"""
        # Some compatibility rules for common edge cases
        compatibility_rules = {
            '.txt': ['text/plain', 'application/octet-stream', 'text/x-plain'],
            '.md': ['text/plain', 'text/markdown', 'application/octet-stream'],
            '.json': ['text/plain', 'application/json', 'text/json']
        }
        
        compatible_types = compatibility_rules.get(extension, [])
        return detected_mime in compatible_types

class SecurityLogger:
    """Security event logger"""
    
    def __init__(self):
        self.security_logger = logging.getLogger('bookbot.security')
        
    def log_security_event(self, event_type: str, details: Dict[str, Any], client_ip: str = None):
        """Log security events for monitoring"""
        log_data = {
            'event_type': event_type,
            'timestamp': None,  # Will be added by logging formatter
            'client_ip': client_ip,
            'details': details
        }
        
        if event_type in ['prompt_injection', 'file_upload_blocked', 'rate_limit_exceeded']:
            self.security_logger.warning(f"SECURITY EVENT: {event_type}", extra=log_data)
        else:
            self.security_logger.info(f"Security event: {event_type}", extra=log_data)

# Singleton instances
security_validator = SecurityValidator()
security_logger = SecurityLogger()

def get_client_ip(request) -> str:
    """Extract client IP from request"""
    # Check for forwarded headers first (for reverse proxy setups)
    forwarded_for = getattr(request, 'headers', {}).get('x-forwarded-for')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    
    real_ip = getattr(request, 'headers', {}).get('x-real-ip')
    if real_ip:
        return real_ip
    
    # Fallback to direct client
    client = getattr(request, 'client', None)
    if client:
        return client.host
    
    return 'unknown'