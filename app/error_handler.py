"""
Enhanced error handling and logging system for BookBot
"""

import logging
import traceback
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from functools import wraps
import asyncio

from .security import security_logger

class ErrorType(str, Enum):
    """Error type classification"""
    VALIDATION_ERROR = "validation_error"
    SECURITY_ERROR = "security_error"
    LLM_ERROR = "llm_error"
    SEARCH_ERROR = "search_error"
    DATABASE_ERROR = "database_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    FILE_ERROR = "file_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BookBotError(Exception):
    """Base exception class for BookBot errors"""
    
    def __init__(
        self, 
        message: str, 
        error_type: ErrorType = ErrorType.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or self._default_user_message()
        self.timestamp = datetime.now(timezone.utc)
        
    def _default_user_message(self) -> str:
        """Generate user-friendly error message"""
        user_messages = {
            ErrorType.VALIDATION_ERROR: "Пожалуйста, проверьте введенные данные",
            ErrorType.SECURITY_ERROR: "Обнаружена потенциально небезопасная активность", 
            ErrorType.LLM_ERROR: "Сервис анализа временно недоступен",
            ErrorType.SEARCH_ERROR: "Поиск временно недоступен",
            ErrorType.DATABASE_ERROR: "База данных временно недоступна",
            ErrorType.TIMEOUT_ERROR: "Запрос выполняется слишком долго",
            ErrorType.RATE_LIMIT_ERROR: "Превышен лимит запросов",
            ErrorType.FILE_ERROR: "Ошибка обработки файла",
            ErrorType.UNKNOWN_ERROR: "Произошла неожиданная ошибка"
        }
        return user_messages.get(self.error_type, "Произошла ошибка")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging"""
        return {
            "error_message": self.message,  # Renamed to avoid LogRecord conflict
            "error_type": self.error_type,
            "severity": self.severity,
            "details": self.details,
            "user_message": self.user_message,
            "timestamp": self.timestamp.isoformat(),
            "traceback": traceback.format_exc()
        }

class ErrorLogger:
    """Enhanced error logging with structured output"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup different loggers
        self.error_logger = self._setup_logger("bookbot.errors", "errors.log")
        self.security_logger = self._setup_logger("bookbot.security", "security.log")
        self.performance_logger = self._setup_logger("bookbot.performance", "performance.log")
        self.audit_logger = self._setup_logger("bookbot.audit", "audit.log")
        
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        """Setup individual logger with file handler"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
            
        # File handler
        file_handler = logging.FileHandler(self.log_dir / filename)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logs
        formatter = JsonFormatter()
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def log_error(
        self, 
        error: BookBotError, 
        context: Optional[Dict[str, Any]] = None,
        client_ip: Optional[str] = None
    ):
        """Log error with context"""
        log_data = error.to_dict()
        log_data.update({
            "client_ip": client_ip,
            "context": context or {}
        })
        
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.error_logger.error("BookBot Error", extra=log_data)
        else:
            self.error_logger.warning("BookBot Error", extra=log_data)
            
        # Also log security errors separately
        if error.error_type == ErrorType.SECURITY_ERROR:
            self.security_logger.warning("Security Event", extra=log_data)
    
    def log_performance(
        self, 
        operation: str, 
        duration: float, 
        context: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics"""
        log_data = {
            "operation": operation,
            "duration": duration,
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if duration > 10.0:  # Slow operations
            self.performance_logger.warning("Slow Operation", extra=log_data)
        else:
            self.performance_logger.info("Performance", extra=log_data)
    
    def log_audit(
        self,
        action: str,
        user_id: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log audit events"""
        log_data = {
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "result": result,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.audit_logger.info("Audit Event", extra=log_data)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields from the record
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", "filename", 
                              "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
                              "thread", "threadName", "processName", "process", "message"]:
                    log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)

# Global error logger instance
error_logger = ErrorLogger()

def handle_errors(
    error_type: ErrorType = ErrorType.UNKNOWN_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    user_message: Optional[str] = None
):
    """Decorator for error handling"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                
                # Log performance
                duration = time.time() - start_time
                error_logger.log_performance(
                    operation=f"{func.__module__}.{func.__name__}",
                    duration=duration,
                    context={"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                )
                
                return result
            except BookBotError:
                raise  # Re-raise our custom errors
            except Exception as e:
                # Convert to BookBotError
                bookbot_error = BookBotError(
                    message=str(e),
                    error_type=error_type,
                    severity=severity,
                    user_message=user_message,
                    details={
                        "function": f"{func.__module__}.{func.__name__}",
                        "args_count": len(args),
                        "exception_type": type(e).__name__
                    }
                )
                
                # Log the error
                error_logger.log_error(bookbot_error)
                raise bookbot_error
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Log performance
                duration = time.time() - start_time
                error_logger.log_performance(
                    operation=f"{func.__module__}.{func.__name__}",
                    duration=duration
                )
                
                return result
            except BookBotError:
                raise  # Re-raise our custom errors
            except Exception as e:
                # Convert to BookBotError
                bookbot_error = BookBotError(
                    message=str(e),
                    error_type=error_type,
                    severity=severity,
                    user_message=user_message,
                    details={
                        "function": f"{func.__module__}.{func.__name__}",
                        "exception_type": type(e).__name__
                    }
                )
                
                # Log the error
                error_logger.log_error(bookbot_error)
                raise bookbot_error
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def __call__(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise BookBotError(
                        message="Service temporarily unavailable (Circuit Breaker Open)",
                        error_type=ErrorType.TIMEOUT_ERROR,
                        severity=ErrorSeverity.HIGH,
                        user_message="Сервис временно недоступен"
                    )
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise BookBotError(
                        message="Service temporarily unavailable (Circuit Breaker Open)",
                        error_type=ErrorType.TIMEOUT_ERROR,
                        severity=ErrorSeverity.HIGH,
                        user_message="Сервис временно недоступен"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _on_success(self):
        """Reset circuit breaker on success"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failure in circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "active_sessions": 0,
            "last_error": None,
            "uptime_start": datetime.now(timezone.utc)
        }
        self.response_times = []
    
    def record_request(self, success: bool, response_time: float):
        """Record request metrics"""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
            
        # Update response time average (keep last 100 requests)
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
            
        self.metrics["average_response_time"] = sum(self.response_times) / len(self.response_times)
    
    def record_error(self, error: BookBotError):
        """Record error in metrics"""
        self.metrics["last_error"] = {
            "timestamp": error.timestamp.isoformat(),
            "type": error.error_type,
            "severity": error.severity,
            "message": error.message
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        total_requests = self.metrics["total_requests"]
        if total_requests == 0:
            error_rate = 0.0
        else:
            error_rate = self.metrics["failed_requests"] / total_requests
        
        uptime = datetime.now(timezone.utc) - self.metrics["uptime_start"]
        
        status = {
            "status": "healthy" if error_rate < 0.1 else "degraded" if error_rate < 0.3 else "unhealthy",
            "uptime_seconds": uptime.total_seconds(),
            "error_rate": error_rate,
            "total_requests": total_requests,
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "average_response_time": round(self.metrics["average_response_time"], 3),
            "last_error": self.metrics["last_error"]
        }
        
        return status

# Global instances
health_monitor = HealthMonitor()

# Specific exception classes
class ValidationError(BookBotError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            details=details,
            user_message="Пожалуйста, проверьте введенные данные"
        )

class SecurityError(BookBotError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type=ErrorType.SECURITY_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            user_message="Обнаружена подозрительная активность"
        )

class LLMError(BookBotError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type=ErrorType.LLM_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            user_message="Сервис анализа временно недоступен"
        )

class SearchError(BookBotError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type=ErrorType.SEARCH_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            user_message="Поиск временно недоступен"
        )

class TimeoutError(BookBotError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type=ErrorType.TIMEOUT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            user_message="Запрос выполняется слишком долго, попробуйте позже"
        )

# Utility functions
def log_audit_event(action: str, user_id: str, resource: str, result: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for audit logging"""
    error_logger.log_audit(action, user_id, resource, result, details)

def log_security_event(event_type: str, details: Dict[str, Any], client_ip: Optional[str] = None):
    """Convenience function for security logging"""
    security_logger.log_security_event(event_type, details, client_ip)