"""Comprehensive logging configuration with structured logging and monitoring.

This module provides production-grade logging with:
- Structured JSON logging
- Context propagation
- Performance tracking
- Error tracking
- Security event logging
"""

import json
import logging
import sys
import time
import traceback
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from app.core.config import settings


# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request context
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            log_data["user_id"] = user_id
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info", "extra"
            ]:
                log_data[key] = value
        
        return json.dumps(log_data)


class PlainFormatter(logging.Formatter):
    """Plain text formatter for development."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as plain text.
        
        Args:
            record: Log record
            
        Returns:
            Plain text formatted log string
        """
        # Color codes for different log levels
        colors = {
            "DEBUG": "\033[36m",     # Cyan
            "INFO": "\033[32m",      # Green
            "WARNING": "\033[33m",   # Yellow
            "ERROR": "\033[31m",     # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        # Get color
        color = colors.get(record.levelname, "")
        
        # Format message
        message = f"{color}[{timestamp}] {record.levelname:8} {record.name:30} {record.getMessage()}{reset}"
        
        # Add exception if present
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
        
        return message


def setup_logging() -> None:
    """Configure application logging."""
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Set formatter based on environment
    if settings.log_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = PlainFormatter()
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger with name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        """Initialize performance logger.
        
        Args:
            operation: Operation name
            logger: Logger instance (optional)
        """
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "PerformanceLogger":
        """Start timing."""
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing and log performance."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type:
            self.logger.error(
                f"Failed: {self.operation}",
                extra={
                    "operation": self.operation,
                    "duration_seconds": duration,
                    "error": str(exc_val),
                }
            )
        else:
            self.logger.info(
                f"Completed: {self.operation}",
                extra={
                    "operation": self.operation,
                    "duration_seconds": duration,
                }
            )


class SecurityLogger:
    """Logger for security events."""
    
    def __init__(self):
        """Initialize security logger."""
        self.logger = get_logger("security")
    
    def log_authentication_attempt(
        self,
        email: str,
        success: bool,
        ip_address: str,
        user_agent: str,
        reason: Optional[str] = None
    ) -> None:
        """Log authentication attempt.
        
        Args:
            email: User email
            success: Whether authentication succeeded
            ip_address: Client IP address
            user_agent: User agent string
            reason: Failure reason (if failed)
        """
        self.logger.info(
            f"Authentication {'succeeded' if success else 'failed'}: {email}",
            extra={
                "event_type": "authentication",
                "email": email,
                "success": success,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "reason": reason,
            }
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        reason: str
    ) -> None:
        """Log authorization failure.
        
        Args:
            user_id: User ID
            resource: Resource being accessed
            action: Action being attempted
            reason: Failure reason
        """
        self.logger.warning(
            f"Authorization failed: {user_id} -> {action} on {resource}",
            extra={
                "event_type": "authorization_failure",
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "reason": reason,
            }
        )
    
    def log_suspicious_activity(
        self,
        activity_type: str,
        details: Dict[str, Any],
        severity: str = "medium"
    ) -> None:
        """Log suspicious activity.
        
        Args:
            activity_type: Type of suspicious activity
            details: Activity details
            severity: Severity level (low, medium, high, critical)
        """
        self.logger.warning(
            f"Suspicious activity detected: {activity_type}",
            extra={
                "event_type": "suspicious_activity",
                "activity_type": activity_type,
                "severity": severity,
                "details": details,
            }
        )


# Global instances
security_logger = SecurityLogger()


# Initialize logging on module import
setup_logging()

