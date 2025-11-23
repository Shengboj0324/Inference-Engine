"""Comprehensive error handling and custom exceptions."""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Standardized error codes."""

    # General errors (1xxx)
    INTERNAL_ERROR = "1000"
    VALIDATION_ERROR = "1001"
    NOT_FOUND = "1002"
    ALREADY_EXISTS = "1003"
    UNAUTHORIZED = "1004"
    FORBIDDEN = "1005"
    RATE_LIMITED = "1006"

    # Database errors (2xxx)
    DATABASE_ERROR = "2000"
    DATABASE_CONNECTION_ERROR = "2001"
    DATABASE_QUERY_ERROR = "2002"
    DATABASE_CONSTRAINT_ERROR = "2003"

    # Connector errors (3xxx)
    CONNECTOR_ERROR = "3000"
    CONNECTOR_AUTH_ERROR = "3001"
    CONNECTOR_RATE_LIMIT = "3002"
    CONNECTOR_TIMEOUT = "3003"
    CONNECTOR_INVALID_RESPONSE = "3004"

    # Scraping errors (4xxx)
    SCRAPING_ERROR = "4000"
    SCRAPING_BLOCKED = "4001"
    SCRAPING_TIMEOUT = "4002"
    SCRAPING_COMPLIANCE_ERROR = "4003"

    # LLM errors (5xxx)
    LLM_ERROR = "5000"
    LLM_TIMEOUT = "5001"
    LLM_RATE_LIMIT = "5002"
    LLM_INVALID_RESPONSE = "5003"
    LLM_CONTEXT_LENGTH_EXCEEDED = "5004"

    # Output generation errors (6xxx)
    OUTPUT_ERROR = "6000"
    OUTPUT_VALIDATION_ERROR = "6001"
    OUTPUT_FORMAT_ERROR = "6002"

    # Storage errors (7xxx)
    STORAGE_ERROR = "7000"
    STORAGE_UPLOAD_ERROR = "7001"
    STORAGE_DOWNLOAD_ERROR = "7002"


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseAppException(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """Initialize exception.

        Args:
            message: Human-readable error message
            error_code: Standardized error code
            severity: Error severity level
            details: Additional error details
            original_exception: Original exception if wrapped
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        self.original_exception = original_exception

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "severity": self.severity.value,
                "details": self.details,
            }
        }


class ValidationError(BaseAppException):
    """Validation error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            details=details,
        )


class DatabaseError(BaseAppException):
    """Database error."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATABASE_ERROR,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.HIGH,
            original_exception=original_exception,
        )


class ConnectorError(BaseAppException):
    """Connector error."""

    def __init__(
        self,
        message: str,
        platform: str,
        error_code: ErrorCode = ErrorCode.CONNECTOR_ERROR,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.MEDIUM,
            details={"platform": platform},
            original_exception=original_exception,
        )


class LLMError(BaseAppException):
    """LLM error."""

    def __init__(
        self,
        message: str,
        provider: str,
        error_code: ErrorCode = ErrorCode.LLM_ERROR,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.HIGH,
            details={"provider": provider},
            original_exception=original_exception,
        )


class OutputGenerationError(BaseAppException):
    """Output generation error."""

    def __init__(
        self,
        message: str,
        format: str,
        error_code: ErrorCode = ErrorCode.OUTPUT_ERROR,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.MEDIUM,
            details={"format": format},
            original_exception=original_exception,
        )


class RateLimitError(BaseAppException):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        resource: Optional[str] = None,
    ):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        if resource:
            details["resource"] = resource

        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMITED,
            severity=ErrorSeverity.MEDIUM,
            details=details,
        )

