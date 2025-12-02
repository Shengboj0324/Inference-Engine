"""Comprehensive input validation and sanitization utilities.

This module provides robust validation for all user inputs, API parameters,
and data processing to prevent injection attacks, data corruption, and errors.
"""

import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

from app.core.errors import ValidationError


class EmailValidator(BaseModel):
    """Email validation with strict rules."""
    
    email: EmailStr
    
    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        """Validate email format and domain."""
        # Additional validation beyond EmailStr
        if len(v) > 255:
            raise ValueError("Email too long (max 255 characters)")
        
        # Check for suspicious patterns
        if ".." in v or v.startswith(".") or v.endswith("."):
            raise ValueError("Invalid email format")
        
        return v.lower()


class URLValidator(BaseModel):
    """URL validation with security checks."""
    
    url: str
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format and security."""
        if not v:
            raise ValueError("URL cannot be empty")
        
        if len(v) > 2048:
            raise ValueError("URL too long (max 2048 characters)")
        
        # Parse URL
        try:
            parsed = urlparse(v)
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")
        
        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
        
        # Check for localhost/private IPs in production
        if parsed.hostname:
            hostname = parsed.hostname.lower()
            if hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
                # Allow in development, but log warning
                pass
            
            # Check for private IP ranges
            if hostname.startswith("192.168.") or hostname.startswith("10."):
                # Allow in development, but log warning
                pass
        
        return v


class TextValidator(BaseModel):
    """Text validation with sanitization."""
    
    text: str
    max_length: int = Field(default=10000)
    allow_html: bool = Field(default=False)
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and sanitize text."""
        if not v:
            return v
        
        # Remove null bytes
        v = v.replace("\x00", "")
        
        # Remove control characters except newlines and tabs
        v = "".join(char for char in v if char.isprintable() or char in ["\n", "\t"])
        
        return v
    
    @model_validator(mode="after")
    def check_length(self) -> "TextValidator":
        """Check text length."""
        if len(self.text) > self.max_length:
            raise ValueError(f"Text too long (max {self.max_length} characters)")
        return self


class UUIDValidator(BaseModel):
    """UUID validation."""
    
    uuid: UUID
    
    @field_validator("uuid")
    @classmethod
    def validate_uuid(cls, v: UUID) -> UUID:
        """Validate UUID format."""
        # UUID is already validated by Pydantic
        return v


class JSONValidator(BaseModel):
    """JSON validation with size limits."""
    
    data: Dict[str, Any]
    max_depth: int = Field(default=10)
    max_keys: int = Field(default=1000)
    
    @field_validator("data")
    @classmethod
    def validate_json(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON structure."""
        if not isinstance(v, dict):
            raise ValueError("Data must be a dictionary")
        
        return v
    
    @model_validator(mode="after")
    def check_complexity(self) -> "JSONValidator":
        """Check JSON complexity."""
        # Check depth
        def get_depth(obj: Any, current_depth: int = 0) -> int:
            if current_depth > self.max_depth:
                raise ValueError(f"JSON too deep (max {self.max_depth} levels)")
            
            if isinstance(obj, dict):
                if not obj:
                    return current_depth
                return max(get_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                if not obj:
                    return current_depth
                return max(get_depth(item, current_depth + 1) for item in obj)
            else:
                return current_depth
        
        get_depth(self.data)
        
        # Check number of keys
        def count_keys(obj: Any) -> int:
            if isinstance(obj, dict):
                count = len(obj)
                for v in obj.values():
                    count += count_keys(v)
                return count
            elif isinstance(obj, list):
                return sum(count_keys(item) for item in obj)
            else:
                return 0
        
        total_keys = count_keys(self.data)
        if total_keys > self.max_keys:
            raise ValueError(f"Too many keys in JSON (max {self.max_keys})")
        
        return self


def sanitize_sql_input(value: str) -> str:
    """Sanitize input to prevent SQL injection.
    
    Note: This is a defense-in-depth measure. Always use parameterized queries!
    
    Args:
        value: Input value
        
    Returns:
        Sanitized value
    """
    if not value:
        return value
    
    # Remove SQL comment markers
    value = value.replace("--", "").replace("/*", "").replace("*/", "")
    
    # Remove common SQL injection patterns
    dangerous_patterns = [
        r";\s*DROP\s+TABLE",
        r";\s*DELETE\s+FROM",
        r";\s*UPDATE\s+",
        r";\s*INSERT\s+INTO",
        r"UNION\s+SELECT",
        r"OR\s+1\s*=\s*1",
        r"OR\s+'1'\s*=\s*'1'",
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValidationError(
                "Potentially dangerous input detected",
                details={"pattern": pattern}
            )
    
    return value


def sanitize_path_input(path: str) -> str:
    """Sanitize file path to prevent path traversal attacks.
    
    Args:
        path: File path
        
    Returns:
        Sanitized path
        
    Raises:
        ValidationError: If path contains dangerous patterns
    """
    if not path:
        return path
    
    # Check for path traversal patterns
    if ".." in path or path.startswith("/") or path.startswith("\\"):
        raise ValidationError(
            "Invalid path: path traversal detected",
            details={"path": path}
        )
    
    # Remove null bytes
    path = path.replace("\x00", "")
    
    # Only allow alphanumeric, dash, underscore, dot, and forward slash
    if not re.match(r'^[a-zA-Z0-9_\-./]+$', path):
        raise ValidationError(
            "Invalid path: contains illegal characters",
            details={"path": path}
        )
    
    return path


def validate_pagination(page: int, page_size: int, max_page_size: int = 100) -> tuple[int, int]:
    """Validate pagination parameters.
    
    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Validated (page, page_size) tuple
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if page < 1:
        raise ValidationError("Page must be >= 1", details={"page": page})
    
    if page_size < 1:
        raise ValidationError("Page size must be >= 1", details={"page_size": page_size})
    
    if page_size > max_page_size:
        raise ValidationError(
            f"Page size too large (max {max_page_size})",
            details={"page_size": page_size, "max": max_page_size}
        )
    
    return page, page_size


def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Validate date range parameters.
    
    Args:
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        
    Returns:
        Validated (start_date, end_date) tuple
        
    Raises:
        ValidationError: If dates are invalid
    """
    from datetime import datetime
    
    if start_date:
        try:
            start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValidationError(f"Invalid start_date format: {e}")
    
    if end_date:
        try:
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValidationError(f"Invalid end_date format: {e}")
    
    if start_date and end_date:
        if start > end:
            raise ValidationError("start_date must be before end_date")
    
    return start_date, end_date

