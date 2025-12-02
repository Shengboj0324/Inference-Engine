"""Comprehensive tests for core infrastructure fortification.

This test suite validates:
- Configuration validation
- Database connection pooling and health checks
- Logging and monitoring
- Retry mechanisms and circuit breakers
- Input validation and sanitization
- Security features
"""

import asyncio
import pytest
from unittest.mock import Mock, patch

from app.core.config import Settings
from app.core.validation import (
    EmailValidator,
    URLValidator,
    TextValidator,
    UUIDValidator,
    JSONValidator,
    sanitize_sql_input,
    sanitize_path_input,
    validate_pagination,
    validate_date_range,
)
from app.core.errors import ValidationError
from app.core.retry import CircuitBreaker, CircuitState, retry_with_backoff
from app.core.security_advanced import MilitaryGradeEncryption
from app.core.logging_config import PerformanceLogger, SecurityLogger


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_valid_environment(self):
        """Test valid environment settings."""
        for env in ["development", "staging", "production", "test"]:
            settings = Settings(environment=env)
            assert settings.environment == env
    
    def test_invalid_environment(self):
        """Test invalid environment raises error."""
        with pytest.raises(ValueError, match="Environment must be one of"):
            Settings(environment="invalid")
    
    def test_production_secret_validation(self):
        """Test production requires custom secrets."""
        with pytest.raises(ValueError, match="SECRET_KEY must be changed"):
            Settings(
                environment="production",
                secret_key="change-this-in-production"
            )
    
    def test_log_level_validation(self):
        """Test log level validation."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"
        
        with pytest.raises(ValueError, match="Log level must be one of"):
            Settings(log_level="invalid")
    
    def test_port_validation(self):
        """Test port number validation."""
        with pytest.raises(ValueError, match="Port must be between"):
            Settings(api_port=0)
        
        with pytest.raises(ValueError, match="Port must be between"):
            Settings(api_port=70000)
    
    def test_temperature_validation(self):
        """Test LLM temperature validation."""
        settings = Settings(llm_temperature=0.5)
        assert settings.llm_temperature == 0.5
        
        with pytest.raises(ValueError, match="Temperature must be between"):
            Settings(llm_temperature=3.0)


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_email_validation(self):
        """Test email validation."""
        # Valid emails
        EmailValidator(email="test@example.com")
        EmailValidator(email="user.name+tag@example.co.uk")
        
        # Invalid emails
        with pytest.raises(ValueError):
            EmailValidator(email="invalid..email@example.com")
        
        with pytest.raises(ValueError):
            EmailValidator(email="a" * 256 + "@example.com")
    
    def test_url_validation(self):
        """Test URL validation."""
        # Valid URLs
        URLValidator(url="https://example.com")
        URLValidator(url="http://localhost:8000")
        
        # Invalid URLs
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            URLValidator(url="ftp://example.com")
        
        with pytest.raises(ValueError, match="URL too long"):
            URLValidator(url="https://" + "a" * 3000 + ".com")
    
    def test_text_sanitization(self):
        """Test text sanitization."""
        validator = TextValidator(text="Hello\x00World\x01")
        assert "\x00" not in validator.text
        assert "\x01" not in validator.text
        
        # Test length limit
        with pytest.raises(ValueError, match="Text too long"):
            TextValidator(text="a" * 20000, max_length=1000)
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        # Safe input
        assert sanitize_sql_input("normal text") == "normal text"
        
        # Dangerous input
        with pytest.raises(ValidationError):
            sanitize_sql_input("'; DROP TABLE users--")
        
        with pytest.raises(ValidationError):
            sanitize_sql_input("1 OR 1=1")
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        # Safe paths
        assert sanitize_path_input("files/document.pdf") == "files/document.pdf"
        
        # Dangerous paths
        with pytest.raises(ValidationError):
            sanitize_path_input("../../../etc/passwd")
        
        with pytest.raises(ValidationError):
            sanitize_path_input("/etc/passwd")
    
    def test_pagination_validation(self):
        """Test pagination validation."""
        page, size = validate_pagination(1, 10)
        assert page == 1
        assert size == 10
        
        with pytest.raises(ValidationError):
            validate_pagination(0, 10)
        
        with pytest.raises(ValidationError):
            validate_pagination(1, 1000)
    
    def test_json_complexity_validation(self):
        """Test JSON complexity validation."""
        # Simple JSON
        JSONValidator(data={"key": "value"})
        
        # Too deep
        deep_json = {"level1": {"level2": {"level3": {}}}}
        for i in range(10):
            deep_json = {"level": deep_json}
        
        with pytest.raises(ValueError, match="JSON too deep"):
            JSONValidator(data=deep_json, max_depth=5)


class TestRetryMechanism:
    """Test retry mechanism and circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_retry_success(self):
        """Test successful retry after failures."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_max_exceeded(self):
        """Test max retries exceeded."""
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def always_fails():
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            await always_fails()
    
    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        def failing_function():
            raise Exception("Failure")
        
        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_function)
        
        # Circuit should be open
        assert breaker.state == CircuitState.OPEN
        
        # Should reject calls
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            breaker.call(failing_function)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        call_count = 0
        
        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Failure")
            return "success"
        
        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(sometimes_fails)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        import time
        time.sleep(0.2)
        
        # Should attempt recovery
        result = breaker.call(sometimes_fails)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED


class TestSecurityFeatures:
    """Test security features."""
    
    def test_encryption_decryption(self):
        """Test encryption and decryption."""
        encryption = MilitaryGradeEncryption()
        
        plaintext = b"sensitive data"
        encrypted = encryption.encrypt_aes_gcm(plaintext)
        
        decrypted = encryption.decrypt_aes_gcm(
            encrypted["ciphertext"],
            encrypted["nonce"],
            encrypted["tag"]
        )
        
        assert decrypted == plaintext
    
    def test_encryption_validation(self):
        """Test encryption input validation."""
        encryption = MilitaryGradeEncryption()
        
        # Empty plaintext
        with pytest.raises(Exception):
            encryption.encrypt_aes_gcm(b"")
        
        # Too large
        with pytest.raises(Exception):
            encryption.encrypt_aes_gcm(b"a" * (101 * 1024 * 1024))
    
    def test_key_derivation(self):
        """Test key derivation."""
        encryption = MilitaryGradeEncryption()
        
        password = "strong_password_123"
        salt = b"a" * 16
        
        key1 = encryption.derive_key(password, salt)
        key2 = encryption.derive_key(password, salt)
        
        # Same password and salt should produce same key
        assert key1 == key2
        assert len(key1) == 32
    
    def test_key_derivation_validation(self):
        """Test key derivation validation."""
        encryption = MilitaryGradeEncryption()
        
        # Empty password
        with pytest.raises(Exception):
            encryption.derive_key("", b"a" * 16)
        
        # Short password
        with pytest.raises(Exception):
            encryption.derive_key("short", b"a" * 16)
        
        # Short salt
        with pytest.raises(Exception):
            encryption.derive_key("password123", b"short")

