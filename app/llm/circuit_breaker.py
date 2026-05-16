"""Circuit breaker pattern for LLM providers.

This module implements the circuit breaker pattern to prevent cascading failures:
- Automatic failure detection
- Circuit state management (CLOSED, OPEN, HALF_OPEN)
- Configurable thresholds
- Automatic recovery attempts
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional, TypeVar

from app.llm.exceptions import LLMCircuitBreakerError, LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for LLM providers.

    The circuit breaker prevents cascading failures by:
    1. Tracking failure rate
    2. Opening circuit after threshold failures
    3. Rejecting requests while open
    4. Attempting recovery after timeout
    5. Closing circuit on successful recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker name (e.g., provider name)
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit from half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock_obj: Optional[asyncio.Lock] = None

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazily construct the asyncio lock so ``__init__`` is loop-agnostic.

        Constructing ``asyncio.Lock()`` at import / wiring time captures the
        current event loop on Python ≤ 3.9, which breaks when the caller is
        not inside a running loop (or a previous ``asyncio.run`` has cleared
        the global loop).  Deferring construction to first async use removes
        that coupling without changing any locking semantics.
        """
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self._state == CircuitState.CLOSED

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function

        Raises:
            LLMCircuitBreakerError: If circuit is open
            Exception: Any exception from func
        """
        async with self._lock:
            # Check if we should attempt recovery
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    logger.info(f"Circuit breaker {self.name}: Attempting recovery (HALF_OPEN)")
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                else:
                    raise LLMCircuitBreakerError(
                        f"Circuit breaker {self.name} is OPEN",
                        failure_count=self._failure_count,
                        threshold=self.failure_threshold,
                        provider=self.name,
                    )

        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure(e)
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if we should attempt recovery.

        Returns:
            True if recovery timeout has elapsed
        """
        if self._last_failure_time is None:
            return True

        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.recovery_timeout

    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.info(
                    f"Circuit breaker {self.name}: Success {self._success_count}/{self.success_threshold}"
                )

                if self._success_count >= self.success_threshold:
                    logger.info(f"Circuit breaker {self.name}: Recovery successful (CLOSED)")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_failure_time = None

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = 0

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed execution.

        Args:
            error: Exception that occurred
        """
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit breaker {self.name}: Recovery failed, reopening circuit (OPEN)"
                )
                self._state = CircuitState.OPEN
                self._success_count = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker {self.name}: Threshold reached "
                        f"({self._failure_count}/{self.failure_threshold}), opening circuit (OPEN)"
                    )
                    self._state = CircuitState.OPEN

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            logger.info(f"Circuit breaker {self.name}: Manual reset (CLOSED)")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self._last_failure_time,
        }

