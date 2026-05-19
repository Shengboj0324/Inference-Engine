"""Runtime orchestrator that turns observations into a ``SituationReport``.

Phase 3 of the migration documented in ``docs/intelligence_migration.md``.
The engine wraps three stages around a single LLM call:

1. **PII scrub.**  Every ``Observation.text`` is passed through
   :meth:`DataResidencyGuard.scrub_text` before it reaches the generator,
   so neither the local fine-tuned Llama nor a routed frontier model ever
   sees raw user PII.
2. **Generate.**  The scrubbed observations are wrapped into the frozen
   two-message conversation produced by ``build_inference_messages`` and
   submitted to a ``GenerateFn`` (LoRA, router, or test mock). The raw
   completion is parsed and schema-validated by ``parse_model_output``.
3. **Verify grounding.**  Citation spans in the parsed report are checked
   against the *scrubbed* observation texts by :class:`CitationVerifier`.
   A semantic LLM-as-judge check is *not* run here; that is the job of
   the offline evaluation in ``app/evals/scenario_eval.py``.

The orchestrator deliberately contains no keyword lists, signal-detector
hooks, or report templates. All domain reasoning lives in the trained
model; this module's responsibility is plumbing and structural safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

from app.core.data_residency import DataResidencyGuard
from app.evals.scenario_loader import Observation
from app.intelligence.citation_verifier import (
    CitationGroundingError,
    CitationVerifier,
)
from app.intelligence.situation_report import SituationReport
from app.llm.training.inferencer import GenerateFn
from app.llm.training.situation_prompt import (
    build_inference_messages,
    load_system_prompt,
    parse_model_output,
)

logger = logging.getLogger(__name__)

AsyncGenerateFn = Callable[[List[Dict[str, str]]], Awaitable[str]]
"""Async counterpart of :data:`GenerateFn`; returns the awaited completion."""

__all__ = [
    "AsyncGenerateFn",
    "AsyncSituationEngine",
    "CitationGroundingError",
    "EngineResult",
    "GenerationError",
    "OutputParseError",
    "SituationEngine",
    "SituationEngineError",
    "build_router_generator",
]


class SituationEngineError(RuntimeError):
    """Base class for non-recoverable engine failures."""


class GenerationError(SituationEngineError):
    """The underlying generator (LoRA model or router) raised."""


class OutputParseError(SituationEngineError):
    """The model returned text that does not parse into a ``SituationReport``."""


@dataclass(frozen=True)
class EngineResult:
    """Successful engine output bundled with provenance metadata.

    ``raw_completion`` is retained so operators can audit exactly what the
    model emitted, and ``redaction_count`` exposes how many PII tokens the
    pre-flight scrubber removed before the call. When the engine escalated
    to its fallback generator, ``used_fallback`` is ``True`` and
    ``primary_report`` holds the discarded lower-confidence first attempt
    (``None`` when the primary call also failed to parse).
    """

    report: SituationReport
    raw_completion: str
    redaction_count: int
    used_fallback: bool = False
    primary_report: Optional[SituationReport] = None


class SituationEngine:
    """Compose PII scrubbing, model generation, and grounding verification.

    The engine is constructed once with a generator callable and reused
    across calls. It carries no per-request state, so a single instance is
    safe to share across async tasks provided the underlying generator is.

    When ``fallback_generate`` is provided, the engine optionally escalates
    a single retry to a stronger generator (typically a frontier model) if
    the primary completion either fails to parse / verify or returns a
    report whose ``calibrated_confidence`` is below ``min_confidence``.
    """

    def __init__(
        self,
        generate: GenerateFn,
        *,
        system_prompt: Optional[str] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        fallback_generate: Optional[GenerateFn] = None,
        min_confidence: float = 0.0,
    ) -> None:
        _validate_min_confidence(min_confidence)
        self._generate = generate
        self._system_prompt = (
            system_prompt if system_prompt is not None else load_system_prompt()
        )
        self._verifier = citation_verifier or CitationVerifier()
        self._fallback_generate = fallback_generate
        self._min_confidence = min_confidence

    def analyze(self, observations: Sequence[Observation]) -> EngineResult:
        """Produce a verified ``SituationReport`` for ``observations``.

        Args:
            observations: Source observations as they arrive from upstream
                connectors. Must be non-empty; an empty list is a caller
                bug, not an abstention signal.

        Returns:
            :class:`EngineResult` containing the validated report, the raw
            model completion, and the redaction count.

        Raises:
            ValueError: If ``observations`` is empty.
            GenerationError: If the generator itself raises and no
                fallback is available (or the fallback also raises).
            OutputParseError: If the completion is not a valid report and
                the fallback (if any) also fails to parse.
            CitationGroundingError: If any citation fails structural checks
                and the fallback (if any) also fails grounding.
        """
        scrubbed, messages, redaction_count = _prepare(
            observations, self._system_prompt
        )
        primary_attempt = _attempt_sync(
            self._generate, messages, scrubbed, self._verifier
        )
        if not self._should_fallback(primary_attempt):
            return _attempt_to_result(primary_attempt, redaction_count)

        logger.info(
            "situation_engine.fallback_triggered",
            extra={"reason": _fallback_reason(primary_attempt)},
        )
        fallback_attempt = _attempt_sync(
            self._fallback_generate, messages, scrubbed, self._verifier
        )
        return _merge_attempts(primary_attempt, fallback_attempt, redaction_count)

    def _should_fallback(self, attempt: "_Attempt") -> bool:
        if self._fallback_generate is None:
            return False
        if attempt.error is not None:
            return True
        return attempt.report.calibrated_confidence < self._min_confidence


class AsyncSituationEngine:
    """Async counterpart of :class:`SituationEngine` for router-backed runtimes.

    Identical pipeline, but the generator is awaited rather than called
    synchronously. Use this from FastAPI / async workers when the underlying
    LLM transport is asynchronous (e.g., :class:`LLMRouter`). The optional
    ``fallback_generate`` and ``min_confidence`` parameters behave exactly
    as on :class:`SituationEngine`.
    """

    def __init__(
        self,
        generate: AsyncGenerateFn,
        *,
        system_prompt: Optional[str] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        fallback_generate: Optional[AsyncGenerateFn] = None,
        min_confidence: float = 0.0,
    ) -> None:
        _validate_min_confidence(min_confidence)
        self._generate = generate
        self._system_prompt = (
            system_prompt if system_prompt is not None else load_system_prompt()
        )
        self._verifier = citation_verifier or CitationVerifier()
        self._fallback_generate = fallback_generate
        self._min_confidence = min_confidence

    async def analyze(
        self, observations: Sequence[Observation]
    ) -> EngineResult:
        """Async analogue of :meth:`SituationEngine.analyze`."""
        scrubbed, messages, redaction_count = _prepare(
            observations, self._system_prompt
        )
        primary_attempt = await _attempt_async(
            self._generate, messages, scrubbed, self._verifier
        )
        if not self._should_fallback(primary_attempt):
            return _attempt_to_result(primary_attempt, redaction_count)

        logger.info(
            "situation_engine.fallback_triggered",
            extra={"reason": _fallback_reason(primary_attempt)},
        )
        fallback_attempt = await _attempt_async(
            self._fallback_generate, messages, scrubbed, self._verifier
        )
        return _merge_attempts(primary_attempt, fallback_attempt, redaction_count)

    def _should_fallback(self, attempt: "_Attempt") -> bool:
        if self._fallback_generate is None:
            return False
        if attempt.error is not None:
            return True
        return attempt.report.calibrated_confidence < self._min_confidence


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------


def _prepare(
    observations: Sequence[Observation],
    system_prompt: str,
) -> tuple[List[Observation], List[Dict[str, str]], int]:
    if not observations:
        raise ValueError("analyze() requires at least one observation")
    scrubbed, redaction_count = _scrub_observations(observations)
    messages = build_inference_messages(scrubbed, system_prompt=system_prompt)
    return scrubbed, messages, redaction_count


@dataclass(frozen=True)
class _Attempt:
    """Outcome of a single generator pass: either a parsed report or an error."""

    raw_completion: str
    report: Optional[SituationReport]
    error: Optional[SituationEngineError]


def _validate_min_confidence(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"min_confidence must be in [0.0, 1.0], got {value!r}"
        )


def _attempt_sync(
    generate: GenerateFn,
    messages: List[Dict[str, str]],
    scrubbed: Sequence[Observation],
    verifier: CitationVerifier,
) -> _Attempt:
    try:
        raw = generate(messages)
    except Exception as exc:
        return _Attempt(
            raw_completion="",
            report=None,
            error=GenerationError(
                f"generator failed: {type(exc).__name__}: {exc}"
            ),
        )
    return _parse_and_verify(raw, scrubbed, verifier)


async def _attempt_async(
    generate: AsyncGenerateFn,
    messages: List[Dict[str, str]],
    scrubbed: Sequence[Observation],
    verifier: CitationVerifier,
) -> _Attempt:
    try:
        raw = await generate(messages)
    except Exception as exc:
        return _Attempt(
            raw_completion="",
            report=None,
            error=GenerationError(
                f"generator failed: {type(exc).__name__}: {exc}"
            ),
        )
    return _parse_and_verify(raw, scrubbed, verifier)


def _parse_and_verify(
    raw: str,
    scrubbed: Sequence[Observation],
    verifier: CitationVerifier,
) -> _Attempt:
    try:
        report = parse_model_output(raw)
    except Exception as exc:
        return _Attempt(
            raw_completion=raw,
            report=None,
            error=OutputParseError(
                f"model output rejected: {type(exc).__name__}: {exc}"
            ),
        )
    try:
        verifier.verify_report(report, scrubbed)
    except CitationGroundingError as exc:
        return _Attempt(raw_completion=raw, report=report, error=exc)
    return _Attempt(raw_completion=raw, report=report, error=None)


def _attempt_to_result(attempt: _Attempt, redaction_count: int) -> EngineResult:
    if attempt.error is not None:
        raise attempt.error
    assert attempt.report is not None
    return EngineResult(
        report=attempt.report,
        raw_completion=attempt.raw_completion,
        redaction_count=redaction_count,
    )


def _merge_attempts(
    primary: _Attempt,
    fallback: _Attempt,
    redaction_count: int,
) -> EngineResult:
    if fallback.error is not None:
        if primary.error is None and primary.report is not None:
            logger.warning(
                "situation_engine.fallback_failed_keeping_primary",
                extra={"fallback_error": type(fallback.error).__name__},
            )
            return EngineResult(
                report=primary.report,
                raw_completion=primary.raw_completion,
                redaction_count=redaction_count,
            )
        raise fallback.error
    assert fallback.report is not None
    return EngineResult(
        report=fallback.report,
        raw_completion=fallback.raw_completion,
        redaction_count=redaction_count,
        used_fallback=True,
        primary_report=primary.report,
    )


def _fallback_reason(attempt: _Attempt) -> str:
    if attempt.error is not None:
        return f"primary_error:{type(attempt.error).__name__}"
    assert attempt.report is not None
    return f"low_confidence:{attempt.report.calibrated_confidence:.3f}"


def _scrub_observations(
    observations: Sequence[Observation],
) -> tuple[List[Observation], int]:
    scrubbed: List[Observation] = []
    total = 0
    for obs in observations:
        clean_text, n = DataResidencyGuard.scrub_text(obs.text)
        total += n
        if n == 0 and clean_text == obs.text:
            scrubbed.append(obs)
        else:
            scrubbed.append(obs.model_copy(update={"text": clean_text}))
    if total:
        logger.info(
            "situation_engine.pii_redacted",
            extra={
                "redaction_count": total,
                "n_observations": len(observations),
            },
        )
    return scrubbed, total


# ---------------------------------------------------------------------------
# Production generator factory
# ---------------------------------------------------------------------------


def build_router_generator(
    router: "object",
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = 2048,
) -> AsyncGenerateFn:
    """Wrap an :class:`LLMRouter` instance into an :data:`AsyncGenerateFn`.

    The router is imported lazily so callers that only need the LoRA path
    (offline inference, tests) do not pay the import cost. ``model`` is
    forwarded as a routing-strategy hint; when ``None``, the router selects
    a model via its configured strategy.
    """
    from app.llm.models import LLMMessage, MessageRole
    from app.llm.router import RoutingStrategy

    async def _generate(messages: List[Dict[str, str]]) -> str:
        llm_messages = [
            LLMMessage(role=MessageRole(m["role"]), content=m["content"])
            for m in messages
        ]
        kwargs: Dict[str, object] = {}
        if model is not None:
            kwargs["model"] = model
        response = await router.generate(  # type: ignore[attr-defined]
            messages=llm_messages,
            strategy=RoutingStrategy.BALANCED,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content

    return _generate
