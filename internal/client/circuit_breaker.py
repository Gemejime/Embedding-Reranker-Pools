"""Resilient Circuit Breaker and Un-Reranked RRF Fallback Engine.

Guarantees:
1. Native circuit breaking across Triton queue saturation (RESOURCE_EXHAUSTED),
   latency breaches (DEADLINE_EXCEEDED), and pod restarts (UNAVAILABLE).
2. Deterministic un-reranked Reciprocal Rank Fusion (RRF) fallback:
   RRF(d) = 1 / (60 + rank(d))
3. Thread-safe state machine transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
"""

import threading
import time
from collections.abc import Sequence
from enum import StrEnum

import grpc

from internal.client.models import CandidateDocument, RerankResult
from internal.logging.secured_logger import get_secured_logger

logger = get_secured_logger("circuit-breaker")

RRF_K_CONSTANT: float = 60.0

TRIPPING_STATUS_CODES = frozenset(
    {
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAVAILABLE,
    }
)


class CircuitState(StrEnum):
    """Circuit breaker operational state machine."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RerankerCircuitBreaker:
    """Thread-safe circuit breaker protecting the online reranker pool."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 5.0,
        half_open_max_probes: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.half_open_max_probes = half_open_max_probes

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_timestamp: float = 0.0
        self._half_open_probes = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, evaluating recovery timeout if OPEN."""
        with self._lock:
            self._check_recovery_timeout_locked()
            return self._state

    def _check_recovery_timeout_locked(self) -> None:
        """Internal helper to transition OPEN to HALF_OPEN after cooldown."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_timestamp
            if elapsed >= self.recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_probes = 0
                logger.info(
                    "Circuit breaker transitioning to HALF_OPEN after cooldown",
                    extra={
                        "status": "half_open_probe_allowed",
                        "elapsed_seconds": round(elapsed, 2),
                    },
                )

    def can_execute(self) -> bool:
        """Determine whether an execution attempt is permitted."""
        with self._lock:
            self._check_recovery_timeout_locked()
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_probes < self.half_open_max_probes:
                    self._half_open_probes += 1
                    return True
                return False
            else:
                return False

    def is_tripping_error(self, error: Exception) -> bool:
        """Determine if an exception represents a tripping condition."""
        if isinstance(error, grpc.RpcError):
            code_func = getattr(error, "code", None)
            if callable(code_func):
                code = code_func()
                return code in TRIPPING_STATUS_CODES
        # Standard timeout or connection exceptions also trip
        error_name = type(error).__name__.lower()
        return "timeout" in error_name or "connection" in error_name

    def record_success(self) -> None:
        """Record a successful invocation, resetting failure metrics and closing circuit."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    "Probe successful, circuit breaker closing",
                    extra={"status": "circuit_closed"},
                )
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._half_open_probes = 0

    def release_probe(self) -> None:
        """Release a consumed half-open probe slot without altering circuit state."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN and self._half_open_probes > 0:
                self._half_open_probes -= 1
                logger.info(
                    "Non-tripping error in HALF_OPEN state; probe slot released",
                    extra={
                        "status": "half_open_probe_released",
                        "remaining_probes": self._half_open_probes,
                    },
                )

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed invocation, potentially tripping the circuit to OPEN."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_timestamp = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed; immediately return to OPEN
                self._state = CircuitState.OPEN
                logger.warning(
                    "Probe failed, circuit breaker returning to OPEN",
                    extra={
                        "status": "circuit_reopened",
                        "consecutive_failures": self._consecutive_failures,
                    },
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    "Failure threshold reached, circuit breaker tripped to OPEN",
                    extra={
                        "status": "circuit_opened",
                        "consecutive_failures": self._consecutive_failures,
                        "failure_threshold": self.failure_threshold,
                    },
                )

    def compute_rrf_fallback(
        self,
        candidates: Sequence[CandidateDocument],
        reason: str = "circuit_open",
    ) -> list[RerankResult]:
        """Compute deterministic un-reranked RRF scores for candidate documents.

        Formula: RRF(d) = 1.0 / (60.0 + initial_rank(d))
        Preserves candidate rankings and marks results with fallback_applied=True.
        """
        if not candidates:
            return []

        # Sort candidates strictly by initial rank ascending
        sorted_candidates = sorted(candidates, key=lambda c: c.initial_rank)
        results: list[RerankResult] = []

        for rank_idx, doc in enumerate(sorted_candidates, start=1):
            # Deterministic RRF calculation
            score = 1.0 / (RRF_K_CONSTANT + float(doc.initial_rank))
            results.append(
                RerankResult(
                    id=doc.id,
                    score=score,
                    rank=rank_idx,
                    fallback_applied=True,
                    fallback_reason=reason,
                )
            )

        return results
