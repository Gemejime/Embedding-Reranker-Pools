"""Unit tests for RerankerCircuitBreaker state machine and RRF fallback."""

import unittest
from unittest.mock import patch

import grpc

from internal.client.circuit_breaker import (
    CircuitState,
    RerankerCircuitBreaker,
)
from internal.client.models import CandidateDocument


class DummyRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self) -> None:
        self.breaker = RerankerCircuitBreaker(
            failure_threshold=3,
            recovery_timeout_seconds=0.1,  # Fast 100ms timeout for testing
            half_open_max_probes=1,
        )

    def test_initial_state_closed(self) -> None:
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.breaker.can_execute())

    def test_successful_calls_preserve_closed_state(self) -> None:
        self.breaker.record_success()
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.breaker.can_execute())

    def test_failure_counter_and_trip_to_open(self) -> None:
        # 1st failure
        self.breaker.record_failure(DummyRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED))
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.breaker.can_execute())

        # 2nd failure
        self.breaker.record_failure(DummyRpcError(grpc.StatusCode.DEADLINE_EXCEEDED))
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.breaker.can_execute())

        # 3rd failure (hits threshold = 3)
        self.breaker.record_failure(DummyRpcError(grpc.StatusCode.UNAVAILABLE))
        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        self.assertFalse(self.breaker.can_execute())

    def test_recovery_timeout_to_half_open(self) -> None:
        # Trip to OPEN
        for _ in range(3):
            self.breaker.record_failure(DummyRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED))
        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        self.assertFalse(self.breaker.can_execute())

        # Mock time elapsed past recovery timeout
        with patch("time.monotonic") as mock_time:
            # Shift time forward by 10 seconds
            mock_time.return_value = self.breaker._last_failure_timestamp + 10.0

            self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)
            # First probe allowed
            self.assertTrue(self.breaker.can_execute())
            # Second probe blocked (max_probes = 1)
            self.assertFalse(self.breaker.can_execute())

    def test_half_open_probe_success_closes_circuit(self) -> None:
        # Trip to OPEN
        for _ in range(3):
            self.breaker.record_failure()

        # Simulate timeout to HALF_OPEN
        self.breaker._last_failure_timestamp -= 1.0  # past 0.1s
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Record success
        self.breaker.record_success()
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertTrue(self.breaker.can_execute())

    def test_half_open_probe_failure_reopens_circuit(self) -> None:
        for _ in range(3):
            self.breaker.record_failure()

        self.breaker._last_failure_timestamp -= 1.0
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Failed probe
        self.breaker.record_failure(DummyRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED))
        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        self.assertFalse(self.breaker.can_execute())

    def test_is_tripping_error(self) -> None:
        self.assertTrue(
            self.breaker.is_tripping_error(DummyRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED))
        )
        self.assertTrue(
            self.breaker.is_tripping_error(DummyRpcError(grpc.StatusCode.DEADLINE_EXCEEDED))
        )
        self.assertTrue(self.breaker.is_tripping_error(DummyRpcError(grpc.StatusCode.UNAVAILABLE)))
        self.assertFalse(
            self.breaker.is_tripping_error(DummyRpcError(grpc.StatusCode.INVALID_ARGUMENT))
        )
        self.assertTrue(self.breaker.is_tripping_error(TimeoutError("Request timed out")))

    def test_deterministic_rrf_fallback_scoring(self) -> None:
        candidates = [
            CandidateDocument(id="doc-2", passage="second passage", initial_rank=2),
            CandidateDocument(id="doc-1", passage="first passage", initial_rank=1),
            CandidateDocument(id="doc-3", passage="third passage", initial_rank=3),
        ]

        results = self.breaker.compute_rrf_fallback(candidates, reason="queue_saturated")

        self.assertEqual(len(results), 3)

        # Document 1: initial_rank = 1 -> score = 1.0 / (60.0 + 1) = 1/61
        self.assertEqual(results[0].id, "doc-1")
        self.assertEqual(results[0].rank, 1)
        self.assertAlmostEqual(results[0].score, 1.0 / 61.0, places=6)
        self.assertTrue(results[0].fallback_applied)
        self.assertEqual(results[0].fallback_reason, "queue_saturated")

        # Document 2: initial_rank = 2 -> score = 1.0 / (60.0 + 2) = 1/62
        self.assertEqual(results[1].id, "doc-2")
        self.assertEqual(results[1].rank, 2)
        self.assertAlmostEqual(results[1].score, 1.0 / 62.0, places=6)

        # Document 3: initial_rank = 3 -> score = 1.0 / (60.0 + 3) = 1/63
        self.assertEqual(results[2].id, "doc-3")
        self.assertEqual(results[2].rank, 3)
        self.assertAlmostEqual(results[2].score, 1.0 / 63.0, places=6)

    def test_rrf_fallback_empty_candidates(self) -> None:
        results = self.breaker.compute_rrf_fallback([])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
