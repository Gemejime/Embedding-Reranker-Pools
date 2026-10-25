"""Chaos Test: Native Queue Saturation, Backpressure & 3-State Circuit Breaker Recovery.

Validates:
1. Native queue saturation where Triton returns RESOURCE_EXHAUSTED (gRPC status 8).
2. Orchestrator-side circuit breaker trips and returns deterministic un-reranked RRF scores:
   RRF(d) = 1 / (60 + rank(d)).
3. Full 3-state transition lifecycle under recovery:
   CLOSED -> OPEN (fail-fast fallback) -> HALF_OPEN (single probe trial) -> CLOSED (traffic resumption).
4. Secured logging under failure: Zero customer query text or passage chunk payloads emitted to logs.
"""

import io
import logging
import time
import unittest
from concurrent import futures

import grpc

from api.proto.v1.triton_stubs_pb2 import (
    InferOutputTensor,
    ModelInferRequest,
    ModelInferResponse,
    ServerLiveRequest,
    ServerLiveResponse,
    ServerReadyRequest,
    ServerReadyResponse,
)
from api.proto.v1.triton_stubs_pb2_grpc import (
    GRPCInferenceServiceServicer,
    add_GRPCInferenceServiceServicer_to_server,
)
from internal.client.circuit_breaker import CircuitState, RerankerCircuitBreaker
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import pack_fp16_tensor, unpack_string_tensor
from internal.client.models import CandidateDocument
from internal.client.search_client import SearchOrchestratorClient
from internal.logging.secured_logger import SecuredJSONFormatter


class SaturatedQueueTritonServicer(GRPCInferenceServiceServicer):
    """Mock Triton servicer capable of dynamic queue saturation injection."""

    def __init__(self) -> None:
        self.queue_saturated = False
        self.infer_calls = 0

    def ServerLive(
        self, request: ServerLiveRequest, context: grpc.ServicerContext
    ) -> ServerLiveResponse:
        return ServerLiveResponse(live=True)

    def ServerReady(
        self, request: ServerReadyRequest, context: grpc.ServicerContext
    ) -> ServerReadyResponse:
        return ServerReadyResponse(ready=True)

    def ModelInfer(
        self, request: ModelInferRequest, context: grpc.ServicerContext
    ) -> ModelInferResponse:
        self.infer_calls += 1
        if self.queue_saturated:
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Triton queue full: max_queue_delay exceeded; shed load",
            )
            return ModelInferResponse()

        passages = (
            unpack_string_tensor(request.raw_input_contents[1])
            if len(request.raw_input_contents) > 1
            else []
        )
        count = len(passages) if passages else 1
        logits = [[float(i + 1)] for i in range(count)]
        raw_out = pack_fp16_tensor(logits)

        return ModelInferResponse(
            model_name=request.model_name,
            outputs=[InferOutputTensor(name="logits", datatype="FP16", shape=[count, 1])],
            raw_output_contents=[raw_out],
        )


class TestQueueSaturationBackpressure(unittest.TestCase):
    server: grpc.Server
    servicer: SaturatedQueueTritonServicer
    server_target: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        cls.servicer = SaturatedQueueTritonServicer()
        add_GRPCInferenceServiceServicer_to_server(cls.servicer, cls.server)
        port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.server_target = f"127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop(grace=0)

    def setUp(self) -> None:
        self.servicer.queue_saturated = False
        self.servicer.infer_calls = 0

        self.pool = TritonGRPCChannelPool(reranker_target=self.server_target)
        self.breaker = RerankerCircuitBreaker(
            failure_threshold=2,
            recovery_timeout_seconds=0.15,  # Short cooldown for deterministic test
            half_open_max_probes=1,
        )
        self.client = SearchOrchestratorClient(
            pool=self.pool,
            circuit_breaker=self.breaker,
            rerank_timeout_seconds=1.0,
        )

        # Secured logging capture
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(SecuredJSONFormatter())
        self.logger = logging.getLogger("search-client")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.handler)
        self.client.close()
        self.pool.close()

    def test_queue_saturation_3_state_lifecycle_and_secured_logging(self) -> None:
        """Validate queue saturation, RRF fallback, 3-state transition, and zero payload leakage in logs."""
        candidates = [
            CandidateDocument(
                id="doc-alpha", passage="sensitive customer query alpha text", initial_rank=1
            ),
            CandidateDocument(
                id="doc-beta",
                passage="sensitive enterprise confidential beta passage",
                initial_rank=2,
            ),
        ]
        query_text = "customer private proprietary query"

        # 1. State: CLOSED and queue healthy
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        res_healthy = self.client.rerank(query_text, candidates)
        self.assertEqual(len(res_healthy), 2)
        self.assertFalse(res_healthy[0].fallback_applied)
        self.assertEqual(self.servicer.infer_calls, 1)

        # 2. Inject queue saturation chaos
        self.servicer.queue_saturated = True

        # Call 1: Triton sheds load via RESOURCE_EXHAUSTED -> immediate un-reranked RRF fallback
        res_sat1 = self.client.rerank(query_text, candidates)
        self.assertEqual(len(res_sat1), 2)
        self.assertTrue(res_sat1[0].fallback_applied)
        self.assertAlmostEqual(res_sat1[0].score, 1.0 / (60.0 + 1.0), places=6)
        self.assertAlmostEqual(res_sat1[1].score, 1.0 / (60.0 + 2.0), places=6)
        self.assertEqual(
            self.breaker.state, CircuitState.CLOSED
        )  # Failure count = 1 (< threshold 2)

        # Call 2: Second failure reaches failure_threshold (2) -> trips breaker to OPEN
        res_sat2 = self.client.rerank(query_text, candidates)
        self.assertTrue(res_sat2[0].fallback_applied)
        self.assertEqual(self.breaker.state, CircuitState.OPEN)

        # 3. State: OPEN -> Fast failover without issuing gRPC calls
        calls_before = self.servicer.infer_calls
        res_open = self.client.rerank(query_text, candidates)
        self.assertTrue(res_open[0].fallback_applied)
        self.assertEqual(res_open[0].fallback_reason, "circuit_breaker_open")
        self.assertEqual(self.servicer.infer_calls, calls_before)  # Zero gRPC calls made

        # 4. State: Recovery after cooldown -> transitions to HALF_OPEN
        time.sleep(0.2)  # Cooldown > recovery_timeout_seconds (0.15s)
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Heal server before probe
        self.servicer.queue_saturated = False

        # 5. Probe trial executes in HALF_OPEN: Probe succeeds -> transitions to CLOSED
        res_recovered = self.client.rerank(query_text, candidates)
        self.assertFalse(res_recovered[0].fallback_applied)
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

        # 6. Verify Secured Logging Invariant: zero query text or passage text logged
        log_contents = self.log_stream.getvalue()
        self.assertNotIn("sensitive customer query alpha text", log_contents)
        self.assertNotIn("sensitive enterprise confidential beta passage", log_contents)
        self.assertNotIn("customer private proprietary query", log_contents)
        self.assertIn("circuit_open_fallback", log_contents)


if __name__ == "__main__":
    unittest.main()
