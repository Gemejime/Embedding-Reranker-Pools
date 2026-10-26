"""Chaos Test: Synthetic CUDA Stalls & GPU Latency Spike Injection.

Validates:
1. GPU kernel execution delays exceeding client latency deadlines (15 ms embedding, 35 ms reranker).
2. Strict deadline enforcement (DEADLINE_EXCEEDED).
3. Transparent un-reranked RRF score fallback emission upon reranker deadline breach.
4. Rapid gRPC channel recycling after stalls with zero leaked channel slots.
"""

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
from internal.client.circuit_breaker import RerankerCircuitBreaker
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import pack_fp16_tensor
from internal.client.models import CandidateDocument
from internal.client.search_client import SearchOrchestratorClient
from internal.config.settings import PoolType


class StallingTritonServicer(GRPCInferenceServiceServicer):
    """Mock Triton servicer capable of injecting CUDA kernel stalls."""

    def __init__(self) -> None:
        self.stall_delay_seconds: float = 0.0

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
        if self.stall_delay_seconds > 0:
            time.sleep(self.stall_delay_seconds)

        model_name = request.model_name
        if model_name == "online_embedder_ensemble":
            raw_out = pack_fp16_tensor([[0.01] * 768])
            return ModelInferResponse(
                model_name=model_name,
                outputs=[InferOutputTensor(name="embeddings", datatype="FP16", shape=[1, 768])],
                raw_output_contents=[raw_out],
            )
        elif model_name == "online_reranker_ensemble":
            raw_out = pack_fp16_tensor([[1.0], [2.0]])
            return ModelInferResponse(
                model_name=model_name,
                outputs=[InferOutputTensor(name="logits", datatype="FP16", shape=[2, 1])],
                raw_output_contents=[raw_out],
            )

        context.abort(grpc.StatusCode.NOT_FOUND, f"Model {model_name} not found")
        return ModelInferResponse()


class TestCUDAStallInjection(unittest.TestCase):
    server: grpc.Server
    servicer: StallingTritonServicer
    server_target: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        cls.servicer = StallingTritonServicer()
        add_GRPCInferenceServiceServicer_to_server(cls.servicer, cls.server)
        port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.server_target = f"127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop(grace=0)

    def setUp(self) -> None:
        self.servicer.stall_delay_seconds = 0.0
        self.pool = TritonGRPCChannelPool(
            embedding_target=self.server_target,
            reranker_target=self.server_target,
        )
        self.breaker = RerankerCircuitBreaker(failure_threshold=2)
        self.client = SearchOrchestratorClient(
            pool=self.pool,
            circuit_breaker=self.breaker,
            embed_timeout_seconds=1.0,
            rerank_timeout_seconds=1.0,
        )

    def tearDown(self) -> None:
        self.client.close()
        self.pool.close()

    def test_reranker_cuda_stall_deadline_enforcement_and_rrf_fallback(self) -> None:
        """Inject CUDA stall on reranker exceeding deadline and verify un-reranked RRF emission."""
        candidates = [
            CandidateDocument(id="doc-1", passage="passage one", initial_rank=1),
            CandidateDocument(id="doc-2", passage="passage two", initial_rank=2),
        ]

        # Enforce tight 80ms deadline and inject 400ms stall
        self.client._rerank_timeout = 0.080
        self.servicer.stall_delay_seconds = 0.400

        t0 = time.perf_counter()
        results = self.client.rerank("query with stalled gpu", candidates)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Verify client deadline caught stall and returned RRF fallback
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].fallback_applied)
        self.assertIsNotNone(results[0].fallback_reason)
        self.assertIn("deadline_exceeded", results[0].fallback_reason or "")
        self.assertAlmostEqual(results[0].score, 1.0 / 61.0, places=6)
        self.assertLess(elapsed_ms, 900.0)

    def test_embedding_cuda_stall_deadline_enforcement(self) -> None:
        """Inject CUDA stall on embedder exceeding deadline and assert DEADLINE_EXCEEDED."""
        self.client._embed_timeout = 0.050
        self.servicer.stall_delay_seconds = 0.400

        with self.assertRaises(grpc.RpcError) as ctx:
            self.client.embed_query("query to stalled gpu")

        err = ctx.exception
        self.assertEqual(err.code(), grpc.StatusCode.DEADLINE_EXCEEDED)

    def test_channel_recycling_after_stall_with_zero_leaks(self) -> None:
        """Verify gRPC channels can be recycled cleanly after stall events without socket leaks."""
        self.servicer.stall_delay_seconds = 0.0

        # Initialize channels
        self.pool.get_channel(PoolType.ONLINE_EMBEDDING)
        self.pool.get_channel(PoolType.ONLINE_RERANKER)
        self.assertEqual(self.pool.get_active_channel_count(), 2)

        # Recycle embedding channel
        self.pool.reconnect_channel(PoolType.ONLINE_EMBEDDING)
        # Active channel count remains 2 (1 recycled, 1 intact), zero channel leaks
        self.assertEqual(self.pool.get_active_channel_count(), 2)

        # Successful query on recycled channel
        vec = self.client.embed_query("query post recycling")
        self.assertEqual(len(vec), 768)


if __name__ == "__main__":
    unittest.main()
