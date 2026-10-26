"""Integration tests validating SearchOrchestratorClient and BatchIngestionClient against a mock Triton gRPC server."""

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
from internal.client.batch_client import BatchIngestionClient
from internal.client.circuit_breaker import CircuitState, RerankerCircuitBreaker
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import pack_fp16_tensor, unpack_string_tensor
from internal.client.models import CandidateDocument
from internal.client.search_client import SearchOrchestratorClient
from internal.config.settings import PoolType


class MockTritonServer(GRPCInferenceServiceServicer):
    """In-memory KServe v2 mock Triton Inference Server."""

    def __init__(self) -> None:
        self.queue_saturated = False
        self.call_counts: dict[str, int] = {
            "online_embedder_ensemble": 0,
            "online_reranker_ensemble": 0,
            "batch_embedder_ensemble": 0,
        }

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
        model_name = request.model_name
        self.call_counts[model_name] = self.call_counts.get(model_name, 0) + 1

        if model_name == "online_embedder_ensemble":
            # Extract texts
            texts = unpack_string_tensor(request.raw_input_contents[0])
            batch_size = len(texts)
            # Emit deterministic 768-dim embeddings
            embeddings = [[0.01 * (i + 1)] * 768 for i in range(batch_size)]
            raw_out = pack_fp16_tensor(embeddings)

            return ModelInferResponse(
                model_name=model_name,
                outputs=[
                    InferOutputTensor(
                        name="embeddings",
                        datatype="FP16",
                        shape=[batch_size, 768],
                    )
                ],
                raw_output_contents=[raw_out],
            )

        elif model_name == "online_reranker_ensemble":
            if self.queue_saturated:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "Triton queue full: max_queue_delay exceeded; shed load",
                )
                return ModelInferResponse()

            # Unpack query and passages
            passages = unpack_string_tensor(request.raw_input_contents[1])
            batch_size = len(passages)
            # Emit higher scores for passages containing 'keyword'
            logits = []
            for p in passages:
                score = 5.0 if "keyword" in p else 1.0
                logits.append([score])

            raw_out = pack_fp16_tensor(logits)
            return ModelInferResponse(
                model_name=model_name,
                outputs=[
                    InferOutputTensor(
                        name="logits",
                        datatype="FP16",
                        shape=[batch_size, 1],
                    )
                ],
                raw_output_contents=[raw_out],
            )

        elif model_name == "batch_embedder_ensemble":
            texts = unpack_string_tensor(request.raw_input_contents[0])
            batch_size = len(texts)
            embeddings = [[0.05] * 768 for _ in range(batch_size)]
            raw_out = pack_fp16_tensor(embeddings)

            return ModelInferResponse(
                model_name=model_name,
                outputs=[
                    InferOutputTensor(
                        name="embeddings",
                        datatype="FP16",
                        shape=[batch_size, 768],
                    )
                ],
                raw_output_contents=[raw_out],
            )

        context.abort(grpc.StatusCode.NOT_FOUND, f"Model {model_name} not found")
        return ModelInferResponse()


class TestOrchestratorIntegration(unittest.TestCase):
    server: grpc.Server
    mock_service: MockTritonServer
    server_target: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        cls.mock_service = MockTritonServer()
        add_GRPCInferenceServiceServicer_to_server(cls.mock_service, cls.server)
        port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.server_target = f"127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop(grace=0)

    def setUp(self) -> None:
        self.mock_service.queue_saturated = False
        self.mock_service.call_counts = dict.fromkeys(self.mock_service.call_counts, 0)

        self.pool = TritonGRPCChannelPool(
            embedding_target=self.server_target,
            reranker_target=self.server_target,
            batch_target=self.server_target,
        )
        self.circuit_breaker = RerankerCircuitBreaker(
            failure_threshold=2,
            recovery_timeout_seconds=0.2,
        )
        self.search_client = SearchOrchestratorClient(
            pool=self.pool,
            circuit_breaker=self.circuit_breaker,
            embed_timeout_seconds=2.0,
            rerank_timeout_seconds=2.0,
        )
        self.batch_client = BatchIngestionClient(
            pool=self.pool,
            max_batch_size=256,
        )

    def tearDown(self) -> None:
        self.search_client.close()
        self.batch_client.close()
        self.pool.close()

    def test_pool_connectivity(self) -> None:
        self.assertTrue(self.pool.is_channel_ready(PoolType.ONLINE_EMBEDDING))
        self.assertTrue(self.pool.is_pool_ready(PoolType.ONLINE_EMBEDDING))

    def test_end_to_end_query_embedding(self) -> None:
        vector = self.search_client.embed_query("enterprise search query")
        self.assertEqual(len(vector), 768)
        self.assertAlmostEqual(vector[0], 0.01, places=3)
        self.assertEqual(self.mock_service.call_counts["online_embedder_ensemble"], 1)

    def test_end_to_end_rerank_success(self) -> None:
        candidates = [
            CandidateDocument(id="doc-1", passage="ordinary passage", initial_rank=1),
            CandidateDocument(id="doc-2", passage="passage with keyword", initial_rank=2),
        ]

        results = self.search_client.rerank("search query", candidates)
        self.assertEqual(len(results), 2)
        # doc-2 contains keyword -> score 5.0 -> rank 1
        self.assertEqual(results[0].id, "doc-2")
        self.assertEqual(results[0].rank, 1)
        self.assertAlmostEqual(results[0].score, 5.0, places=2)
        self.assertFalse(results[0].fallback_applied)

        self.assertEqual(results[1].id, "doc-1")
        self.assertEqual(results[1].rank, 2)
        self.assertAlmostEqual(results[1].score, 1.0, places=2)

    def test_end_to_end_batch_ingestion(self) -> None:
        chunks = [f"document chunk {i}" for i in range(10)]
        embeddings = self.batch_client.embed_chunks(chunks)
        self.assertEqual(len(embeddings), 10)
        self.assertEqual(len(embeddings[0]), 768)
        self.assertAlmostEqual(embeddings[0][0], 0.05, places=3)
        self.assertEqual(self.mock_service.call_counts["batch_embedder_ensemble"], 1)

    def test_queue_saturation_trips_circuit_and_rrf_fallback(self) -> None:
        candidates = [
            CandidateDocument(id="doc-1", passage="first document", initial_rank=1),
            CandidateDocument(id="doc-2", passage="second document", initial_rank=2),
        ]

        # Enable simulated Triton queue saturation
        self.mock_service.queue_saturated = True

        # Call 1: Triton rejects with RESOURCE_EXHAUSTED -> immediate un-reranked RRF fallback
        results1 = self.search_client.rerank("query", candidates)
        self.assertEqual(len(results1), 2)
        self.assertTrue(results1[0].fallback_applied)
        self.assertAlmostEqual(results1[0].score, 1.0 / 61.0, places=6)
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)

        # Call 2: Second failure reaches failure_threshold (2) -> trips to OPEN
        results2 = self.search_client.rerank("query", candidates)
        self.assertTrue(results2[0].fallback_applied)
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)

        # Call 3: Circuit is OPEN -> bypassed gRPC entirely (call count unchanged)
        call_count_before = self.mock_service.call_counts["online_reranker_ensemble"]
        results3 = self.search_client.rerank("query", candidates)
        self.assertEqual(
            self.mock_service.call_counts["online_reranker_ensemble"], call_count_before
        )
        self.assertTrue(results3[0].fallback_applied)
        self.assertEqual(results3[0].fallback_reason, "circuit_breaker_open")


if __name__ == "__main__":
    unittest.main()
