"""Unit tests for SearchOrchestratorClient query embedding and reranking."""

import io
import logging
import unittest
from unittest.mock import MagicMock

import grpc

from api.proto.v1.triton_stubs_pb2 import (
    InferOutputTensor,
    ModelInferResponse,
)
from internal.client.circuit_breaker import RerankerCircuitBreaker
from internal.client.kserve_codec import pack_fp16_tensor
from internal.client.models import CandidateDocument
from internal.client.search_client import SearchOrchestratorClient
from internal.config.settings import PoolType
from internal.logging.secured_logger import SecuredJSONFormatter


class DummyRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


class TestSearchClient(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_pool = MagicMock()
        self.mock_embed_stub = MagicMock()
        self.mock_rerank_stub = MagicMock()

        def stub_for_pool(pool_type: PoolType) -> MagicMock:
            if pool_type == PoolType.ONLINE_EMBEDDING:
                return self.mock_embed_stub
            elif pool_type == PoolType.ONLINE_RERANKER:
                return self.mock_rerank_stub
            return MagicMock()

        self.mock_pool.get_stub.side_effect = stub_for_pool
        self.circuit_breaker = RerankerCircuitBreaker(failure_threshold=2)
        self.client = SearchOrchestratorClient(
            pool=self.mock_pool,
            circuit_breaker=self.circuit_breaker,
        )

        # Setup log capture for secured logging audit
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(SecuredJSONFormatter())
        self.logger = logging.getLogger("search-client")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.handler)

    def test_embed_query_success(self) -> None:
        mock_vector = [0.123] * 768
        raw_fp16 = pack_fp16_tensor([mock_vector])

        resp = ModelInferResponse(
            model_name="online_embedder_ensemble",
            outputs=[InferOutputTensor(name="embeddings", shape=[1, 768], datatype="FP16")],
            raw_output_contents=[raw_fp16],
        )
        self.mock_embed_stub.ModelInfer.return_value = resp

        vector = self.client.embed_query(
            "financial quarter earnings query", request_id="req-embed-001"
        )
        self.assertEqual(len(vector), 768)
        self.assertAlmostEqual(vector[0], 0.123, places=3)

        # Assert 15ms client deadline was requested
        call_args = self.mock_embed_stub.ModelInfer.call_args
        self.assertAlmostEqual(call_args.kwargs["timeout"], 0.015)

        # Secured logging invariant audit
        log_output = self.log_stream.getvalue()
        self.assertIn("req-embed-001", log_output)
        self.assertIn("online_embedder_ensemble", log_output)
        self.assertNotIn("financial quarter earnings query", log_output)

    def test_embed_query_failure_raises(self) -> None:
        self.mock_embed_stub.ModelInfer.side_effect = TimeoutError("Deadline reached")
        with self.assertRaises(TimeoutError):
            self.client.embed_query("my confidential search query")

        log_output = self.log_stream.getvalue()
        self.assertNotIn("my confidential search query", log_output)

    def test_rerank_empty_candidates_fast_return(self) -> None:
        results = self.client.rerank("some query", [])
        self.assertEqual(results, [])
        self.assertEqual(self.mock_rerank_stub.ModelInfer.call_count, 0)

    def test_rerank_success_and_ranking_order(self) -> None:
        candidates = [
            CandidateDocument(id="doc-A", passage="Passage A about economics", initial_rank=1),
            CandidateDocument(id="doc-B", passage="Passage B with high relevance", initial_rank=2),
            CandidateDocument(id="doc-C", passage="Passage C with low relevance", initial_rank=3),
        ]

        # Logits: B highest (3.5), A medium (1.2), C lowest (-0.5)
        raw_logits = pack_fp16_tensor([[1.2], [3.5], [-0.5]])
        resp = ModelInferResponse(
            model_name="online_reranker_ensemble",
            outputs=[InferOutputTensor(name="logits", shape=[3, 1], datatype="FP16")],
            raw_output_contents=[raw_logits],
        )
        self.mock_rerank_stub.ModelInfer.return_value = resp

        results = self.client.rerank("interest rates query", candidates, request_id="req-rerank-01")

        self.assertEqual(len(results), 3)
        # doc-B must be rank 1
        self.assertEqual(results[0].id, "doc-B")
        self.assertEqual(results[0].rank, 1)
        self.assertAlmostEqual(results[0].score, 3.5, places=2)
        self.assertFalse(results[0].fallback_applied)

        # doc-A must be rank 2
        self.assertEqual(results[1].id, "doc-A")
        self.assertEqual(results[1].rank, 2)

        # doc-C must be rank 3
        self.assertEqual(results[2].id, "doc-C")
        self.assertEqual(results[2].rank, 3)

        # Check deadline is 35ms
        call_args = self.mock_rerank_stub.ModelInfer.call_args
        self.assertAlmostEqual(call_args.kwargs["timeout"], 0.035)

        # Check secured logging invariant
        log_output = self.log_stream.getvalue()
        self.assertIn("req-rerank-01", log_output)
        self.assertNotIn("interest rates query", log_output)
        self.assertNotIn("Passage B with high relevance", log_output)

    def test_rerank_queue_saturation_triggers_rrf_fallback(self) -> None:
        candidates = [
            CandidateDocument(id="doc-1", passage="Passage 1", initial_rank=1),
            CandidateDocument(id="doc-2", passage="Passage 2", initial_rank=2),
        ]
        # Simulate Triton queue full rejection
        self.mock_rerank_stub.ModelInfer.side_effect = DummyRpcError(
            grpc.StatusCode.RESOURCE_EXHAUSTED
        )

        results = self.client.rerank("any query", candidates)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].fallback_applied)
        self.assertIn("RESOURCE_EXHAUSTED", str(results[0].fallback_reason))
        # Initial rank 1 -> 1 / (60 + 1) = 1/61
        self.assertAlmostEqual(results[0].score, 1.0 / 61.0, places=6)

    def test_rerank_circuit_open_instant_fallback(self) -> None:
        candidates = [
            CandidateDocument(id="doc-1", passage="Passage 1", initial_rank=1),
        ]
        # Force circuit OPEN
        self.circuit_breaker.record_failure(DummyRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED))
        self.circuit_breaker.record_failure(DummyRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED))

        results = self.client.rerank("query", candidates)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].fallback_applied)
        self.assertEqual(results[0].fallback_reason, "circuit_breaker_open")
        # Ensure zero gRPC network calls were made
        self.assertEqual(self.mock_rerank_stub.ModelInfer.call_count, 0)

    def test_rerank_batch_chunking_exceeding_32(self) -> None:
        candidates = [
            CandidateDocument(id=f"doc-{i}", passage=f"passage-{i}", initial_rank=i)
            for i in range(1, 41)  # 40 candidates (> 32)
        ]

        # Chunk 1: 32 candidates, Chunk 2: 8 candidates
        chunk1_logits = pack_fp16_tensor([[0.5] for _ in range(32)])
        chunk2_logits = pack_fp16_tensor([[0.8] for _ in range(8)])

        resp1 = ModelInferResponse(
            model_name="online_reranker_ensemble",
            outputs=[InferOutputTensor(name="logits", shape=[32, 1], datatype="FP16")],
            raw_output_contents=[chunk1_logits],
        )
        resp2 = ModelInferResponse(
            model_name="online_reranker_ensemble",
            outputs=[InferOutputTensor(name="logits", shape=[8, 1], datatype="FP16")],
            raw_output_contents=[chunk2_logits],
        )
        self.mock_rerank_stub.ModelInfer.side_effect = [resp1, resp2]

        results = self.client.rerank("query", candidates, max_batch_size=32)
        self.assertEqual(len(results), 40)
        self.assertEqual(self.mock_rerank_stub.ModelInfer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
