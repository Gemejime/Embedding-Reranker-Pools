"""High-Performance Search Orchestrator Client for RAG Serving.

Provides:
- embed_query: Sub-15ms client deadline query embedding via online_embedder_ensemble.
- rerank: Sub-35ms client deadline candidate reranking via online_reranker_ensemble with
  automatic circuit-breaking and un-reranked RRF fallback.
- Strict secured logging integration with zero raw payload emission.
"""

import time
import uuid
from collections.abc import Sequence

import grpc

from internal.client.circuit_breaker import RerankerCircuitBreaker
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import (
    build_single_text_infer_request,
    build_text_pair_infer_request,
    unpack_fp16_tensor,
)
from internal.client.models import CandidateDocument, RerankResult
from internal.config.settings import PoolType, ServingSettings, get_settings
from internal.logging.secured_logger import get_secured_logger

logger = get_secured_logger("search-client")

DEFAULT_EMBED_TIMEOUT_SECONDS: float = 0.015  # 15 ms client deadline
DEFAULT_RERANK_TIMEOUT_SECONDS: float = 0.035  # 35 ms client deadline


class SearchOrchestratorClient:
    """Production gRPC client connecting search orchestrators to Triton serving pools."""

    def __init__(
        self,
        pool: TritonGRPCChannelPool | None = None,
        circuit_breaker: RerankerCircuitBreaker | None = None,
        settings: ServingSettings | None = None,
        embed_timeout_seconds: float = DEFAULT_EMBED_TIMEOUT_SECONDS,
        rerank_timeout_seconds: float = DEFAULT_RERANK_TIMEOUT_SECONDS,
    ) -> None:
        self._settings = settings or get_settings()
        self._pool = pool or TritonGRPCChannelPool(settings=self._settings)
        self._owns_pool = pool is None
        self._circuit_breaker = circuit_breaker or RerankerCircuitBreaker()
        self._embed_timeout = embed_timeout_seconds
        self._rerank_timeout = rerank_timeout_seconds

    @property
    def circuit_breaker(self) -> RerankerCircuitBreaker:
        """Return the circuit breaker instance."""
        return self._circuit_breaker

    @property
    def pool(self) -> TritonGRPCChannelPool:
        """Return the channel pool instance."""
        return self._pool

    def embed_query(
        self,
        query: str,
        request_id: str | None = None,
        model_name: str = "online_embedder_ensemble",
    ) -> list[float]:
        """Embed a single search query text within the 15ms client latency budget.

        Unpacks 768-dimensional FP16 vector from Triton online_embedder_ensemble.
        """
        req_id = request_id or f"embed-{uuid.uuid4().hex[:8]}"
        req = build_single_text_infer_request(
            model_name=model_name,
            texts=[query],
            input_name="raw_text",
            output_name="embeddings",
            request_id=req_id,
        )

        stub = self._pool.get_stub(PoolType.ONLINE_EMBEDDING)
        t0 = time.perf_counter()

        try:
            resp = stub.ModelInfer(req, timeout=self._embed_timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if not resp.raw_output_contents:
                raise RuntimeError(
                    f"ModelInferResponse for '{model_name}' did not return raw_output_contents"
                )

            # Unpack FP16 tensor [1, 768]
            output_tensor = resp.outputs[0] if resp.outputs else None
            shape = tuple(output_tensor.shape) if output_tensor else (1, 768)
            unpacked = unpack_fp16_tensor(resp.raw_output_contents[0], shape=shape)
            vector = unpacked[0]

            logger.info(
                "Query embedding generated successfully",
                extra={
                    "request_id": req_id,
                    "model_name": model_name,
                    "pool_type": PoolType.ONLINE_EMBEDDING.value,
                    "batch_size": 1,
                    "compute_latency_ms": round(latency_ms, 2),
                    "status": "success",
                },
            )
            return vector

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "Query embedding request failed",
                extra={
                    "request_id": req_id,
                    "model_name": model_name,
                    "pool_type": PoolType.ONLINE_EMBEDDING.value,
                    "batch_size": 1,
                    "compute_latency_ms": round(latency_ms, 2),
                    "status": "error",
                    "error_type": type(exc).__name__,
                },
                exc_info=exc,
            )
            raise

    def rerank(
        self,
        query: str,
        candidates: Sequence[CandidateDocument],
        request_id: str | None = None,
        model_name: str = "online_reranker_ensemble",
        max_batch_size: int = 32,
    ) -> list[RerankResult]:
        """Rerank candidate passages against the query within the 35ms client latency budget.

        Validates inputs, chunks batches exceeding 32 candidates, and instantly falls back
        to un-reranked RRF scores if the circuit is OPEN or on server queue errors / timeouts.
        """
        if not candidates:
            return []

        req_id = request_id or f"rerank-{uuid.uuid4().hex[:8]}"

        # Circuit breaker fast-path check
        if not self._circuit_breaker.can_execute():
            logger.warning(
                "Reranker circuit breaker OPEN; applying instantaneous un-reranked RRF fallback",
                extra={
                    "request_id": req_id,
                    "model_name": model_name,
                    "pool_type": PoolType.ONLINE_RERANKER.value,
                    "batch_size": len(candidates),
                    "status": "circuit_open_fallback",
                },
            )
            return self._circuit_breaker.compute_rrf_fallback(
                candidates, reason="circuit_breaker_open"
            )

        stub = self._pool.get_stub(PoolType.ONLINE_RERANKER)
        all_scored_docs: list[tuple[CandidateDocument, float]] = []
        t0_total = time.perf_counter()

        # Dynamic chunking for batches > max_batch_size (32)
        candidate_list = list(candidates)
        for i in range(0, len(candidate_list), max_batch_size):
            chunk = candidate_list[i : i + max_batch_size]
            queries = [query] * len(chunk)
            passages = [c.passage for c in chunk]

            req = build_text_pair_infer_request(
                model_name=model_name,
                queries=queries,
                passages=passages,
                query_input_name="query",
                passage_input_name="passage",
                output_name="logits",
                request_id=req_id,
            )

            try:
                resp = stub.ModelInfer(req, timeout=self._rerank_timeout)
                if not resp.raw_output_contents:
                    raise RuntimeError("Missing raw_output_contents in reranker response")

                # Unpack FP16 logits [len(chunk), 1]
                shape = (len(chunk), 1)
                unpacked_logits = unpack_fp16_tensor(resp.raw_output_contents[0], shape=shape)

                for doc, logit_row in zip(chunk, unpacked_logits, strict=False):
                    score = float(logit_row[0]) if logit_row else 0.0
                    all_scored_docs.append((doc, score))

                self._circuit_breaker.record_success()

            except Exception as exc:
                latency_ms = (time.perf_counter() - t0_total) * 1000.0
                if self._circuit_breaker.is_tripping_error(exc):
                    self._circuit_breaker.record_failure(exc)

                status_desc = "queue_saturated_fallback"
                reason_desc = f"call_failed_{type(exc).__name__}"
                if isinstance(exc, grpc.RpcError):
                    code = getattr(exc, "code", lambda: None)()
                    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                        status_desc = "resource_exhausted_fallback"
                        reason_desc = "resource_exhausted_fallback_RESOURCE_EXHAUSTED"
                    elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
                        status_desc = "deadline_exceeded_fallback"
                        reason_desc = "deadline_exceeded_fallback_DEADLINE_EXCEEDED"
                    elif code is not None:
                        reason_desc = f"call_failed_{code.name}"

                logger.warning(
                    "Online reranker call failed; returning deterministic un-reranked RRF scores",
                    extra={
                        "request_id": req_id,
                        "model_name": model_name,
                        "pool_type": PoolType.ONLINE_RERANKER.value,
                        "batch_size": len(candidates),
                        "compute_latency_ms": round(latency_ms, 2),
                        "status": status_desc,
                        "error_type": type(exc).__name__,
                    },
                )
                return self._circuit_breaker.compute_rrf_fallback(candidates, reason=reason_desc)

        latency_ms = (time.perf_counter() - t0_total) * 1000.0

        # Sort all candidates by relevance score descending
        all_scored_docs.sort(key=lambda item: item[1], reverse=True)

        results: list[RerankResult] = [
            RerankResult(
                id=doc.id,
                score=score,
                rank=idx,
                fallback_applied=False,
                fallback_reason=None,
            )
            for idx, (doc, score) in enumerate(all_scored_docs, start=1)
        ]

        logger.info(
            "Candidate documents successfully reranked",
            extra={
                "request_id": req_id,
                "model_name": model_name,
                "pool_type": PoolType.ONLINE_RERANKER.value,
                "batch_size": len(candidates),
                "compute_latency_ms": round(latency_ms, 2),
                "status": "success",
            },
        )
        return results

    def close(self) -> None:
        """Close client resources if pool is owned."""
        if self._owns_pool:
            self._pool.close()

    def __enter__(self) -> "SearchOrchestratorClient":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
