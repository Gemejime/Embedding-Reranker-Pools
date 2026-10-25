"""Client library for rag-inference-serving microservice."""

from internal.client.batch_client import BatchIngestionClient
from internal.client.circuit_breaker import CircuitState, RerankerCircuitBreaker
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import (
    build_model_infer_request,
    build_single_text_infer_request,
    build_text_pair_infer_request,
    pack_fp16_tensor,
    pack_string_tensor,
    unpack_fp16_tensor,
    unpack_string_tensor,
)
from internal.client.models import CandidateDocument, RerankResult
from internal.client.search_client import SearchOrchestratorClient

__all__ = [
    "BatchIngestionClient",
    "CandidateDocument",
    "CircuitState",
    "RerankResult",
    "RerankerCircuitBreaker",
    "SearchOrchestratorClient",
    "TritonGRPCChannelPool",
    "build_model_infer_request",
    "build_single_text_infer_request",
    "build_text_pair_infer_request",
    "pack_fp16_tensor",
    "pack_string_tensor",
    "unpack_fp16_tensor",
    "unpack_string_tensor",
]
