"""Configuration settings and environment variable definitions for rag-inference-serving.

Enforces calibrated invariants across the three physically isolated compute pools:
- Online Embedding Pool (Pool A): <= 2 ms queue delay, max batch 32
- Online Reranker Pool (Pool B): <= 4 ms queue delay, max batch 32
- Batch Ingestion Pool: <= 50 ms queue delay, max batch 256

VRAM headroom & ceilings:
- Aggregate static model allocations <= 20 GiB (83.33% of 24 GB)
- Active runtime memory threshold: > 85% (>20.4 GiB) triggers 503 shedding
- Dedicated dynamic batching activation headroom: >= 4 GiB
"""

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PoolType(StrEnum):
    """Physically isolated GPU compute pools."""

    ONLINE_EMBEDDING = "online_embedding"
    ONLINE_RERANKER = "online_reranker"
    BATCH_INGESTION = "batch_ingestion"


# Calibrated pool delay and batch size ceilings
POOL_MAX_QUEUE_DELAY_US: dict[PoolType, int] = {
    PoolType.ONLINE_EMBEDDING: 2000,  # <= 2 ms
    PoolType.ONLINE_RERANKER: 4000,  # <= 4 ms
    PoolType.BATCH_INGESTION: 50000,  # <= 50 ms
}

POOL_MAX_BATCH_SIZE: dict[PoolType, int] = {
    PoolType.ONLINE_EMBEDDING: 32,
    PoolType.ONLINE_RERANKER: 32,
    PoolType.BATCH_INGESTION: 256,
}

# 24 GiB NVIDIA L4/A10G memory layout
DEFAULT_VRAM_CAPACITY_BYTES: int = 24 * 1024 * 1024 * 1024  # 25,769,803,776 bytes (24 GiB)
MAX_STATIC_ENGINE_VRAM_BYTES: int = (
    20 * 1024 * 1024 * 1024
)  # 21,474,836,480 bytes (20 GiB, ~83.33%)
MAX_RUNTIME_VRAM_UTILIZATION_RATIO: float = 0.85  # 85% (> 20.4 GiB triggers shedding)
DEDICATED_ACTIVATION_HEADROOM_BYTES: int = 4 * 1024 * 1024 * 1024  # 4 GiB dedicated headroom


class ServingSettings(BaseSettings):
    """Runtime microservice settings for operational server and Triton integration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_SERVING_",
        case_sensitive=False,
        extra="ignore",
    )

    pool_type: PoolType = Field(
        default=PoolType.ONLINE_EMBEDDING,
        description="Target compute pool type defining queue delay and batch ceilings.",
    )

    server_host: str = Field(
        default="0.0.0.0",
        description="Host address for operational ASGI server.",
    )
    server_port: int = Field(
        default=8080,
        description="Port for out-of-band operational ASGI server (port 8080).",
    )

    triton_http_url: str = Field(
        default="http://localhost:8000",
        description="Internal HTTP endpoint for Triton KServe v2 health/readiness.",
    )
    triton_grpc_url: str = Field(
        default="localhost:8001",
        description="Internal gRPC endpoint for Triton KServe v2 inference.",
    )
    online_embed_grpc_url: str = Field(
        default="localhost:8001",
        description="Endpoint for online embedding pool Triton gRPC service.",
    )
    online_rerank_grpc_url: str = Field(
        default="localhost:8001",
        description="Endpoint for online reranker pool Triton gRPC service.",
    )
    batch_ingest_grpc_url: str = Field(
        default="localhost:8001",
        description="Endpoint for batch ingestion pool Triton gRPC service.",
    )

    max_queue_delay_microseconds: int = Field(
        default=2000,
        description="Native Triton queue delay timeout in microseconds.",
    )
    max_batch_size: int = Field(
        default=32,
        description="Maximum dynamic batch size permitted for this pool.",
    )

    vram_capacity_bytes: int = Field(
        default=DEFAULT_VRAM_CAPACITY_BYTES,
        description="Total GPU VRAM capacity in bytes (default 24 GiB).",
    )
    max_static_vram_bytes: int = Field(
        default=MAX_STATIC_ENGINE_VRAM_BYTES,
        description="Ceiling for aggregate static model allocations in bytes (<= 20 GiB).",
    )
    vram_max_utilization_ratio: float = Field(
        default=MAX_RUNTIME_VRAM_UTILIZATION_RATIO,
        description="Active runtime VRAM utilization ratio ceiling triggering shedding (default 0.85).",
    )

    models_expected: list[str] = Field(
        default_factory=lambda: [
            "online_embedder_ensemble",
            "dense_embedder_trt",
            "text_tokenizer",
        ],
        description="List of model engine names expected to be loaded and healthy.",
    )

    shm_base_path: str = Field(
        default="/dev/shm",
        description="Base path for pod-scoped POSIX shared memory emptyDir volume.",
    )

    # gRPC Transport Security (TLS / mTLS)
    grpc_tls_enabled: bool = Field(
        default=False,
        description="Enable TLS encryption for outbound gRPC client channels.",
    )
    grpc_ca_cert_path: str | None = Field(
        default=None,
        description="Filesystem path to root CA certificate for verifying TLS server certificates.",
    )
    grpc_client_cert_path: str | None = Field(
        default=None,
        description="Filesystem path to client certificate for mutual TLS (mTLS) authentication.",
    )
    grpc_client_key_path: str | None = Field(
        default=None,
        description="Filesystem path to client private key for mutual TLS (mTLS) authentication.",
    )
    grpc_server_name_override: str | None = Field(
        default=None,
        description="Optional SNI server name override for TLS certificate validation.",
    )

    # Operational Server Security & Telemetry
    api_key: str | None = Field(
        default=None,
        description="Optional bearer API key for authenticating operational scrape endpoints.",
    )
    log_level: str = Field(
        default="INFO",
        description="Structured logging verbosity level (DEBUG, INFO, WARNING, ERROR).",
    )

    @model_validator(mode="before")
    @classmethod
    def apply_pool_defaults_if_not_explicit(cls, data: Any) -> Any:
        """Apply calibrated pool defaults if queue delay or batch size are not explicitly passed."""
        if not isinstance(data, dict):
            return data

        pool_type_raw = data.get("pool_type", PoolType.ONLINE_EMBEDDING)
        if isinstance(pool_type_raw, str):
            try:
                pool = PoolType(pool_type_raw.lower())
            except ValueError:
                pool = PoolType.ONLINE_EMBEDDING
        else:
            pool = pool_type_raw

        if "max_queue_delay_microseconds" not in data:
            data["max_queue_delay_microseconds"] = POOL_MAX_QUEUE_DELAY_US.get(pool, 2000)

        if "max_batch_size" not in data:
            data["max_batch_size"] = POOL_MAX_BATCH_SIZE.get(pool, 32)

        if "models_expected" not in data:
            if pool == PoolType.ONLINE_EMBEDDING:
                data["models_expected"] = [
                    "online_embedder_ensemble",
                    "dense_embedder_trt",
                    "text_tokenizer",
                ]
            elif pool == PoolType.ONLINE_RERANKER:
                data["models_expected"] = [
                    "online_reranker_ensemble",
                    "cross_encoder_trt",
                    "text_tokenizer",
                ]
            elif pool == PoolType.BATCH_INGESTION:
                data["models_expected"] = [
                    "batch_embedder_ensemble",
                    "dense_embedder_trt",
                    "text_tokenizer",
                ]

        return data


@lru_cache(maxsize=1)
def get_settings() -> ServingSettings:
    """Return cached singleton ServingSettings instance."""
    return ServingSettings()
