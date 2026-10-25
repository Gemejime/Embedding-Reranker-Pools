"""High-performance gRPC connection pooling for decoupled Triton Inference Server pools.

Manages dedicated HTTP/2 channels for:
- Online Embedding Pool (Pool A)
- Online Reranker Pool (Pool B)
- Batch Ingestion Pool

Enforces production keepalive ping intervals and payload boundaries.
"""

import threading
from collections.abc import Sequence
from typing import Any

import grpc

from api.proto.v1.triton_stubs_pb2 import ServerLiveRequest, ServerReadyRequest
from api.proto.v1.triton_stubs_pb2_grpc import GRPCInferenceServiceStub
from internal.config.settings import PoolType, ServingSettings, get_settings
from internal.logging.secured_logger import get_secured_logger

logger = get_secured_logger("grpc-pool")

DEFAULT_GRPC_OPTIONS: list[tuple[str, Any]] = [
    ("grpc.keepalive_time_ms", 10000),  # 10s pings
    ("grpc.keepalive_timeout_ms", 5000),  # 5s ack timeout
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),  # 64 MiB
    ("grpc.max_send_message_length", 64 * 1024 * 1024),  # 64 MiB
]


class TritonGRPCChannelPool:
    """Thread-safe connection pool managing discrete gRPC channels per compute pool."""

    def __init__(
        self,
        embedding_target: str | Sequence[str] | None = None,
        reranker_target: str | Sequence[str] | None = None,
        batch_target: str | Sequence[str] | None = None,
        options: list[tuple[str, Any]] | None = None,
        settings: ServingSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._targets: dict[PoolType, str] = {
            PoolType.ONLINE_EMBEDDING: self._resolve_target(
                embedding_target or self._settings.online_embed_grpc_url
            ),
            PoolType.ONLINE_RERANKER: self._resolve_target(
                reranker_target or self._settings.online_rerank_grpc_url
            ),
            PoolType.BATCH_INGESTION: self._resolve_target(
                batch_target or self._settings.batch_ingest_grpc_url
            ),
        }
        self._options = list(options if options is not None else DEFAULT_GRPC_OPTIONS)
        self._channels: dict[PoolType, grpc.Channel] = {}
        self._stubs: dict[PoolType, GRPCInferenceServiceStub] = {}
        self._lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _resolve_target(raw_target: str | Sequence[str]) -> str:
        """Resolve string or multi-endpoint sequence into gRPC target string."""
        if isinstance(raw_target, str):
            return raw_target
        if isinstance(raw_target, Sequence):
            targets = list(raw_target)
            if not targets:
                return "localhost:8001"
            if len(targets) == 1:
                return targets[0]
            joined = ",".join(targets)
            if not (joined.startswith("ipv4:") or joined.startswith("dns:")):
                return f"ipv4:{joined}"
            return joined
        return str(raw_target)

    def get_target(self, pool_type: PoolType) -> str:
        """Return the target address configured for the specified compute pool."""
        return self._targets.get(pool_type, "localhost:8001")

    def get_active_channel_count(self) -> int:
        """Return the number of currently active channels in the pool."""
        with self._lock:
            return len(self._channels)

    def get_channel(self, pool_type: PoolType) -> grpc.Channel:
        """Lazily instantiate and return a thread-safe gRPC channel for the compute pool."""
        if self._closed:
            raise RuntimeError("Cannot get channel from closed TritonGRPCChannelPool")

        with self._lock:
            channel = self._channels.get(pool_type)
            if channel is None:
                target = self.get_target(pool_type)
                options = list(self._options)
                # Enable round-robin LB for multi-instance endpoints
                if "," in target and not any(k == "grpc.lb_policy_name" for k, _ in options):
                    options.append(("grpc.lb_policy_name", "round_robin"))

                logger.info(
                    "Initializing gRPC channel for compute pool",
                    extra={
                        "pool_type": pool_type.value,
                        "target": target,
                        "status": "channel_created",
                    },
                )
                channel = grpc.insecure_channel(target, options=options)
                self._channels[pool_type] = channel
                self._stubs[pool_type] = GRPCInferenceServiceStub(channel)
            return channel

    def reconnect_channel(self, pool_type: PoolType) -> grpc.Channel:
        """Recycle and recreate a channel for the specified pool, draining old sockets cleanly."""
        if self._closed:
            raise RuntimeError("Cannot reconnect channel on closed TritonGRPCChannelPool")

        with self._lock:
            old_channel = self._channels.pop(pool_type, None)
            self._stubs.pop(pool_type, None)
            if old_channel is not None:
                try:
                    logger.info(
                        "Recycling gRPC channel socket",
                        extra={"pool_type": pool_type.value, "status": "recycling"},
                    )
                    old_channel.close()
                except Exception as exc:
                    logger.warning("Error draining recycled channel", exc_info=exc)

        return self.get_channel(pool_type)

    def get_stub(self, pool_type: PoolType) -> GRPCInferenceServiceStub:
        """Return the GRPCInferenceServiceStub for the specified compute pool."""
        self.get_channel(pool_type)
        with self._lock:
            return self._stubs[pool_type]

    def is_channel_ready(self, pool_type: PoolType, timeout_seconds: float = 2.0) -> bool:
        """Check active connectivity and Triton liveness for the specified pool."""
        if self._closed:
            return False
        try:
            stub = self.get_stub(pool_type)
            resp = stub.ServerLive(ServerLiveRequest(), timeout=timeout_seconds)
            return bool(resp.live)
        except Exception:
            return False

    def is_pool_ready(self, pool_type: PoolType, timeout_seconds: float = 2.0) -> bool:
        """Check Triton readiness for the specified pool."""
        if self._closed:
            return False
        try:
            stub = self.get_stub(pool_type)
            resp = stub.ServerReady(ServerReadyRequest(), timeout=timeout_seconds)
            return bool(resp.ready)
        except Exception:
            return False

    def close(self) -> None:
        """Close all channels and drain connection sockets cleanly."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for pool_type, channel in self._channels.items():
                try:
                    logger.info(
                        "Closing gRPC channel for compute pool",
                        extra={"pool_type": pool_type.value, "status": "closing"},
                    )
                    channel.close()
                except Exception as exc:
                    logger.warning("Error closing channel", exc_info=exc)
            self._channels.clear()
            self._stubs.clear()

    def __enter__(self) -> "TritonGRPCChannelPool":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
