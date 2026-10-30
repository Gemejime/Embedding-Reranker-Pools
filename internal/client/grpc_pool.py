"""High-performance gRPC connection pooling for decoupled Triton Inference Server pools.

Manages dedicated HTTP/2 channels for:
- Online Embedding Pool (Pool A)
- Online Reranker Pool (Pool B)
- Batch Ingestion Pool

Enforces production keepalive ping intervals and payload boundaries.
"""

import ipaddress
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
    ("grpc.keepalive_permit_without_calls", 0),
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
        loopback_target: str | None = None,
        options: list[tuple[str, Any]] | None = None,
        settings: ServingSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._targets: dict[PoolType, str] = {
            PoolType.ONLINE_EMBEDDING: self._resolve_target(
                embedding_target
                or self._settings.online_embed_grpc_url
                or self._settings.triton_grpc_url
            ),
            PoolType.ONLINE_RERANKER: self._resolve_target(
                reranker_target
                or self._settings.online_rerank_grpc_url
                or self._settings.triton_grpc_url
            ),
            PoolType.BATCH_INGESTION: self._resolve_target(
                batch_target
                or self._settings.batch_ingest_grpc_url
                or self._settings.triton_grpc_url
            ),
        }
        self._options = list(options if options is not None else DEFAULT_GRPC_OPTIONS)
        self._channels: dict[PoolType, grpc.Channel] = {}
        self._stubs: dict[PoolType, GRPCInferenceServiceStub] = {}
        # Resolve loopback target: default to triton_grpc_url or local batch target if specified
        if loopback_target:
            resolved_loopback = loopback_target
        elif (
            isinstance(batch_target, str)
            and (batch_target.startswith("127.0.0.1:") or batch_target.startswith("localhost:"))
            and not batch_target.startswith("dns:")
        ):
            resolved_loopback = batch_target
        else:
            resolved_loopback = self._settings.triton_grpc_url

        self._loopback_target = self._resolve_target(resolved_loopback)
        self._loopback_channel: grpc.Channel | None = None
        self._loopback_stub: GRPCInferenceServiceStub | None = None
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
            if joined.startswith("ipv4:") or joined.startswith("dns:"):
                return joined
            for entry in targets:
                host = entry.split(":")[0] if ":" in entry else entry
                try:
                    ip = ipaddress.ip_address(host)
                    if ip.version != 4:
                        raise ValueError(f"'{entry}' is not an IPv4 address")
                except ValueError as err:
                    raise ValueError(
                        f"Invalid multi-target entry '{entry}': multi-target entries without an explicit resolver prefix must be IPv4 literals. For hostnames, specify a 'dns:' prefix."
                    ) from err
            return f"ipv4:{joined}"
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
                # Enable round-robin LB for multi-instance or dns resolver endpoints.
                # Note: DNS targets (dns:///) require headless Kubernetes Services (clusterIP: None)
                # so the c-ares resolver discovers individual pod replica IPs instead of a single virtual ClusterIP.
                if ("," in target or target.startswith("dns:")) and not any(
                    k == "grpc.lb_policy_name" for k, _ in options
                ):
                    options.append(("grpc.lb_policy_name", "round_robin"))

                logger.info(
                    "Initializing gRPC channel for compute pool",
                    extra={
                        "pool_type": pool_type.value,
                        "target": target,
                        "status": "channel_created",
                    },
                )
                if self._settings.grpc_tls_enabled:
                    root_certs: bytes | None = None
                    if self._settings.grpc_ca_cert_path:
                        with open(self._settings.grpc_ca_cert_path, "rb") as f:
                            root_certs = f.read()
                    private_key: bytes | None = None
                    cert_chain: bytes | None = None
                    if self._settings.grpc_client_key_path and self._settings.grpc_client_cert_path:
                        with open(self._settings.grpc_client_key_path, "rb") as f:
                            private_key = f.read()
                        with open(self._settings.grpc_client_cert_path, "rb") as f:
                            cert_chain = f.read()

                    credentials = grpc.ssl_channel_credentials(
                        root_certificates=root_certs,
                        private_key=private_key,
                        certificate_chain=cert_chain,
                    )
                    if self._settings.grpc_server_name_override:
                        options.append(
                            (
                                "grpc.ssl_target_name_override",
                                self._settings.grpc_server_name_override,
                            )
                        )
                    channel = grpc.secure_channel(target, credentials, options=options)
                else:
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

    def get_loopback_channel(self) -> grpc.Channel:
        """Lazily instantiate and return a dedicated loopback gRPC channel for pod-local operations."""
        if self._closed:
            raise RuntimeError("Cannot get channel from closed TritonGRPCChannelPool")

        with self._lock:
            if self._loopback_channel is None:
                target = self._loopback_target
                options = list(self._options)
                logger.info(
                    "Initializing loopback gRPC channel for pod-local operations",
                    extra={"target": target, "status": "loopback_channel_created"},
                )
                if self._settings.grpc_tls_enabled:
                    root_certs: bytes | None = None
                    if self._settings.grpc_ca_cert_path:
                        with open(self._settings.grpc_ca_cert_path, "rb") as f:
                            root_certs = f.read()
                    private_key: bytes | None = None
                    cert_chain: bytes | None = None
                    if self._settings.grpc_client_key_path and self._settings.grpc_client_cert_path:
                        with open(self._settings.grpc_client_key_path, "rb") as f:
                            private_key = f.read()
                        with open(self._settings.grpc_client_cert_path, "rb") as f:
                            cert_chain = f.read()

                    credentials = grpc.ssl_channel_credentials(
                        root_certificates=root_certs,
                        private_key=private_key,
                        certificate_chain=cert_chain,
                    )
                    if self._settings.grpc_server_name_override:
                        options.append(
                            (
                                "grpc.ssl_target_name_override",
                                self._settings.grpc_server_name_override,
                            )
                        )
                    channel = grpc.secure_channel(target, credentials, options=options)
                else:
                    channel = grpc.insecure_channel(target, options=options)
                self._loopback_channel = channel
                self._loopback_stub = GRPCInferenceServiceStub(channel)
            return self._loopback_channel

    def get_loopback_stub(self) -> GRPCInferenceServiceStub:
        """Return the GRPCInferenceServiceStub for pod-local loopback operations."""
        self.get_loopback_channel()
        with self._lock:
            assert self._loopback_stub is not None
            return self._loopback_stub

    def reconnect_loopback_channel(self) -> grpc.Channel:
        """Recycle and recreate the loopback channel cleanly."""
        if self._closed:
            raise RuntimeError("Cannot reconnect loopback channel on closed TritonGRPCChannelPool")

        with self._lock:
            old_channel = self._loopback_channel
            self._loopback_channel = None
            self._loopback_stub = None
            if old_channel is not None:
                try:
                    logger.info(
                        "Recycling loopback gRPC channel socket",
                        extra={"target": self._loopback_target, "status": "recycling"},
                    )
                    old_channel.close()
                except Exception as exc:
                    logger.warning("Error draining recycled loopback channel", exc_info=exc)

        return self.get_loopback_channel()

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
            if self._loopback_channel is not None:
                try:
                    logger.info(
                        "Closing loopback gRPC channel",
                        extra={"target": self._loopback_target, "status": "closing"},
                    )
                    self._loopback_channel.close()
                except Exception as exc:
                    logger.warning("Error closing loopback channel", exc_info=exc)
                self._loopback_channel = None
                self._loopback_stub = None

    def __enter__(self) -> "TritonGRPCChannelPool":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
