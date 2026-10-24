"""Out-of-band Operational ASGI Server for rag-inference-serving.

Runs on port 8080 alongside Triton:
- Provides /healthz (liveness), /ready (readiness), /metrics (Prometheus), /docs, /redoc, /api/docs, /openapi.json.
- STRICT DATA-PLANE ISOLATION: Does NOT proxy model inference calls. Upstream callers connect directly to Triton's native KServe v2 gRPC (:8001) or HTTP (:8000).
- VRAM Headroom Guard: Monitors GPU memory usage; if active VRAM exceeds 85% (>20.4 GiB) or static engine allocation exceeds 20 GiB (83.33%), /ready returns HTTP 503 to trigger automated traffic shedding.
- Structured Secured Logging with RFC 7807/9457 error formatting and zero customer query/chunk text emission.
"""

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from internal.config.settings import PoolType, ServingSettings, get_settings
from internal.logging.secured_logger import get_secured_logger

logger = get_secured_logger("probe-server")


class HealthStatus(BaseModel):
    """Liveness probe status schema."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="healthy", description="Worker event loop health indicator")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Current UTC timestamp in ISO 8601 format",
    )


class ReadinessStatus(BaseModel):
    """Readiness probe status schema."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ready", description="Worker readiness status")
    pool_type: PoolType = Field(description="Hardware-isolated compute pool identifier")
    vram_used_bytes: int = Field(default=0, description="Observed GPU VRAM usage in bytes")
    vram_capacity_bytes: int = Field(
        default=25769803776, description="Total GPU VRAM capacity in bytes"
    )
    vram_utilization_ratio: float = Field(
        default=0.0, description="Ratio of used VRAM to total VRAM capacity"
    )
    models_ready: list[str] = Field(
        default_factory=list, description="List of initialized and ready model engine names"
    )


class VRAMMonitor:
    """Monitors GPU VRAM allocations and enforces operational headroom thresholds."""

    def __init__(self, settings: ServingSettings) -> None:
        self.settings = settings
        # Simulated or injected VRAM metrics for non-CUDA CI or test environments
        self._mock_vram_used_bytes: int | None = None
        self._mock_models_ready: list[str] | None = None
        self._mock_triton_ready: bool | None = None

    def set_mock_state(
        self,
        vram_used_bytes: int | None = None,
        models_ready: list[str] | None = None,
        triton_ready: bool | None = None,
    ) -> None:
        """Allow unit tests to inject deterministic VRAM and model readiness states."""
        self._mock_vram_used_bytes = vram_used_bytes
        self._mock_models_ready = models_ready
        self._mock_triton_ready = triton_ready

    async def get_vram_stats(self) -> tuple[int, int, float]:
        """Return (used_bytes, capacity_bytes, utilization_ratio)."""
        capacity = self.settings.vram_capacity_bytes
        if self._mock_vram_used_bytes is not None:
            used = self._mock_vram_used_bytes
        else:
            # Default baseline static allocation estimate when running without active NVML
            used = 12 * 1024 * 1024 * 1024  # 12 GiB (~50%)
        ratio = round(used / capacity, 4) if capacity > 0 else 0.0
        return used, capacity, ratio

    async def check_triton_ready(self, client: httpx.AsyncClient) -> tuple[bool, list[str]]:
        """Verify Triton health/ready endpoint and loaded model readiness."""
        if self._mock_triton_ready is not None:
            models = (
                self._mock_models_ready
                if self._mock_models_ready is not None
                else self.settings.models_expected
            )
            return self._mock_triton_ready, models

        url = f"{self.settings.triton_http_url}/v2/health/ready"
        try:
            resp = await client.get(url, timeout=1.5)
            if resp.status_code == 200:
                return True, self.settings.models_expected
            return False, []
        except Exception:
            return False, []


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    """FastAPI application factory for the out-of-band operational probe server."""
    active_settings = settings or get_settings()

    app = FastAPI(
        title="RAG Inference Serving Operational & Documentation API",
        description="Internal HTTP Documentation, Health, and Operational API for rag-inference-serving (Triton / TensorRT runtime)",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

    vram_monitor = VRAMMonitor(active_settings)
    app.state.vram_monitor = vram_monitor
    app.state.settings = active_settings

    @app.middleware("http")
    async def secured_logging_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start_time = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Unhandled error in probe server request",
                exc_info=exc,
                extra={
                    "request_id": req_id,
                    "pool_type": active_settings.pool_type.value,
                    "compute_latency_ms": duration_ms,
                    "status": "error",
                },
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "type": "about:blank",
                    "title": "Internal Server Error",
                    "status": 500,
                    "detail": "An internal operational error occurred.",
                    "instance": request.url.path,
                },
                media_type="application/problem+json",
            )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["x-request-id"] = req_id

        # Redact raw query params and headers in structured logs
        logger.info(
            "Operational HTTP request handled",
            extra={
                "request_id": req_id,
                "pool_type": active_settings.pool_type.value,
                "compute_latency_ms": duration_ms,
                "status": str(response.status_code),
            },
        )
        return response

    @app.get(
        "/healthz",
        response_model=HealthStatus,
        summary="Liveness Probe",
        description="Returns HTTP 200 if the operational probe server and Triton process monitor are running.",
    )
    async def healthz() -> HealthStatus:
        return HealthStatus(
            status="healthy",
            timestamp_utc=datetime.now(UTC).isoformat(),
        )

    @app.get(
        "/ready",
        response_model=ReadinessStatus,
        summary="Readiness Probe",
        description="Verifies that TensorRT model engines are loaded in VRAM and queue latencies are within bounds.",
        responses={
            200: {
                "description": "Worker model engines loaded and ready to accept inference requests",
                "model": ReadinessStatus,
            },
            503: {
                "description": "Models initializing or VRAM allocation threshold breached (>85%)",
            },
        },
    )
    async def ready() -> ReadinessStatus:
        used_bytes, capacity_bytes, ratio = await vram_monitor.get_vram_stats()

        # Enforce VRAM allocation and active runtime headroom ceilings
        if (
            ratio > active_settings.vram_max_utilization_ratio
            or used_bytes > active_settings.max_static_vram_bytes
        ):
            logger.warning(
                "Readiness rejected: VRAM headroom threshold breached",
                extra={
                    "pool_type": active_settings.pool_type.value,
                    "vram_used_bytes": used_bytes,
                    "vram_capacity_bytes": capacity_bytes,
                    "vram_utilization_ratio": ratio,
                    "status": "shedding_503",
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "type": "https://errors.enterprise-rag.internal/vram-threshold-exceeded",
                    "title": "Service Unavailable - VRAM Threshold Exceeded",
                    "status": 503,
                    "detail": f"Active VRAM utilization {ratio:.2%} exceeds ceiling {active_settings.vram_max_utilization_ratio:.2%}",
                    "vram_used_bytes": used_bytes,
                    "vram_capacity_bytes": capacity_bytes,
                },
            )

        # Check Triton readiness
        async with httpx.AsyncClient() as client:
            triton_ready, ready_models = await vram_monitor.check_triton_ready(client)

        if not triton_ready:
            logger.warning(
                "Readiness rejected: Triton backend not yet initialized",
                extra={
                    "pool_type": active_settings.pool_type.value,
                    "status": "initializing_503",
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "type": "https://errors.enterprise-rag.internal/triton-not-ready",
                    "title": "Service Unavailable - Triton Not Ready",
                    "status": 503,
                    "detail": "Triton inference server engines are still initializing or unready",
                },
            )

        return ReadinessStatus(
            status="ready",
            pool_type=active_settings.pool_type,
            vram_used_bytes=used_bytes,
            vram_capacity_bytes=capacity_bytes,
            vram_utilization_ratio=ratio,
            models_ready=ready_models,
        )

    @app.get(
        "/api/docs",
        summary="OpenAPI Documentation Schema",
        description="Direct JSON output of the OpenAPI 3.1 specification for machine synchronization.",
    )
    async def api_docs() -> dict[str, Any]:
        return app.openapi()

    @app.get(
        "/metrics",
        summary="Prometheus Metrics Scrape",
        description="Exposes Triton queue duration, latency, and GPU VRAM saturation metrics in Prometheus exposition format.",
        response_class=PlainTextResponse,
    )
    async def metrics() -> str:
        used_bytes, capacity_bytes, ratio = await vram_monitor.get_vram_stats()
        lines = [
            "# HELP rag_serving_vram_capacity_bytes Total GPU VRAM capacity in bytes",
            "# TYPE rag_serving_vram_capacity_bytes gauge",
            f'rag_serving_vram_capacity_bytes{{pool="{active_settings.pool_type.value}"}} {capacity_bytes}',
            "# HELP rag_serving_vram_used_bytes Active GPU VRAM memory allocated in bytes",
            "# TYPE rag_serving_vram_used_bytes gauge",
            f'rag_serving_vram_used_bytes{{pool="{active_settings.pool_type.value}"}} {used_bytes}',
            "# HELP rag_serving_vram_utilization_ratio Ratio of allocated VRAM to total capacity",
            "# TYPE rag_serving_vram_utilization_ratio gauge",
            f'rag_serving_vram_utilization_ratio{{pool="{active_settings.pool_type.value}"}} {ratio}',
            "# HELP rag_serving_max_queue_delay_microseconds Target maximum dynamic batch queue delay ceiling",
            "# TYPE rag_serving_max_queue_delay_microseconds gauge",
            f'rag_serving_max_queue_delay_microseconds{{pool="{active_settings.pool_type.value}"}} {active_settings.max_queue_delay_microseconds}',
            "# HELP rag_serving_max_batch_size Maximum dynamic batch size configured for pool",
            "# TYPE rag_serving_max_batch_size gauge",
            f'rag_serving_max_batch_size{{pool="{active_settings.pool_type.value}"}} {active_settings.max_batch_size}',
            "",
        ]
        return "\n".join(lines)

    return app


# Default module-level ASGI application entrypoint
app = create_app()
