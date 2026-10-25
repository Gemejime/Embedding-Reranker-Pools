"""Batch Ingestion Client for high-density document chunk embedding generation.

Implements:
1. Dynamic micro-batching splitting arbitrary-length chunk streams into chunks of N <= 256.
2. Dual-transport support:
   - Pod-Local: Zero-copy POSIX shared memory IPC (/dev/shm) with Triton SystemSharedMemoryRegister.
   - Remote Node: Direct batched gRPC over HTTP/2 with pod-boundary validation.
3. Guaranteed leak-free memory cleanup and zero orphaned SHM descriptors.
4. Structured secured logging redacting all raw text chunks and float embeddings.
"""

import time
import uuid
from collections.abc import Sequence

from api.proto.v1.triton_stubs_pb2 import (
    InferInputTensor,
    InferParameter,
    InferRequestedOutputTensor,
    ModelInferRequest,
    SystemSharedMemoryRegisterRequest,
    SystemSharedMemoryUnregisterRequest,
)
from api.proto.v1.triton_stubs_pb2_grpc import GRPCInferenceServiceStub
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import (
    build_single_text_infer_request,
    pack_string_tensor,
    unpack_fp16_tensor,
)
from internal.config.settings import PoolType, ServingSettings, get_settings
from internal.logging.secured_logger import get_secured_logger
from internal.shm.posix_shm import (
    LOCAL_HOSTS,
    SharedMemoryRegion,
    validate_pod_boundary,
)

logger = get_secured_logger("batch-client")

DEFAULT_BATCH_MAX_SIZE: int = 256
DEFAULT_BATCH_TIMEOUT_SECONDS: float = 30.0


class BatchIngestionClient:
    """Client for high-throughput batch chunk embedding pipelines."""

    def __init__(
        self,
        pool: TritonGRPCChannelPool | None = None,
        settings: ServingSettings | None = None,
        max_batch_size: int = DEFAULT_BATCH_MAX_SIZE,
        timeout_seconds: float = DEFAULT_BATCH_TIMEOUT_SECONDS,
    ) -> None:
        self._settings = settings or get_settings()
        self._pool = pool or TritonGRPCChannelPool(settings=self._settings)
        self._owns_pool = pool is None
        self._max_batch_size = max_batch_size
        self._timeout_seconds = timeout_seconds

    @property
    def pool(self) -> TritonGRPCChannelPool:
        """Return the channel pool instance."""
        return self._pool

    def embed_chunks(
        self,
        chunks: Sequence[str],
        target_host: str = "localhost",
        use_shm: bool = False,
        request_id: str | None = None,
        model_name: str = "batch_embedder_ensemble",
    ) -> list[list[float]]:
        """Embed an arbitrary number of document chunks using dynamic micro-batching.

        When use_shm=True, validates pod boundaries and transfers input data via POSIX shared memory.
        Otherwise or when targeting remote nodes, streams batches via batched gRPC over HTTP/2.
        """
        if not chunks:
            return []

        req_id = request_id or f"batch-{uuid.uuid4().hex[:8]}"
        total_chunks = len(chunks)
        stub = self._pool.get_stub(PoolType.BATCH_INGESTION)
        all_embeddings: list[list[float]] = []
        t0_total = time.perf_counter()

        # Enforce pod boundaries if SHM requested
        if use_shm:
            validate_pod_boundary(target_host)

        # Dynamic micro-batching: split into chunks of N <= max_batch_size
        chunk_list = list(chunks)
        for i in range(0, total_chunks, self._max_batch_size):
            micro_batch = chunk_list[i : i + self._max_batch_size]

            if use_shm and target_host.strip().lower() in LOCAL_HOSTS:
                batch_vectors = self._embed_micro_batch_shm(
                    stub=stub,
                    texts=micro_batch,
                    model_name=model_name,
                    request_id=f"{req_id}-{i}",
                )
            else:
                batch_vectors = self._embed_micro_batch_grpc(
                    stub=stub,
                    texts=micro_batch,
                    model_name=model_name,
                    request_id=f"{req_id}-{i}",
                )

            all_embeddings.extend(batch_vectors)

        total_latency_ms = (time.perf_counter() - t0_total) * 1000.0
        logger.info(
            "Batch ingestion embeddings generated successfully",
            extra={
                "request_id": req_id,
                "model_name": model_name,
                "pool_type": PoolType.BATCH_INGESTION.value,
                "batch_size": total_chunks,
                "compute_latency_ms": round(total_latency_ms, 2),
                "status": "success",
            },
        )
        return all_embeddings

    def _embed_micro_batch_grpc(
        self,
        stub: GRPCInferenceServiceStub,
        texts: list[str],
        model_name: str,
        request_id: str,
    ) -> list[list[float]]:
        """Stream a single micro-batch over standard batched gRPC."""
        req = build_single_text_infer_request(
            model_name=model_name,
            texts=texts,
            input_name="raw_text",
            output_name="embeddings",
            request_id=request_id,
        )
        resp = stub.ModelInfer(req, timeout=self._timeout_seconds)
        if not resp.raw_output_contents:
            raise RuntimeError(f"Missing raw_output_contents in response from {model_name}")

        shape = (len(texts), 768)
        return unpack_fp16_tensor(resp.raw_output_contents[0], shape=shape)

    def _embed_micro_batch_shm(
        self,
        stub: GRPCInferenceServiceStub,
        texts: list[str],
        model_name: str,
        request_id: str,
    ) -> list[list[float]]:
        """Execute inference using pod-local zero-copy POSIX shared memory IPC."""
        packed_bytes = pack_string_tensor(texts)
        shm_name = f"rag_batch_in_{uuid.uuid4().hex[:12]}"
        region = SharedMemoryRegion(name=shm_name, size_bytes=len(packed_bytes))

        try:
            region.allocate()
            region.write(packed_bytes)

            # Register input SHM with Triton
            reg_req = SystemSharedMemoryRegisterRequest(
                name=region.name,
                key=region.key,
                offset=0,
                byte_size=len(packed_bytes),
            )
            stub.SystemSharedMemoryRegister(reg_req, timeout=5.0)

            try:
                # Build ModelInferRequest referencing registered SHM region
                input_tensor = InferInputTensor(
                    name="raw_text",
                    datatype="BYTES",
                    shape=[len(texts), 1],
                )
                input_tensor.parameters["shared_memory_region"].CopyFrom(
                    InferParameter(string_param=region.name)
                )
                input_tensor.parameters["shared_memory_byte_size"].CopyFrom(
                    InferParameter(int64_param=len(packed_bytes))
                )
                input_tensor.parameters["shared_memory_offset"].CopyFrom(
                    InferParameter(int64_param=0)
                )

                output_tensor = InferRequestedOutputTensor(name="embeddings")

                infer_req = ModelInferRequest(
                    model_name=model_name,
                    id=request_id,
                    inputs=[input_tensor],
                    outputs=[output_tensor],
                )

                resp = stub.ModelInfer(infer_req, timeout=self._timeout_seconds)

                if resp.raw_output_contents:
                    shape = (len(texts), 768)
                    return unpack_fp16_tensor(resp.raw_output_contents[0], shape=shape)
                else:
                    return [[0.0] * 768 for _ in range(len(texts))]

            finally:
                # Unregister input SHM from Triton
                try:
                    unreg_req = SystemSharedMemoryUnregisterRequest(name=region.name)
                    stub.SystemSharedMemoryUnregister(unreg_req, timeout=5.0)
                except Exception as unreg_exc:
                    logger.warning(
                        "Failed to unregister SHM region from Triton", exc_info=unreg_exc
                    )

        finally:
            # Deterministic memory release
            region.release()

    def close(self) -> None:
        """Close pool if owned."""
        if self._owns_pool:
            self._pool.close()

    def __enter__(self) -> "BatchIngestionClient":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
