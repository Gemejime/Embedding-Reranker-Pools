"""Chaos Test: Spot Node Preemption, Eviction Resilience & SHM Reclamation.

Validates:
1. Kubernetes spot instance preemption / eviction (SIGTERM / node drain / UNAVAILABLE).
2. Batch ingestion client retry logic and transparent connection failover.
3. Task re-queueing without chunk loss across eviction events.
4. Clean /dev/shm resource reclamation preventing orphaned shared memory segments or kernel leaks.
"""

import unittest
from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc

from api.proto.v1.triton_stubs_pb2 import (
    InferOutputTensor,
    ModelInferRequest,
    ModelInferResponse,
    ServerLiveRequest,
    ServerLiveResponse,
    ServerReadyRequest,
    ServerReadyResponse,
    SystemSharedMemoryRegisterRequest,
    SystemSharedMemoryRegisterResponse,
    SystemSharedMemoryUnregisterRequest,
    SystemSharedMemoryUnregisterResponse,
)
from api.proto.v1.triton_stubs_pb2_grpc import (
    GRPCInferenceServiceServicer,
    add_GRPCInferenceServiceServicer_to_server,
)
from internal.client.batch_client import BatchIngestionClient
from internal.client.grpc_pool import TritonGRPCChannelPool
from internal.client.kserve_codec import pack_fp16_tensor


class EvictableSpotTritonServicer(GRPCInferenceServiceServicer):
    """Mock Triton servicer simulating a Kubernetes spot worker pod undergoing termination."""

    def __init__(self) -> None:
        self.evicted = False
        self.evict_on_first_n_calls = 0
        self.call_count = 0

    def ServerLive(
        self, request: ServerLiveRequest, context: grpc.ServicerContext
    ) -> ServerLiveResponse:
        return ServerLiveResponse(live=not self.evicted)

    def ServerReady(
        self, request: ServerReadyRequest, context: grpc.ServicerContext
    ) -> ServerReadyResponse:
        return ServerReadyResponse(ready=not self.evicted)

    def SystemSharedMemoryRegister(
        self, request: SystemSharedMemoryRegisterRequest, context: grpc.ServicerContext
    ) -> SystemSharedMemoryRegisterResponse:
        if self.evicted:
            context.abort(
                grpc.StatusCode.UNAVAILABLE, "Pod terminating: spot eviction signal received"
            )
            return SystemSharedMemoryRegisterResponse()
        return SystemSharedMemoryRegisterResponse()

    def SystemSharedMemoryUnregister(
        self, request: SystemSharedMemoryUnregisterRequest, context: grpc.ServicerContext
    ) -> SystemSharedMemoryUnregisterResponse:
        return SystemSharedMemoryUnregisterResponse()

    def ModelInfer(
        self, request: ModelInferRequest, context: grpc.ServicerContext
    ) -> ModelInferResponse:
        self.call_count += 1
        if self.evicted or self.call_count <= self.evict_on_first_n_calls:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "Pod terminating: spot eviction signal received (SIGTERM)",
            )
            return ModelInferResponse()

        # Normal response
        batch_size = 32
        raw_out = pack_fp16_tensor([[0.05] * 768 for _ in range(batch_size)])
        return ModelInferResponse(
            model_name=request.model_name,
            outputs=[
                InferOutputTensor(name="embeddings", datatype="FP16", shape=[batch_size, 768])
            ],
            raw_output_contents=[raw_out],
        )


class TestSpotEviction(unittest.TestCase):
    server: grpc.Server
    servicer: EvictableSpotTritonServicer
    server_target: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        cls.servicer = EvictableSpotTritonServicer()
        add_GRPCInferenceServiceServicer_to_server(cls.servicer, cls.server)
        port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.server_target = f"127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop(grace=0)

    def setUp(self) -> None:
        self.servicer.evicted = False
        self.servicer.evict_on_first_n_calls = 0
        self.servicer.call_count = 0
        self.pool = TritonGRPCChannelPool(batch_target=self.server_target)
        self.client = BatchIngestionClient(
            pool=self.pool,
            max_batch_size=32,
            max_retries=3,
            backoff_seconds=0.01,
        )

    def tearDown(self) -> None:
        self.client.close()
        self.pool.close()

    def test_spot_eviction_failover_and_zero_chunk_loss(self) -> None:
        """Simulate spot node termination mid-batch and assert client reconnects without chunk loss."""
        chunks = [f"document chunk {i}" for i in range(32)]

        # First call aborts with UNAVAILABLE (eviction), second retry succeeds
        self.servicer.call_count = 0
        self.servicer.evict_on_first_n_calls = 1

        vectors = self.client.embed_chunks(chunks)
        # Assert all 32 embeddings were returned without any chunk loss
        self.assertEqual(len(vectors), 32)
        self.assertEqual(len(vectors[0]), 768)
        # Assert retry executed (attempt 1 failed with eviction, attempt 2 succeeded)
        self.assertEqual(self.servicer.call_count, 2)

    @patch("internal.client.batch_client.SharedMemoryRegion")
    def test_spot_eviction_shm_leak_free_reclamation(self, mock_shm_cls: MagicMock) -> None:
        """Verify POSIX shared memory resources are deterministically released during eviction."""
        mock_region = MagicMock()
        mock_region.name = "rag_eviction_test"
        mock_region.key = "/rag_eviction_test"
        mock_shm_cls.return_value = mock_region

        chunks = ["chunk 1", "chunk 2"]

        # Evict spot node
        self.servicer.evicted = True

        client = BatchIngestionClient(
            pool=self.pool,
            max_batch_size=32,
            max_retries=1,
            backoff_seconds=0.001,
        )

        with self.assertRaises(grpc.RpcError) as ctx:
            client.embed_chunks(chunks, target_host="localhost", use_shm=True)

        self.assertEqual(ctx.exception.code(), grpc.StatusCode.UNAVAILABLE)

        # Assert region.release() was called deterministically, preventing kernel memory leaks
        self.assertGreaterEqual(mock_region.release.call_count, 1)


if __name__ == "__main__":
    unittest.main()
