"""Unit tests for BatchIngestionClient dynamic micro-batching and dual-transport."""

import io
import logging
import unittest
from unittest.mock import MagicMock, patch

from api.proto.v1.triton_stubs_pb2 import (
    InferOutputTensor,
    ModelInferResponse,
    SystemSharedMemoryRegisterResponse,
    SystemSharedMemoryUnregisterResponse,
)
from internal.client.batch_client import BatchIngestionClient
from internal.client.kserve_codec import pack_fp16_tensor
from internal.config.settings import PoolType
from internal.logging.secured_logger import SecuredJSONFormatter
from internal.shm.posix_shm import PodBoundaryViolationError


class TestBatchClient(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_pool = MagicMock()
        self.mock_batch_stub = MagicMock()
        self.mock_pool.get_stub.return_value = self.mock_batch_stub
        self.mock_pool.get_loopback_stub.return_value = self.mock_batch_stub

        self.client = BatchIngestionClient(
            pool=self.mock_pool,
            max_batch_size=256,
        )

        # Logging capture
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(SecuredJSONFormatter())
        self.logger = logging.getLogger("batch-client")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.handler)

    def test_empty_chunks_returns_empty(self) -> None:
        self.assertEqual(self.client.embed_chunks([]), [])
        self.assertEqual(self.mock_batch_stub.ModelInfer.call_count, 0)

    def test_micro_batching_exceeding_256(self) -> None:
        # 300 chunks -> batch 1: 256, batch 2: 44
        chunks = [f"document chunk {i}" for i in range(300)]

        out1 = pack_fp16_tensor([[0.1] * 768 for _ in range(256)])
        out2 = pack_fp16_tensor([[0.2] * 768 for _ in range(44)])

        resp1 = ModelInferResponse(
            model_name="batch_embedder_ensemble",
            outputs=[InferOutputTensor(name="embeddings", shape=[256, 768], datatype="FP16")],
            raw_output_contents=[out1],
        )
        resp2 = ModelInferResponse(
            model_name="batch_embedder_ensemble",
            outputs=[InferOutputTensor(name="embeddings", shape=[44, 768], datatype="FP16")],
            raw_output_contents=[out2],
        )
        self.mock_batch_stub.ModelInfer.side_effect = [resp1, resp2]

        vectors = self.client.embed_chunks(chunks, request_id="batch-req-001")
        self.assertEqual(len(vectors), 300)
        self.assertEqual(len(vectors[0]), 768)
        self.assertEqual(self.mock_batch_stub.ModelInfer.call_count, 2)

        # Secured logging check
        log_output = self.log_stream.getvalue()
        self.assertIn("batch-req-001", log_output)
        self.assertNotIn("document chunk 0", log_output)

    def test_remote_host_shm_raises_pod_boundary_violation(self) -> None:
        chunks = ["chunk 1", "chunk 2"]
        with self.assertRaises(PodBoundaryViolationError):
            self.client.embed_chunks(
                chunks,
                target_host="worker-pod-2.cluster.local",
                use_shm=True,
            )

    @patch("internal.client.batch_client.SharedMemoryRegion")
    def test_pod_local_shm_ipc_lifecycle(self, mock_shm_class: MagicMock) -> None:
        mock_region = MagicMock()
        mock_region.name = "rag_batch_in_test"
        mock_region.key = "/rag_batch_in_test"
        mock_shm_class.return_value = mock_region

        chunks = ["chunk 1", "chunk 2"]
        out_bytes = pack_fp16_tensor([[0.5] * 768, [0.6] * 768])

        self.mock_batch_stub.SystemSharedMemoryRegister.return_value = (
            SystemSharedMemoryRegisterResponse()
        )
        self.mock_batch_stub.SystemSharedMemoryUnregister.return_value = (
            SystemSharedMemoryUnregisterResponse()
        )
        self.mock_batch_stub.ModelInfer.return_value = ModelInferResponse(
            model_name="batch_embedder_ensemble",
            outputs=[InferOutputTensor(name="embeddings", shape=[2, 768], datatype="FP16")],
            raw_output_contents=[out_bytes],
        )

        vectors = self.client.embed_chunks(chunks, target_host="localhost", use_shm=True)
        self.assertEqual(len(vectors), 2)

        # Verify SHM lifecycle calls
        mock_region.allocate.assert_called_once()
        mock_region.write.assert_called_once()
        self.mock_batch_stub.SystemSharedMemoryRegister.assert_called_once()
        self.mock_batch_stub.ModelInfer.assert_called_once()
        self.mock_batch_stub.SystemSharedMemoryUnregister.assert_called_once()
        mock_region.release.assert_called_once()

    @patch("internal.client.batch_client.SharedMemoryRegion")
    def test_pod_local_shm_empty_output_raises_runtime_error(
        self, mock_shm_class: MagicMock
    ) -> None:
        mock_region = MagicMock()
        mock_region.name = "rag_batch_in_empty"
        mock_region.key = "/rag_batch_in_empty"
        mock_shm_class.return_value = mock_region

        self.mock_batch_stub.SystemSharedMemoryRegister.return_value = (
            SystemSharedMemoryRegisterResponse()
        )
        self.mock_batch_stub.SystemSharedMemoryUnregister.return_value = (
            SystemSharedMemoryUnregisterResponse()
        )
        # Empty raw_output_contents
        self.mock_batch_stub.ModelInfer.return_value = ModelInferResponse(
            model_name="batch_embedder_ensemble",
            outputs=[],
            raw_output_contents=[],
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.client.embed_chunks(["chunk 1"], target_host="localhost", use_shm=True)
        self.assertIn("Missing raw_output_contents", str(ctx.exception))
        mock_region.release.assert_called_once()

    def test_transient_failure_retries_and_reconnects(self) -> None:
        import grpc

        stub_old = MagicMock()
        stub_new = MagicMock()

        err = grpc.RpcError()
        err.code = lambda: grpc.StatusCode.UNAVAILABLE

        resp_batch_1 = ModelInferResponse(
            model_name="batch_embedder_ensemble",
            outputs=[InferOutputTensor(name="embeddings", shape=[2, 768], datatype="FP16")],
            raw_output_contents=[pack_fp16_tensor([[0.7] * 768, [0.8] * 768])],
        )
        resp_batch_2 = ModelInferResponse(
            model_name="batch_embedder_ensemble",
            outputs=[InferOutputTensor(name="embeddings", shape=[1, 768], datatype="FP16")],
            raw_output_contents=[pack_fp16_tensor([[0.9] * 768])],
        )

        stub_old.ModelInfer.side_effect = err
        stub_new.ModelInfer.side_effect = [resp_batch_1, resp_batch_2]

        # Initial stub is stub_old; after reconnect_channel, pool returns stub_new
        current_stub = stub_old

        def fake_get_stub(pool_type: PoolType) -> MagicMock:
            nonlocal current_stub
            return current_stub

        def fake_reconnect(pool_type: PoolType) -> None:
            nonlocal current_stub
            current_stub = stub_new

        self.mock_pool.get_stub.side_effect = fake_get_stub
        self.mock_pool.reconnect_channel.side_effect = fake_reconnect

        client = BatchIngestionClient(
            pool=self.mock_pool,
            max_batch_size=2,
            max_retries=2,
            backoff_seconds=0.001,
        )

        # 3 chunks -> 2 micro-batches (batch 1: 2 chunks, batch 2: 1 chunk)
        vectors = client.embed_chunks(["chunk 1", "chunk 2", "chunk 3"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(stub_old.ModelInfer.call_count, 1)
        self.assertEqual(stub_new.ModelInfer.call_count, 2)
        self.mock_pool.reconnect_channel.assert_called_once()

    @patch("internal.client.batch_client.SharedMemoryRegion")
    def test_shm_uses_dedicated_loopback_stub(self, mock_shm_class: MagicMock) -> None:
        mock_region = MagicMock()
        mock_region.name = "rag_batch_in_loopback"
        mock_region.key = "/rag_batch_in_loopback"
        mock_shm_class.return_value = mock_region

        loopback_stub = MagicMock()
        remote_stub = MagicMock()
        self.mock_pool.get_loopback_stub.return_value = loopback_stub
        self.mock_pool.get_stub.return_value = remote_stub

        out_bytes = pack_fp16_tensor([[0.5] * 768])
        loopback_stub.SystemSharedMemoryRegister.return_value = SystemSharedMemoryRegisterResponse()
        loopback_stub.SystemSharedMemoryUnregister.return_value = (
            SystemSharedMemoryUnregisterResponse()
        )
        loopback_stub.ModelInfer.return_value = ModelInferResponse(
            model_name="batch_embedder_ensemble",
            outputs=[InferOutputTensor(name="embeddings", shape=[1, 768], datatype="FP16")],
            raw_output_contents=[out_bytes],
        )

        self.client.embed_chunks(["chunk 1"], target_host="localhost", use_shm=True)
        self.mock_pool.get_loopback_stub.assert_called_once()
        self.mock_pool.get_stub.assert_not_called()
        loopback_stub.SystemSharedMemoryRegister.assert_called_once()
        loopback_stub.ModelInfer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
