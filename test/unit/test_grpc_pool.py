"""Unit tests for TritonGRPCChannelPool connection pooling."""

import threading
import unittest
from unittest.mock import MagicMock, patch

from api.proto.v1.triton_stubs_pb2 import ServerLiveResponse, ServerReadyResponse
from internal.client.grpc_pool import DEFAULT_GRPC_OPTIONS, TritonGRPCChannelPool
from internal.config.settings import PoolType, ServingSettings


class TestTritonGRPCChannelPool(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ServingSettings(
            online_embed_grpc_url="embed-host:8001",
            online_rerank_grpc_url="rerank-host:8001",
            batch_ingest_grpc_url="batch-host:8001",
        )

    def test_target_resolution(self) -> None:
        pool = TritonGRPCChannelPool(settings=self.settings)
        self.assertEqual(pool.get_target(PoolType.ONLINE_EMBEDDING), "embed-host:8001")
        self.assertEqual(pool.get_target(PoolType.ONLINE_RERANKER), "rerank-host:8001")
        self.assertEqual(pool.get_target(PoolType.BATCH_INGESTION), "batch-host:8001")
        pool.close()

    def test_custom_target_overrides(self) -> None:
        pool = TritonGRPCChannelPool(
            embedding_target="custom-embed:9001",
            reranker_target="custom-rerank:9002",
            batch_target="custom-batch:9003",
        )
        self.assertEqual(pool.get_target(PoolType.ONLINE_EMBEDDING), "custom-embed:9001")
        self.assertEqual(pool.get_target(PoolType.ONLINE_RERANKER), "custom-rerank:9002")
        self.assertEqual(pool.get_target(PoolType.BATCH_INGESTION), "custom-batch:9003")
        pool.close()

    @patch("grpc.insecure_channel")
    def test_lazy_channel_creation_and_options(self, mock_channel_fn: MagicMock) -> None:
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel

        pool = TritonGRPCChannelPool(settings=self.settings)
        self.assertEqual(len(pool._channels), 0)

        # Access channel
        chan = pool.get_channel(PoolType.ONLINE_EMBEDDING)
        self.assertEqual(chan, mock_channel)
        mock_channel_fn.assert_called_once_with("embed-host:8001", options=DEFAULT_GRPC_OPTIONS)

        # Repeated access returns cached channel without creating another
        chan2 = pool.get_channel(PoolType.ONLINE_EMBEDDING)
        self.assertEqual(chan2, mock_channel)
        self.assertEqual(mock_channel_fn.call_count, 1)

        pool.close()

    @patch("grpc.insecure_channel")
    def test_thread_safe_channel_instantiation(self, mock_channel_fn: MagicMock) -> None:
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel

        pool = TritonGRPCChannelPool(settings=self.settings)
        channels = []

        def worker() -> None:
            c = pool.get_channel(PoolType.ONLINE_RERANKER)
            channels.append(c)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(channels), 10)
        self.assertEqual(mock_channel_fn.call_count, 1)
        pool.close()

    @patch("grpc.insecure_channel")
    def test_clean_channel_closure_and_context_manager(self, mock_channel_fn: MagicMock) -> None:
        mock_chan_1 = MagicMock()
        mock_chan_2 = MagicMock()
        mock_channel_fn.side_effect = [mock_chan_1, mock_chan_2]

        with TritonGRPCChannelPool(settings=self.settings) as pool:
            pool.get_channel(PoolType.ONLINE_EMBEDDING)
            pool.get_channel(PoolType.ONLINE_RERANKER)

        mock_chan_1.close.assert_called_once()
        mock_chan_2.close.assert_called_once()
        self.assertTrue(pool._closed)

        with self.assertRaises(RuntimeError):
            pool.get_channel(PoolType.ONLINE_EMBEDDING)

    @patch("grpc.insecure_channel")
    def test_liveness_and_readiness_probes(self, mock_channel_fn: MagicMock) -> None:
        pool = TritonGRPCChannelPool(settings=self.settings)
        mock_stub = MagicMock()
        pool._stubs[PoolType.ONLINE_EMBEDDING] = mock_stub
        pool._channels[PoolType.ONLINE_EMBEDDING] = MagicMock()

        # Success case
        mock_stub.ServerLive.return_value = ServerLiveResponse(live=True)
        mock_stub.ServerReady.return_value = ServerReadyResponse(ready=True)

        self.assertTrue(pool.is_channel_ready(PoolType.ONLINE_EMBEDDING))
        self.assertTrue(pool.is_pool_ready(PoolType.ONLINE_EMBEDDING))

        # Failure case
        mock_stub.ServerLive.side_effect = Exception("Connection refused")
        self.assertFalse(pool.is_channel_ready(PoolType.ONLINE_EMBEDDING))

        pool.close()


if __name__ == "__main__":
    unittest.main()
