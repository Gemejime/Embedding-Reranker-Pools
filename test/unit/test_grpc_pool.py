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

    @patch("grpc.insecure_channel")
    def test_multi_target_round_robin_resolution(self, mock_channel_fn: MagicMock) -> None:
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel

        pool = TritonGRPCChannelPool(
            batch_target=["10.0.0.1:8001", "10.0.0.2:8001", "10.0.0.3:8001"]
        )
        target = pool.get_target(PoolType.BATCH_INGESTION)
        self.assertEqual(target, "ipv4:10.0.0.1:8001,10.0.0.2:8001,10.0.0.3:8001")

        chan = pool.get_channel(PoolType.BATCH_INGESTION)
        self.assertEqual(chan, mock_channel)
        self.assertEqual(pool.get_active_channel_count(), 1)

        # Verify round_robin LB policy is passed in options
        call_kwargs = mock_channel_fn.call_args[1]
        options = call_kwargs["options"]
        self.assertIn(("grpc.lb_policy_name", "round_robin"), options)
        pool.close()

    @patch("grpc.insecure_channel")
    def test_dns_target_round_robin_resolution(self, mock_channel_fn: MagicMock) -> None:
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel

        pool = TritonGRPCChannelPool(
            embedding_target="dns:///rag-inference-online-embedding.rag-inference-serving.svc.cluster.local:8001"
        )
        target = pool.get_target(PoolType.ONLINE_EMBEDDING)
        self.assertEqual(
            target,
            "dns:///rag-inference-online-embedding.rag-inference-serving.svc.cluster.local:8001",
        )

        chan = pool.get_channel(PoolType.ONLINE_EMBEDDING)
        self.assertEqual(chan, mock_channel)
        call_kwargs = mock_channel_fn.call_args[1]
        options = call_kwargs["options"]
        self.assertIn(("grpc.lb_policy_name", "round_robin"), options)
        pool.close()

    def test_unprefixed_hostname_multi_target_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            TritonGRPCChannelPool(batch_target=["worker-1.local:8001", "worker-2.local:8001"])
        self.assertIn("must be IPv4 literals", str(ctx.exception))

    @patch("grpc.insecure_channel")
    def test_reconnect_channel_recycling(self, mock_channel_fn: MagicMock) -> None:
        mock_chan_old = MagicMock()
        mock_chan_new = MagicMock()
        mock_channel_fn.side_effect = [mock_chan_old, mock_chan_new]

        pool = TritonGRPCChannelPool(settings=self.settings)
        chan1 = pool.get_channel(PoolType.ONLINE_EMBEDDING)
        self.assertEqual(chan1, mock_chan_old)
        self.assertEqual(pool.get_active_channel_count(), 1)

        # Recycle channel after synthetic failure
        chan2 = pool.reconnect_channel(PoolType.ONLINE_EMBEDDING)
        self.assertEqual(chan2, mock_chan_new)
        mock_chan_old.close.assert_called_once()
        self.assertEqual(pool.get_active_channel_count(), 1)
        pool.close()

    @patch("grpc.secure_channel")
    @patch("grpc.ssl_channel_credentials")
    def test_secure_channel_with_tls(
        self, mock_creds_fn: MagicMock, mock_secure_channel_fn: MagicMock
    ) -> None:
        mock_creds = MagicMock()
        mock_creds_fn.return_value = mock_creds
        mock_channel = MagicMock()
        mock_secure_channel_fn.return_value = mock_channel

        tls_settings = ServingSettings(
            online_embed_grpc_url="embed-tls:8001",
            grpc_tls_enabled=True,
            grpc_server_name_override="triton.rag.internal",
        )
        pool = TritonGRPCChannelPool(settings=tls_settings)
        chan = pool.get_channel(PoolType.ONLINE_EMBEDDING)
        self.assertEqual(chan, mock_channel)

        mock_creds_fn.assert_called_once_with(
            root_certificates=None,
            private_key=None,
            certificate_chain=None,
        )
        mock_secure_channel_fn.assert_called_once()
        args, kwargs = mock_secure_channel_fn.call_args
        self.assertEqual(args[0], "embed-tls:8001")
        self.assertEqual(args[1], mock_creds)
        options = kwargs["options"]
        self.assertIn(("grpc.ssl_target_name_override", "triton.rag.internal"), options)
        pool.close()

    def test_default_grpc_options_idle_keepalive_disabled(self) -> None:
        opts_dict = dict(DEFAULT_GRPC_OPTIONS)
        self.assertEqual(opts_dict.get("grpc.keepalive_permit_without_calls"), 0)

    @patch("grpc.secure_channel")
    def test_ca_cert_read_failure_propagates(self, mock_secure_channel_fn: MagicMock) -> None:
        tls_settings = ServingSettings(
            _env_file=None,
            online_embed_grpc_url="embed-tls:8001",
            grpc_tls_enabled=True,
            grpc_ca_cert_path="/nonexistent/ca.crt",
        )
        pool = TritonGRPCChannelPool(settings=tls_settings)
        with self.assertRaises(FileNotFoundError):
            pool.get_channel(PoolType.ONLINE_EMBEDDING)
        pool.close()

    @patch("grpc.insecure_channel")
    def test_loopback_channel_and_stub(self, mock_channel_fn: MagicMock) -> None:
        mock_channel = MagicMock()
        mock_channel_fn.return_value = mock_channel

        pool = TritonGRPCChannelPool(settings=self.settings)
        self.assertIsNone(pool._loopback_channel)

        chan = pool.get_loopback_channel()
        self.assertEqual(chan, mock_channel)
        mock_channel_fn.assert_called_once_with(
            self.settings.triton_grpc_url, options=DEFAULT_GRPC_OPTIONS
        )

        stub = pool.get_loopback_stub()
        self.assertIsNotNone(stub)

        reconnected = pool.reconnect_loopback_channel()
        self.assertEqual(reconnected, mock_channel)
        mock_channel.close.assert_called_once()

        pool.close()

    @patch("grpc.secure_channel")
    def test_client_cert_key_read_failure_propagates(
        self, mock_secure_channel_fn: MagicMock
    ) -> None:
        tls_settings = ServingSettings(
            _env_file=None,
            online_embed_grpc_url="embed-tls:8001",
            grpc_tls_enabled=True,
            grpc_client_cert_path="/nonexistent/client.crt",
            grpc_client_key_path="/nonexistent/client.key",
        )
        pool = TritonGRPCChannelPool(settings=tls_settings)
        with self.assertRaises(FileNotFoundError):
            pool.get_channel(PoolType.ONLINE_EMBEDDING)
        pool.close()


if __name__ == "__main__":
    unittest.main()
