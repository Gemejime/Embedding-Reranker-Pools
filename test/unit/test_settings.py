"""Unit tests for configuration settings and pool-specific calibrated invariants."""

import unittest
from unittest import mock

from internal.config.settings import (
    DEFAULT_VRAM_CAPACITY_BYTES,
    MAX_RUNTIME_VRAM_UTILIZATION_RATIO,
    MAX_STATIC_ENGINE_VRAM_BYTES,
    POOL_MAX_BATCH_SIZE,
    POOL_MAX_QUEUE_DELAY_US,
    PoolType,
    ServingSettings,
)


class TestServingSettings(unittest.TestCase):
    def test_online_embedding_defaults(self) -> None:
        settings = ServingSettings(pool_type=PoolType.ONLINE_EMBEDDING)
        self.assertEqual(settings.pool_type, PoolType.ONLINE_EMBEDDING)
        self.assertEqual(
            settings.max_queue_delay_microseconds,
            POOL_MAX_QUEUE_DELAY_US[PoolType.ONLINE_EMBEDDING],
        )
        self.assertEqual(settings.max_batch_size, POOL_MAX_BATCH_SIZE[PoolType.ONLINE_EMBEDDING])
        self.assertIn("online_embedder_ensemble", settings.models_expected)
        self.assertIn("dense_embedder_trt", settings.models_expected)

    def test_online_reranker_defaults(self) -> None:
        settings = ServingSettings(pool_type=PoolType.ONLINE_RERANKER)
        self.assertEqual(settings.pool_type, PoolType.ONLINE_RERANKER)
        self.assertEqual(settings.max_queue_delay_microseconds, 4000)
        self.assertEqual(settings.max_batch_size, 32)
        self.assertIn("online_reranker_ensemble", settings.models_expected)
        self.assertIn("cross_encoder_trt", settings.models_expected)

    def test_batch_ingestion_defaults(self) -> None:
        settings = ServingSettings(pool_type=PoolType.BATCH_INGESTION)
        self.assertEqual(settings.pool_type, PoolType.BATCH_INGESTION)
        self.assertEqual(settings.max_queue_delay_microseconds, 50000)
        self.assertEqual(settings.max_batch_size, 256)
        self.assertIn("batch_embedder_ensemble", settings.models_expected)

    def test_vram_and_headroom_constants(self) -> None:
        settings = ServingSettings()
        # 24 GiB total capacity
        self.assertEqual(settings.vram_capacity_bytes, 24 * 1024 * 1024 * 1024)
        self.assertEqual(settings.vram_capacity_bytes, DEFAULT_VRAM_CAPACITY_BYTES)
        # Static engine allocation ceiling <= 20 GiB (83.33%)
        self.assertEqual(settings.max_static_vram_bytes, 20 * 1024 * 1024 * 1024)
        self.assertEqual(settings.max_static_vram_bytes, MAX_STATIC_ENGINE_VRAM_BYTES)
        # Dynamic active runtime shedding ceiling at 85%
        self.assertEqual(settings.vram_max_utilization_ratio, 0.85)
        self.assertEqual(settings.vram_max_utilization_ratio, MAX_RUNTIME_VRAM_UTILIZATION_RATIO)

    def test_explicit_overrides_respected(self) -> None:
        settings = ServingSettings(
            pool_type=PoolType.ONLINE_EMBEDDING,
            max_queue_delay_microseconds=1500,
            max_batch_size=16,
        )
        self.assertEqual(settings.max_queue_delay_microseconds, 1500)
        self.assertEqual(settings.max_batch_size, 16)

    def test_environment_variable_overrides(self) -> None:
        env_vars = {
            "RAG_SERVING_POOL_TYPE": "online_reranker",
            "RAG_SERVING_SERVER_HOST": "127.0.0.1",
            "RAG_SERVING_SERVER_PORT": "9090",
            "RAG_SERVING_TRITON_HTTP_URL": "http://10.0.0.5:8000",
            "RAG_SERVING_TRITON_GRPC_URL": "10.0.0.5:8001",
            "RAG_SERVING_MAX_QUEUE_DELAY_MICROSECONDS": "3500",
            "RAG_SERVING_MAX_BATCH_SIZE": "16",
            "RAG_SERVING_VRAM_CAPACITY_BYTES": "25769803776",
            "RAG_SERVING_MAX_STATIC_VRAM_BYTES": "21474836480",
            "RAG_SERVING_VRAM_MAX_UTILIZATION_RATIO": "0.80",
            "RAG_SERVING_MODELS_EXPECTED": '["custom_ensemble", "custom_model"]',
            "RAG_SERVING_SHM_BASE_PATH": "/tmp/custom_shm",
        }
        with mock.patch.dict("os.environ", env_vars, clear=False):
            settings = ServingSettings()
            self.assertEqual(settings.pool_type, PoolType.ONLINE_RERANKER)
            self.assertEqual(settings.server_host, "127.0.0.1")
            self.assertEqual(settings.server_port, 9090)
            self.assertEqual(settings.triton_http_url, "http://10.0.0.5:8000")
            self.assertEqual(settings.triton_grpc_url, "10.0.0.5:8001")
            self.assertEqual(settings.max_queue_delay_microseconds, 3500)
            self.assertEqual(settings.max_batch_size, 16)
            self.assertEqual(settings.vram_capacity_bytes, 25769803776)
            self.assertEqual(settings.max_static_vram_bytes, 21474836480)
            self.assertEqual(settings.vram_max_utilization_ratio, 0.80)
            self.assertEqual(settings.models_expected, ["custom_ensemble", "custom_model"])
            self.assertEqual(settings.shm_base_path, "/tmp/custom_shm")


if __name__ == "__main__":
    unittest.main()
