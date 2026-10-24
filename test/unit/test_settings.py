"""Unit tests for configuration settings and pool-specific calibrated invariants."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
