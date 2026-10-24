"""Unit tests for the out-of-band operational ASGI probe server."""

import unittest

from fastapi.testclient import TestClient

from internal.config.settings import PoolType, ServingSettings
from internal.server.probe_server import create_app


class TestProbeServer(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ServingSettings(pool_type=PoolType.ONLINE_EMBEDDING)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)
        self.vram_monitor = self.app.state.vram_monitor

    def test_healthz_endpoint(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp_utc", data)

    def test_ready_endpoint_healthy(self) -> None:
        # Inject healthy mock state: 12 GiB used (< 20 GiB ceiling, ~50% ratio < 85%), Triton ready
        self.vram_monitor.set_mock_state(
            vram_used_bytes=12 * 1024 * 1024 * 1024,
            models_ready=["online_embedder_ensemble", "dense_embedder_trt", "text_tokenizer"],
            triton_ready=True,
        )
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["pool_type"], "online_embedding")
        self.assertEqual(data["vram_used_bytes"], 12 * 1024 * 1024 * 1024)
        self.assertAlmostEqual(data["vram_utilization_ratio"], 0.5, places=2)
        self.assertIn("online_embedder_ensemble", data["models_ready"])

    def test_ready_endpoint_sheds_on_vram_ratio_breach(self) -> None:
        # Inject breached VRAM state: 22 GiB used (> 85% of 24 GiB = > 20.4 GiB)
        self.vram_monitor.set_mock_state(
            vram_used_bytes=22 * 1024 * 1024 * 1024,
            triton_ready=True,
        )
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        detail = response.json().get("detail", {})
        self.assertEqual(detail.get("status"), 503)
        self.assertIn("vram-threshold-exceeded", detail.get("type", ""))

    def test_ready_endpoint_sheds_on_triton_unready(self) -> None:
        # Inject healthy VRAM but Triton unready
        self.vram_monitor.set_mock_state(
            vram_used_bytes=10 * 1024 * 1024 * 1024,
            triton_ready=False,
        )
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        detail = response.json().get("detail", {})
        self.assertIn("triton-not-ready", detail.get("type", ""))

    def test_api_docs_endpoint(self) -> None:
        response = self.client.get("/api/docs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("paths", data)
        self.assertIn("/healthz", data["paths"])
        self.assertIn("/ready", data["paths"])

    def test_metrics_endpoint(self) -> None:
        self.vram_monitor.set_mock_state(
            vram_used_bytes=10 * 1024 * 1024 * 1024,
            triton_ready=True,
        )
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("rag_serving_vram_capacity_bytes", body)
        self.assertIn("rag_serving_vram_used_bytes", body)
        self.assertIn("rag_serving_vram_utilization_ratio", body)
        self.assertIn("rag_serving_max_queue_delay_microseconds", body)


if __name__ == "__main__":
    unittest.main()
