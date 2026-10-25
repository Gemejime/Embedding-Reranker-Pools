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
        self.assertIn("/metrics", data["paths"])
        # Verify security schemes and /metrics optional security
        self.assertIn("securitySchemes", data.get("components", {}))
        self.assertIn("BearerAuth", data["components"]["securitySchemes"])
        self.assertIn("ApiKeyAuth", data["components"]["securitySchemes"])
        metrics_sec = data["paths"]["/metrics"]["get"].get("security", [])
        self.assertTrue(any("BearerAuth" in s for s in metrics_sec))
        self.assertTrue(any("ApiKeyAuth" in s for s in metrics_sec))
        self.assertIn({}, metrics_sec)

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

    def test_metrics_endpoint_with_api_key_auth(self) -> None:
        secured_settings = ServingSettings(
            pool_type=PoolType.ONLINE_EMBEDDING,
            api_key="operational-secret-token",
        )
        app = create_app(secured_settings)
        client = TestClient(app)

        # 1. Unauthenticated request rejected with 401 and WWW-Authenticate header
        res_unauth = client.get("/metrics")
        self.assertEqual(res_unauth.status_code, 401)
        self.assertEqual(res_unauth.headers.get("www-authenticate"), "Bearer")

        # 2. Invalid bearer token rejected with 401
        res_invalid = client.get("/metrics", headers={"Authorization": "Bearer wrong-token"})
        self.assertEqual(res_invalid.status_code, 401)
        self.assertEqual(res_invalid.headers.get("www-authenticate"), "Bearer")

        # 3. Non-ASCII header handled gracefully without TypeError
        res_non_ascii = client.get(
            "/metrics",
            headers=[(b"authorization", "Bearer 🔑-utf8-token-🚀".encode())],
        )
        self.assertEqual(res_non_ascii.status_code, 401)

        # 4. Valid Bearer token accepted with 200
        res_bearer = client.get(
            "/metrics", headers={"Authorization": "Bearer operational-secret-token"}
        )
        self.assertEqual(res_bearer.status_code, 200)

        # 5. Valid x-api-key header accepted with 200
        res_apikey = client.get("/metrics", headers={"x-api-key": "operational-secret-token"})
        self.assertEqual(res_apikey.status_code, 200)


if __name__ == "__main__":
    unittest.main()
