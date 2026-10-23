"""Unit tests verifying OpenAPI 3.1 specification schema validity and required endpoints."""

import json
import os
import unittest


class TestOpenApiSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema_path = os.path.join("api", "openapi", "openapi.json")
        self.assertTrue(os.path.isfile(self.schema_path), "OpenAPI specification must exist")
        with open(self.schema_path, encoding="utf-8") as f:
            self.schema = json.load(f)

    def test_openapi_version_and_info(self) -> None:
        self.assertTrue(
            self.schema.get("openapi", "").startswith("3.1"), "OpenAPI specification must be 3.1.x"
        )
        self.assertEqual(
            self.schema["info"]["title"], "RAG Inference Serving Operational & Documentation API"
        )

    def test_required_operational_endpoints_present(self) -> None:
        paths = self.schema.get("paths", {})
        expected_endpoints = ["/healthz", "/ready", "/api/docs", "/metrics"]
        for endpoint in expected_endpoints:
            self.assertIn(endpoint, paths, f"Missing required Doc API endpoint: {endpoint}")

    def test_readiness_schema_properties(self) -> None:
        components = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("ReadinessStatus", components)
        readiness = components["ReadinessStatus"]
        self.assertIn("pool_type", readiness.get("properties", {}))
        self.assertIn("models_ready", readiness.get("properties", {}))

    def test_schema_additional_properties_false(self) -> None:
        components = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("HealthStatus", components)
        self.assertIn("ReadinessStatus", components)
        self.assertFalse(components["HealthStatus"].get("additionalProperties", True))
        self.assertFalse(components["ReadinessStatus"].get("additionalProperties", True))


if __name__ == "__main__":
    unittest.main()
