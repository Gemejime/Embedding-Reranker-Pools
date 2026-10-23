"""Unit tests verifying secured logging redaction and JSON output structure."""

import json
import logging
import unittest
from internal.logging.secured_logger import SecuredJSONFormatter, sanitize_value


class TestSecuredLogging(unittest.TestCase):
    def test_sanitize_forbidden_raw_fields(self) -> None:
        self.assertEqual(sanitize_value("query_text", "what is the capital of France?"), "[REDACTED len=30]")
        self.assertEqual(sanitize_value("chunk_text", "sensitive customer agreement"), "[REDACTED len=28]")
        self.assertEqual(sanitize_value("embedding", [0.12, 0.45, -0.89]), "[REDACTED count=3]")
        self.assertEqual(sanitize_value("token", "secret-token-123"), "[REDACTED len=16]")

    def test_sanitize_bearer_and_secrets(self) -> None:
        msg = "User authenticated with Bearer eyJhbGciOiJIUzI1NiJ9.test and api_key=super_secret_key"
        sanitized = sanitize_value("message", msg)
        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", sanitized)
        self.assertNotIn("super_secret_key", sanitized)
        self.assertIn("[REDACTED]", sanitized)

    def test_secured_json_formatter_output(self) -> None:
        formatter = SecuredJSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Inference completed for query",
            args=(),
            exc_info=None,
        )
        record.request_id = "req-12345"  # type: ignore[attr-defined]
        record.model_name = "dense_embedder_trt"  # type: ignore[attr-defined]
        record.pool_type = "online_embedding"  # type: ignore[attr-defined]
        record.batch_size = 1  # type: ignore[attr-defined]

        formatted_str = formatter.format(record)
        parsed = json.loads(formatted_str)

        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["logger"], "test_logger")
        self.assertEqual(parsed["request_id"], "req-12345")
        self.assertEqual(parsed["model_name"], "dense_embedder_trt")
        self.assertEqual(parsed["pool_type"], "online_embedding")
        self.assertEqual(parsed["batch_size"], 1)
        self.assertIn("timestamp_utc", parsed)


if __name__ == "__main__":
    unittest.main()
