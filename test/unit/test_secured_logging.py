"""Unit tests verifying secured logging redaction, schema conformity, and JSON output structure."""

import json
import logging
import sys
import unittest

from internal.logging.secured_logger import (
    SecuredJSONFormatter,
    get_secured_logger,
    sanitize_traceback,
    sanitize_value,
)


class TestSecuredLogging(unittest.TestCase):
    def test_sanitize_forbidden_raw_fields(self) -> None:
        self.assertEqual(
            sanitize_value("query_text", "what is the capital of France?"), "[REDACTED len=30]"
        )
        self.assertEqual(
            sanitize_value("chunk_text", "sensitive customer agreement"), "[REDACTED len=28]"
        )
        self.assertEqual(sanitize_value("embedding", [0.12, 0.45, -0.89]), "[REDACTED count=3]")
        self.assertEqual(sanitize_value("token", "secret-token-123"), "[REDACTED len=16]")
        self.assertEqual(sanitize_value("password", "super-secret-pass"), "[REDACTED len=17]")
        self.assertEqual(sanitize_value("api_key", "sk-1234567890"), "[REDACTED len=13]")
        self.assertEqual(sanitize_value("apiKey", "sk-abc"), "[REDACTED len=6]")
        self.assertEqual(sanitize_value("passwd", "mypassword"), "[REDACTED len=10]")

    def test_sanitize_tuples(self) -> None:
        raw_tuple = ("api_key=secret_in_tuple", [1.0, 2.0], "safe_item")
        sanitized = sanitize_value("data", raw_tuple)
        self.assertIsInstance(sanitized, tuple)
        self.assertNotIn("secret_in_tuple", sanitized[0])
        self.assertEqual(sanitized[2], "safe_item")

        tuple_of_secrets = sanitize_value("token", ("secret1", "secret2"))
        self.assertEqual(tuple_of_secrets, "[REDACTED count=2]")

    def test_sanitize_bearer_basic_and_presigned_urls(self) -> None:
        msg = (
            "User authenticated with Bearer eyJhbGciOiJIUzI1NiJ9.test and "
            "Basic dXNlcjpwYXNz and api_key=super_secret_key and "
            "url=https://s3.amazonaws.com/bucket/key.bin?X-Amz-Signature=abcd1234ef5678"
        )
        sanitized = sanitize_value("message", msg)
        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", sanitized)
        self.assertNotIn("dXNlcjpwYXNz", sanitized)
        self.assertNotIn("super_secret_key", sanitized)
        self.assertNotIn("abcd1234ef5678", sanitized)
        self.assertIn("[REDACTED]", sanitized)

    def test_message_content_payload_redaction(self) -> None:
        # Verify unstructured message strings cannot leak raw queries, chunks, or embedding arrays
        msg_query = "Processing incoming request for query: who directed the movie Inception?"
        sanitized_query = sanitize_value("message", msg_query)
        self.assertNotIn("who directed the movie Inception?", sanitized_query)
        self.assertIn("[REDACTED]", sanitized_query)

        msg_chunk = "Ingesting raw_chunk: internal financial earnings preview text"
        sanitized_chunk = sanitize_value("message", msg_chunk)
        self.assertNotIn("internal financial earnings preview text", sanitized_chunk)
        self.assertIn("[REDACTED]", sanitized_chunk)

        msg_vector = "Computed vector output: [0.1234, -0.5678, 0.9012, 0.3456]"
        sanitized_vector = sanitize_value("message", msg_vector)
        self.assertNotIn("0.1234", sanitized_vector)
        self.assertIn("[REDACTED embedding vector]", sanitized_vector)

    def test_traceback_sanitization(self) -> None:
        raw_tb = (
            "Traceback (most recent call last):\n"
            '  File "C:\\Users\\Administrator\\repo\\internal\\engine.py", line 42, in forward\n'
            "    cuda_ctx = 0x7f9a88001234\n"
            '  File "/var/app/internal/client.py", line 12, in call\n'
            '    connect("https://internal.gpu-node.local:8080/v2/infer")\n'
            "RuntimeError: CUDA memory failed at 0x00007ffec0123456\n"
        )
        sanitized = sanitize_traceback(raw_tb)
        self.assertNotIn("C:\\Users\\Administrator\\repo\\internal", sanitized)
        self.assertNotIn("/var/app/internal", sanitized)
        self.assertNotIn("0x7f9a88001234", sanitized)
        self.assertNotIn("0x00007ffec0123456", sanitized)
        self.assertNotIn("internal.gpu-node.local", sanitized)
        self.assertIn("[PATH]/engine.py", sanitized)
        self.assertIn("[PATH]/client.py", sanitized)
        self.assertIn("0x[REDACTED_GPU_ADDR]", sanitized)

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
        record.__dict__["request_id"] = "req-12345"
        record.__dict__["model_name"] = "dense_embedder_trt"
        record.__dict__["pool_type"] = "online_embedding"
        record.__dict__["batch_size"] = 1
        record.__dict__["queue_latency_ms"] = 1.85
        record.__dict__["compute_latency_ms"] = 4.20
        record.__dict__["status"] = "OK"

        formatted_str = formatter.format(record)
        parsed = json.loads(formatted_str)

        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["logger"], "test_logger")
        self.assertEqual(parsed["request_id"], "req-12345")
        self.assertEqual(parsed["model_name"], "dense_embedder_trt")
        self.assertEqual(parsed["pool_type"], "online_embedding")
        self.assertEqual(parsed["batch_size"], 1)
        self.assertEqual(parsed["queue_latency_ms"], 1.85)
        self.assertEqual(parsed["compute_latency_ms"], 4.20)
        self.assertEqual(parsed["status"], "OK")
        self.assertIn("timestamp_utc", parsed)

    def test_formatter_includes_all_mandatory_fields_by_default(self) -> None:
        formatter = SecuredJSONFormatter()
        # Record with no explicit correlation attributes attached
        record = logging.LogRecord(
            name="minimal_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=25,
            msg="Server warning event",
            args=(),
            exc_info=None,
        )
        formatted_str = formatter.format(record)
        parsed = json.loads(formatted_str)

        for mandatory_field in [
            "request_id",
            "model_name",
            "pool_type",
            "batch_size",
            "queue_latency_ms",
            "compute_latency_ms",
            "status",
        ]:
            self.assertIn(
                mandatory_field,
                parsed,
                f"Missing required logging schema field: {mandatory_field}",
            )

    def test_formatter_formats_exc_info_when_exc_text_is_absent(self) -> None:
        formatter = SecuredJSONFormatter()
        try:
            raise ValueError("GPU device error at 0x7f0123456789")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="exc_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=50,
            msg="Failed inference call",
            args=(),
            exc_info=exc_info,
        )
        # Ensure exc_text is None initially
        record.exc_text = None

        formatted_str = formatter.format(record)
        parsed = json.loads(formatted_str)

        self.assertIn("exception", parsed)
        self.assertNotIn("0x7f0123456789", parsed["exception"])
        self.assertIn("ValueError: GPU device error", parsed["exception"])

    def test_get_secured_logger_configures_existing_and_new_handlers(self) -> None:
        # Logger with pre-existing custom handler
        logger_name = "test_custom_handlers_logger"
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        custom_handler = logging.StreamHandler()
        custom_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(custom_handler)
        logger.propagate = True

        secured_logger = get_secured_logger(logger_name)
        self.assertFalse(secured_logger.propagate)
        self.assertEqual(len(secured_logger.handlers), 1)
        self.assertIsInstance(secured_logger.handlers[0].formatter, SecuredJSONFormatter)

        # Logger with no pre-existing handlers
        new_logger_name = "test_new_handlers_logger"
        new_logger = get_secured_logger(new_logger_name)
        self.assertFalse(new_logger.propagate)
        self.assertEqual(len(new_logger.handlers), 1)
        self.assertIsInstance(new_logger.handlers[0].formatter, SecuredJSONFormatter)


if __name__ == "__main__":
    unittest.main()
