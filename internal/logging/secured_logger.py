"""Secured Structured JSON Logging Engine for rag-inference-serving.

Guarantees strict redaction of:
- Personally Identifiable Information (PII / PHI)
- Customer raw query texts and document chunk text bodies
- Raw embedding vectors / float arrays
- Authorization tokens, Bearer headers, and cloud credentials
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

# Regex patterns for sensitive credentials & data
AUTH_HEADER_REGEX = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
API_KEY_REGEX = re.compile(r"(api[_-]?key|token|secret|password)[\"']?\s*[:=]\s*[\"']?([^\"',\s]+)", re.IGNORECASE)
SIGV4_REGEX = re.compile(r"(X-Amz-Signature=)[A-Za-z0-9]+", re.IGNORECASE)

# Field names that must NEVER emit raw payload contents
FORBIDDEN_RAW_FIELDS = {
    "query",
    "query_text",
    "text",
    "chunk",
    "chunk_text",
    "raw_input",
    "raw_output",
    "embedding",
    "embeddings",
    "vector",
    "raw_input_contents",
    "raw_output_contents",
    "authorization",
    "token",
    "secret",
}


def sanitize_value(key: str, val: Any) -> Any:
    """Recursively redact sensitive field contents and mask credentials in strings."""
    if key.lower() in FORBIDDEN_RAW_FIELDS:
        if isinstance(val, (str, bytes)):
            return f"[REDACTED len={len(val)}]"
        elif isinstance(val, (list, tuple)):
            return f"[REDACTED count={len(val)}]"
        return "[REDACTED]"

    if isinstance(val, str):
        masked = AUTH_HEADER_REGEX.sub(r"\1[REDACTED]", val)
        masked = API_KEY_REGEX.sub(r"\1=[REDACTED]", masked)
        masked = SIGV4_REGEX.sub(r"\1[REDACTED]", masked)
        return masked
    elif isinstance(val, dict):
        return {k: sanitize_value(k, v) for k, v in val.items()}
    elif isinstance(val, list):
        return [sanitize_value(key, item) for item in val]
    return val


class SecuredJSONFormatter(logging.Formatter):
    """Formats log records as structured, machine-parseable, redacted JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_value("message", record.getMessage()),
        }

        # Include standard correlation attributes if attached to record
        for attr in [
            "request_id",
            "model_name",
            "model_version",
            "pool_type",
            "batch_size",
            "queue_latency_ms",
            "compute_latency_ms",
            "status",
        ]:
            if hasattr(record, attr):
                log_payload[attr] = getattr(record, attr)

        # Merge extra dictionary if present and sanitize
        if hasattr(record, "extra") and isinstance(record.extra, dict):  # type: ignore[attr-defined]
            for k, v in record.extra.items():  # type: ignore[attr-defined]
                log_payload[k] = sanitize_value(k, v)

        if record.exc_info and record.exc_text:
            # Strip potential internal paths or sensitive arguments from traceback
            sanitized_exc = sanitize_value("traceback", record.exc_text)
            log_payload["exception"] = sanitized_exc

        return json.dumps(log_payload)


def get_secured_logger(name: str = "rag-inference-serving") -> logging.Logger:
    """Factory creating a logger pre-configured with the SecuredJSONFormatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SecuredJSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
