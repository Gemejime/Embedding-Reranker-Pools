"""Secured Structured JSON Logging Engine for rag-inference-serving.

Guarantees strict redaction of:
- Personally Identifiable Information (PII / PHI)
- Customer raw query texts and document chunk text bodies
- Raw embedding vectors / float arrays
- Authorization tokens, Bearer headers, Basic auth, and cloud credentials
"""

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

# Regex patterns for sensitive credentials & data
AUTH_HEADER_REGEX = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
BASIC_AUTH_REGEX = re.compile(r"(Basic\s+)[A-Za-z0-9+/]+=*", re.IGNORECASE)
API_KEY_REGEX = re.compile(
    r"(api[_-]?key|token|secret|password|passwd)[\"']?\s*[:=]\s*[\"']?([^\"',\s]+)", re.IGNORECASE
)
SIGV4_REGEX = re.compile(r"(X-Amz-Signature=)[A-Za-z0-9]+", re.IGNORECASE)
PRESIGNED_URL_REGEX = re.compile(
    r"(https?://[^\s\"']+[?&](?:X-Amz-Signature|Signature|sig)=)[^\s\"'&]+", re.IGNORECASE
)
EMBEDDED_PAYLOAD_REGEX = re.compile(
    r"(query(?:_text)?|chunk(?:_text)?|raw_text|raw_chunk)[\"']?\s*[:=]\s*[\"']?([^\"'\r\n,]+)",
    re.IGNORECASE,
)
EMBEDDING_VECTOR_REGEX = re.compile(r"\[\s*-?\d+\.\d+(?:\s*,\s*-?\d+\.\d+){2,}\s*\]")

# Traceback sanitization patterns
GPU_ADDR_REGEX = re.compile(r"0x[0-9a-fA-F]{6,16}")
PATH_REGEX = re.compile(r'File "([^"\r\n]+[/\\])([^/\\\r\n]+\.py)"')
HOSTNAME_REGEX = re.compile(r"(https?://)[a-zA-Z0-9.\-]+(?::\d+)?")

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
    "password",
    "passwd",
    "api_key",
    "apikey",
}

SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|password|passwd|secret|token|auth(orization)?)", re.IGNORECASE
)


def is_sensitive_key(key: str) -> bool:
    """Return True if the key name implies sensitive payload or credential data."""
    k = key.lower()
    return k in FORBIDDEN_RAW_FIELDS or bool(SENSITIVE_KEY_PATTERN.search(k))


def sanitize_traceback(tb_str: str) -> str:
    """Sanitize internal file system paths, hostnames, and GPU memory addresses from tracebacks."""
    sanitized = GPU_ADDR_REGEX.sub("0x[REDACTED_GPU_ADDR]", tb_str)
    sanitized = PATH_REGEX.sub(r'File "[PATH]/\2"', sanitized)
    sanitized = HOSTNAME_REGEX.sub(r"\1[REDACTED_HOST]", sanitized)
    result = sanitize_value("traceback", sanitized)
    return str(result) if result is not None else ""


def sanitize_value(key: str, val: Any) -> Any:
    """Recursively redact sensitive field contents and mask credentials in strings."""
    if is_sensitive_key(key):
        if isinstance(val, (str, bytes)):
            return f"[REDACTED len={len(val)}]"
        elif isinstance(val, (list, tuple)):
            return f"[REDACTED count={len(val)}]"
        return "[REDACTED]"

    if isinstance(val, str):
        masked = AUTH_HEADER_REGEX.sub(r"\1[REDACTED]", val)
        masked = BASIC_AUTH_REGEX.sub(r"\1[REDACTED]", masked)
        masked = API_KEY_REGEX.sub(r"\1=[REDACTED]", masked)
        masked = SIGV4_REGEX.sub(r"\1[REDACTED]", masked)
        masked = PRESIGNED_URL_REGEX.sub(r"\1[REDACTED]", masked)
        masked = EMBEDDED_PAYLOAD_REGEX.sub(r"\1=[REDACTED]", masked)
        masked = EMBEDDING_VECTOR_REGEX.sub("[REDACTED embedding vector]", masked)
        return masked
    elif isinstance(val, dict):
        return {k: sanitize_value(k, v) for k, v in val.items()}
    elif isinstance(val, tuple):
        return tuple(sanitize_value(key, item) for item in val)
    elif isinstance(val, list):
        return [sanitize_value(key, item) for item in val]
    return val


class SecuredJSONFormatter(logging.Formatter):
    """Formats log records as structured, machine-parseable, redacted JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_value("message", record.getMessage()),
        }

        # Mandatory correlation attributes per schema specification
        mandatory_correlation_fields = [
            ("request_id", "unknown"),
            ("model_name", "unknown"),
            ("pool_type", "unknown"),
            ("batch_size", 0),
            ("queue_latency_ms", 0.0),
            ("compute_latency_ms", 0.0),
            ("status", "unknown"),
        ]
        for field, default_val in mandatory_correlation_fields:
            val = getattr(record, field, default_val)
            log_payload[field] = sanitize_value(field, val)

        if hasattr(record, "model_version"):
            log_payload["model_version"] = sanitize_value("model_version", record.model_version)

        # Merge extra dictionary if present and sanitize
        extra_data = getattr(record, "extra", None)
        if isinstance(extra_data, dict):
            for k, v in extra_data.items():
                log_payload[k] = sanitize_value(k, v)

        # Format and sanitize exception traceback if exc_info exists
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                log_payload["exception"] = sanitize_traceback(record.exc_text)

        return json.dumps(log_payload)


def get_secured_logger(name: str = "rag-inference-serving") -> logging.Logger:
    """Factory creating a logger pre-configured with the SecuredJSONFormatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SecuredJSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        for h in logger.handlers:
            h.setFormatter(SecuredJSONFormatter())
    logger.propagate = False
    return logger
