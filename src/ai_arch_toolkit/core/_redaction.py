"""Central redaction utilities for traces, logs, and runtime payloads."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from typing import Any, Literal

type TraceMode = Literal["metadata_only", "redacted", "full_debug"]

REDACTED = "[REDACTED]"


class RedactionMode(StrEnum):
    """Supported trace serialization modes."""

    METADATA_ONLY = "metadata_only"
    REDACTED = "redacted"
    FULL_DEBUG = "full_debug"


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Configuration for recursive payload redaction."""

    trace_mode: TraceMode = "redacted"
    replacement: str = REDACTED


class Redactor:
    """Redact sensitive values from nested runtime payloads."""

    def __init__(self, policy: RedactionPolicy | None = None) -> None:
        self.policy = policy or RedactionPolicy()

    def redact(self, value: Any) -> Any:
        """Return a redacted copy of ``value``."""
        if self.policy.trace_mode == "full_debug":
            return value
        return self._redact_value(value)

    def redact_text(self, text: str) -> str:
        """Redact secrets from a text value."""
        if self.policy.trace_mode == "full_debug":
            return text
        return _redact_text(text, self.policy.replacement)

    def _redact_value(self, value: Any) -> Any:
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, bytes):
            return self.policy.replacement
        if isinstance(value, tuple):
            return tuple(self._redact_value(item) for item in value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            for key, item in value.items():
                if _is_sensitive_key(key):
                    redacted[key] = self.policy.replacement
                else:
                    redacted[key] = self._redact_value(item)
            return redacted
        if is_dataclass(value) and not isinstance(value, type):
            return self._redact_value(asdict(value))
        return self.redact_text(repr(value))


def redact_text(text: str, policy: RedactionPolicy | None = None) -> str:
    """Redact secrets from text with the default redaction policy."""
    return Redactor(policy).redact_text(text)


def redact(value: Any, policy: RedactionPolicy | None = None) -> Any:
    """Redact secrets from a nested value with the default redaction policy."""
    return Redactor(policy).redact(value)


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace("-", "_")
    sensitive_fragments = (
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "client_secret",
        "connection_string",
        "database_url",
        "password",
        "private_key",
        "secret",
        "token",
    )
    return any(fragment in normalized for fragment in sensitive_fragments)


def _redact_text(text: str, replacement: str) -> str:
    result = text
    patterns = (
        (
            re.compile(
                r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                re.DOTALL,
            ),
            replacement,
        ),
        (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+"), f"Bearer {replacement}"),
        (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), replacement),
        (
            re.compile(r"\b(?:postgresql|postgres|mysql|mongodb|redis)://[^\s'\"<>]+"),
            replacement,
        ),
        (
            re.compile(
                r"(?im)^([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY)"
                r"[A-Z0-9_]*\s*=\s*)([^\n#]+)"
            ),
            rf"\1{replacement}",
        ),
        (
            re.compile(
                r"(?i)\b(api[_-]?key|token|secret|password|private[_-]?key)"
                r"(\s*[:=]\s*)(['\"]?)[^\s,'\"}]+"
            ),
            rf"\1\2\3{replacement}",
        ),
    )
    for pattern, substitute in patterns:
        result = pattern.sub(substitute, result)
    return result
