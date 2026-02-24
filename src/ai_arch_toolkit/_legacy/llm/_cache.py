"""Response caching middleware and in-memory cache backend."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import is_dataclass
from threading import Lock
from typing import Any, Protocol

from ai_arch_toolkit._legacy.llm._middleware import Request
from ai_arch_toolkit._legacy.llm._types import Response

logger = logging.getLogger(__name__)

_SHORT_CIRCUIT_RESULT_KEY = "middleware.short_circuit_result"
_CACHE_KEY_CONTEXT = "response_cache.key"
_CACHE_HIT_CONTEXT = "response_cache.hit"


class CacheBackend(Protocol):
    """Backend contract used by ResponseCache middleware."""

    def get(self, key: str) -> Response | None: ...

    def set(self, key: str, value: Response, ttl_seconds: float | None) -> None: ...


class InMemoryCacheBackend:
    """Thread-safe in-memory cache with optional TTL expiration."""

    def __init__(self, *, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, tuple[float | None, Response]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Response | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and self._clock() >= expires_at:
                del self._entries[key]
                return None
            return value

    def set(self, key: str, value: Response, ttl_seconds: float | None) -> None:
        expires_at: float | None = None
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                return
            expires_at = self._clock() + ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, value)


class ResponseCache:
    """Middleware that caches chat responses based on normalized request payload."""

    def __init__(
        self,
        *,
        backend: CacheBackend | None = None,
        ttl_seconds: float | None = 300.0,
        key_fn: Any | None = None,
    ) -> None:
        self._backend = backend or InMemoryCacheBackend()
        self._ttl_seconds = ttl_seconds
        self._key_fn = key_fn or _default_cache_key

    def before(self, request: Request) -> Request:
        if request.operation != "chat":
            return request
        key = self._key_fn(request)
        request.context[_CACHE_KEY_CONTEXT] = key
        cached = self._backend.get(key)
        if cached is None:
            request.context[_CACHE_HIT_CONTEXT] = False
            return request
        request.context[_CACHE_HIT_CONTEXT] = True
        request.context[_SHORT_CIRCUIT_RESULT_KEY] = cached
        logger.debug("Cache hit for provider=%s model=%s", request.provider, request.model)
        return request

    def after(self, request: Request, result: Any) -> Any:
        if request.operation != "chat" or not isinstance(result, Response):
            return result
        if request.context.get(_CACHE_HIT_CONTEXT):
            return result
        key = request.context.get(_CACHE_KEY_CONTEXT)
        if isinstance(key, str):
            self._backend.set(key, result, self._ttl_seconds)
            logger.debug(
                "Cached response for provider=%s model=%s",
                request.provider,
                request.model,
            )
        return result

    async def abefore(self, request: Request) -> Request:
        return self.before(request)

    async def aafter(self, request: Request, result: Any) -> Any:
        return self.after(request, result)


def _default_cache_key(request: Request) -> str:
    payload = {
        "operation": request.operation,
        "provider": request.provider,
        "model": request.model,
        "system": request.system,
        "messages": _normalize_for_hash(request.messages),
        "tools": _normalize_for_hash(request.tools),
        "json_schema": _normalize_for_hash(request.json_schema),
        "kwargs": _normalize_for_hash(request.kwargs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_for_hash(value: Any) -> Any:
    if is_dataclass(value):
        return {
            name: _normalize_for_hash(getattr(value, name)) for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        sorted_items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(k): _normalize_for_hash(v) for k, v in sorted_items}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
