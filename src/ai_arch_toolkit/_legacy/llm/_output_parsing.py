"""Helpers for parsing common LLM text output formats."""

from __future__ import annotations

import json
import re
from dataclasses import is_dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any

_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\n(?P<body>.*?)```", re.DOTALL)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+[\.\)]))\s+")


def parse_json(text: str) -> dict[str, Any] | list[Any]:
    """Parse JSON from text, including fenced code blocks."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("No JSON found: empty text")
    try:
        parsed = json.loads(stripped)
        return _ensure_json_container(parsed)
    except JSONDecodeError:
        pass

    code = extract_code_block(text, language="json")
    if code is None:
        code = extract_code_block(text)
    if code is not None:
        try:
            parsed = json.loads(code.strip())
            return _ensure_json_container(parsed)
        except JSONDecodeError:
            pass

    snippet = _extract_json_snippet(text)
    if snippet is None:
        raise ValueError("No JSON object/array found in text")
    try:
        parsed = json.loads(snippet)
    except JSONDecodeError as exc:
        raise ValueError("Found JSON-like snippet but failed to parse JSON") from exc
    return _ensure_json_container(parsed)


def parse_json_as[T](text: str, target_type: type[T]) -> T:
    """Parse JSON text and coerce it into ``target_type`` when possible."""
    parsed = parse_json(text)

    if target_type in (dict, list):
        return parsed  # type: ignore[return-value]
    if is_dataclass(target_type):
        if not isinstance(parsed, dict):
            raise ValueError("Dataclass conversion requires a JSON object")
        return target_type(**parsed)  # type: ignore[misc,return-value]
    if isinstance(parsed, target_type):
        return parsed
    if isinstance(parsed, dict):
        return target_type(**parsed)  # type: ignore[misc,return-value]
    return target_type(parsed)  # type: ignore[misc,return-value]


def extract_code_block(text: str, language: str | None = None) -> str | None:
    """Extract first fenced code block, optionally filtering by language."""
    for match in _CODE_BLOCK_RE.finditer(text):
        block_lang = (match.group("lang") or "").strip().lower()
        if language is not None and block_lang != language.lower():
            continue
        return match.group("body")
    return None


def extract_list(text: str) -> list[str]:
    """Extract list-like items from bullet/numbered text."""
    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = _LIST_PREFIX_RE.sub("", line, count=1).strip()
        if item:
            items.append(item)
    return items


def _ensure_json_container(value: Any) -> dict[str, Any] | list[Any]:
    if isinstance(value, (dict, list)):
        return value
    raise ValueError("JSON payload must be an object or array")


def _extract_json_snippet(text: str) -> str | None:
    decoder = JSONDecoder()
    starts = "{["
    for i, ch in enumerate(text):
        if ch not in starts:
            continue
        try:
            parsed, end = decoder.raw_decode(text, idx=i)
        except JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return text[i:end]
    return None
