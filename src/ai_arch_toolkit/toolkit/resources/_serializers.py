"""Deterministic serializers for selected resource values."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ai_arch_toolkit.toolkit.resources._errors import ResourceSerializationError


class ResourceSerializer(Protocol):
    """Serialize a selected resource value to prompt-ready text."""

    name: str

    def serialize(self, value: Any) -> str:
        """Serialize one value."""
        ...


class TextSerializer:
    """Serialize values as readable plain text."""

    name = "text"

    def serialize(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, Mapping):
            return "\n".join(f"{key}: {_inline(item)}" for key, item in value.items())
        if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            return "\n".join(_inline(item) for item in value)
        if isinstance(value, bytes):
            raise ResourceSerializationError("binary values cannot be serialized as text")
        return str(value)


class JsonSerializer:
    """Serialize values as deterministic human-readable JSON."""

    name = "json"

    def serialize(self, value: Any) -> str:
        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ResourceSerializationError(f"value is not JSON serializable: {exc}") from exc


class YamlSerializer:
    """Serialize values as safe YAML when PyYAML is installed."""

    name = "yaml"

    def serialize(self, value: Any) -> str:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required for YAML serialization: pip install 'ai-arch-toolkit[yaml]'"
            ) from None
        try:
            return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip("\n")
        except yaml.YAMLError as exc:
            raise ResourceSerializationError(f"value cannot be serialized as YAML: {exc}") from exc


class MarkdownSerializer:
    """Serialize common mappings and sequences as Markdown."""

    name = "markdown"

    def serialize(self, value: Any) -> str:
        return _markdown(value).rstrip()


def _inline(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _markdown(value: Any, *, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, Mapping | list | tuple):
                lines.append(f"{prefix}- **{key}**:")
                lines.append(_markdown(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}- **{key}**: {_inline(item)}")
        return "\n".join(lines)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        lines = []
        for item in value:
            if isinstance(item, Mapping | list | tuple):
                lines.append(f"{prefix}-")
                lines.append(_markdown(item, indent=indent + 1))
            else:
                lines.append(f"{prefix}- {_inline(item)}")
        return "\n".join(lines)
    return f"{prefix}{_inline(value)}"


class SerializerRegistry:
    """Isolated name-to-serializer registry with built-ins by default."""

    __slots__ = ("_serializers",)

    def __init__(self, serializers: Mapping[str, ResourceSerializer] | None = None) -> None:
        self._serializers: dict[str, ResourceSerializer] = {
            "json": JsonSerializer(),
            "markdown": MarkdownSerializer(),
            "text": TextSerializer(),
            "yaml": YamlSerializer(),
        }
        if serializers:
            for name, serializer in serializers.items():
                self.register(name, serializer)

    def register(self, name: str, serializer: ResourceSerializer) -> None:
        """Register or replace a serializer in this registry only."""
        if not isinstance(name, str) or not name:
            raise ValueError("resource serializer name must be a non-empty string")
        if not hasattr(serializer, "serialize"):
            raise TypeError("resource serializer must implement serialize(value)")
        self._serializers[name.lower()] = serializer

    def resolve(self, serializer: str | ResourceSerializer) -> ResourceSerializer:
        """Resolve a serializer object or registered name."""
        if not isinstance(serializer, str):
            if not hasattr(serializer, "serialize"):
                raise TypeError("resource serializer must implement serialize(value)")
            return serializer
        try:
            return self._serializers[serializer.lower()]
        except KeyError:
            choices = ", ".join(sorted(self._serializers))
            raise ResourceSerializationError(
                f"unknown resource serializer {serializer!r}; expected one of: {choices}"
            ) from None

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._serializers))


def serialize_resource_value(
    value: Any,
    *,
    as_format: str | ResourceSerializer = "text",
    registry: SerializerRegistry | None = None,
) -> str:
    """Serialize a value with a registered name or serializer object."""
    serializer = (registry or SerializerRegistry()).resolve(as_format)
    content = serializer.serialize(value)
    if not isinstance(content, str):
        raise ResourceSerializationError(
            f"resource serializer {type(serializer).__name__} must return a string"
        )
    return content


__all__ = [
    "JsonSerializer",
    "MarkdownSerializer",
    "ResourceSerializer",
    "SerializerRegistry",
    "TextSerializer",
    "YamlSerializer",
    "serialize_resource_value",
]
