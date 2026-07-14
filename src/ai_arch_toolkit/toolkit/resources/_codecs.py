"""Built-in codecs for turning resource bytes into text and structured data."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from typing import Any, Protocol

from ai_arch_toolkit.toolkit.resources._errors import ResourceDecodeError
from ai_arch_toolkit.toolkit.resources._types import ResourceRef


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedResource:
    """Output of a resource codec."""

    data: Any
    text: str | None


class ResourceCodec(Protocol):
    """Decode raw bytes for one or more media types."""

    name: str

    def decode(self, raw: bytes, ref: ResourceRef) -> DecodedResource:
        """Decode resource bytes."""
        ...


def _decode_text(raw: bytes, ref: ResourceRef) -> str:
    try:
        return raw.decode(ref.encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise ResourceDecodeError(
            f"could not decode {ref.uri!r} using {ref.encoding!r}: {exc}"
        ) from exc


class TextCodec:
    """Decode plain text and Markdown without parsing their structure."""

    name = "text"

    def decode(self, raw: bytes, ref: ResourceRef) -> DecodedResource:
        text = _decode_text(raw, ref)
        return DecodedResource(data=text, text=text)


class JsonCodec:
    """Decode JSON into Python values while preserving source text."""

    name = "json"

    def decode(self, raw: bytes, ref: ResourceRef) -> DecodedResource:
        text = _decode_text(raw, ref)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ResourceDecodeError(f"invalid JSON in {ref.uri!r}: {exc}") from exc
        return DecodedResource(data=data, text=text)


class TomlCodec:
    """Decode TOML into Python values while preserving source text."""

    name = "toml"

    def decode(self, raw: bytes, ref: ResourceRef) -> DecodedResource:
        text = _decode_text(raw, ref)
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ResourceDecodeError(f"invalid TOML in {ref.uri!r}: {exc}") from exc
        return DecodedResource(data=data, text=text)


class YamlCodec:
    """Decode YAML with safe loading when PyYAML is installed."""

    name = "yaml"

    def decode(self, raw: bytes, ref: ResourceRef) -> DecodedResource:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required for YAML resources: pip install 'ai-arch-toolkit[yaml]'"
            ) from None
        text = _decode_text(raw, ref)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ResourceDecodeError(f"invalid YAML in {ref.uri!r}: {exc}") from exc
        return DecodedResource(data=data, text=text)


class BinaryCodec:
    """Keep binary resources as bytes."""

    name = "binary"

    def decode(self, raw: bytes, ref: ResourceRef) -> DecodedResource:
        return DecodedResource(data=raw, text=None)


__all__ = [
    "BinaryCodec",
    "DecodedResource",
    "JsonCodec",
    "ResourceCodec",
    "TextCodec",
    "TomlCodec",
    "YamlCodec",
]
