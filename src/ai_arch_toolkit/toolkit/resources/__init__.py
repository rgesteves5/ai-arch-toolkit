"""Reusable resource loading, selection, and serialization."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.resources._codecs import DecodedResource, ResourceCodec
from ai_arch_toolkit.toolkit.resources._errors import (
    ResourceDecodeError,
    ResourceError,
    ResourceLoadError,
    ResourcePolicyError,
    ResourceSelectorError,
    ResourceSerializationError,
    ResourceTooLargeError,
)
from ai_arch_toolkit.toolkit.resources._loaders import ResourceLoader
from ai_arch_toolkit.toolkit.resources._policy import ResourcePolicy
from ai_arch_toolkit.toolkit.resources._resolver import (
    ResourceResolver,
    load_resource,
    load_resources,
)
from ai_arch_toolkit.toolkit.resources._selectors import (
    IdentitySelector,
    JsonPointer,
    LineRange,
    MarkdownHeading,
    NamedBlock,
    ResourceSelector,
    select_resource,
)
from ai_arch_toolkit.toolkit.resources._serializers import (
    JsonSerializer,
    MarkdownSerializer,
    ResourceSerializer,
    SerializerRegistry,
    TextSerializer,
    YamlSerializer,
    serialize_resource_value,
)
from ai_arch_toolkit.toolkit.resources._types import Resource, ResourceProvenance, ResourceRef

__all__ = [
    "DecodedResource",
    "IdentitySelector",
    "JsonPointer",
    "JsonSerializer",
    "LineRange",
    "MarkdownHeading",
    "MarkdownSerializer",
    "NamedBlock",
    "Resource",
    "ResourceCodec",
    "ResourceDecodeError",
    "ResourceError",
    "ResourceLoadError",
    "ResourceLoader",
    "ResourcePolicy",
    "ResourcePolicyError",
    "ResourceProvenance",
    "ResourceRef",
    "ResourceResolver",
    "ResourceSelector",
    "ResourceSelectorError",
    "ResourceSerializationError",
    "ResourceSerializer",
    "ResourceTooLargeError",
    "SerializerRegistry",
    "TextSerializer",
    "YamlSerializer",
    "load_resource",
    "load_resources",
    "select_resource",
    "serialize_resource_value",
]
