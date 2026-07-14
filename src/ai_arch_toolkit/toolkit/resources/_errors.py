"""Exceptions raised by resource loading and transformation."""

from __future__ import annotations


class ResourceError(Exception):
    """Base class for resource subsystem failures."""


class ResourceLoadError(ResourceError, OSError):
    """A resource could not be located or read."""


class ResourceDecodeError(ResourceError, ValueError):
    """A resource could not be decoded or parsed."""


class ResourcePolicyError(ResourceError, PermissionError):
    """A resource violates the active loading policy."""


class ResourceTooLargeError(ResourcePolicyError):
    """A resource exceeds the configured byte limit."""


class ResourceSelectorError(ResourceError, ValueError):
    """A selector is invalid or cannot resolve a resource fragment."""


class ResourceSerializationError(ResourceError, ValueError):
    """A selected resource value cannot be serialized as requested."""


__all__ = [
    "ResourceDecodeError",
    "ResourceError",
    "ResourceLoadError",
    "ResourcePolicyError",
    "ResourceSelectorError",
    "ResourceSerializationError",
    "ResourceTooLargeError",
]
