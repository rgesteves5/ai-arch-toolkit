"""Immutable resource contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceRef:
    """Reference to content that can be resolved by a resource loader."""

    uri: str
    media_type: str | None = None
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("ResourceRef.uri is required")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("ResourceRef.media_type must be a string or None")
        if not isinstance(self.encoding, str) or not self.encoding:
            raise ValueError("ResourceRef.encoding is required")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
        encoding: str = "utf-8",
    ) -> ResourceRef:
        """Create a reference for a local filesystem path."""
        return cls(uri=str(path), media_type=media_type, encoding=encoding)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceProvenance:
    """Diagnostics describing how a resource was obtained and decoded."""

    uri: str
    resolved_uri: str
    loader: str
    codec: str
    media_type: str
    byte_length: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Resource:
    """Loaded content with its raw, text, parsed, and provenance representations."""

    ref: ResourceRef
    raw: bytes = field(repr=False)
    media_type: str
    data: Any = field(compare=False, hash=False)
    text: str | None = field(default=None, compare=False, hash=False)
    fingerprint: str = ""
    provenance: ResourceProvenance | None = field(default=None, compare=False, hash=False)
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.raw, bytes):
            raise TypeError("Resource.raw must be bytes")
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("Resource.media_type is required")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("Resource.text must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Resource.metadata must be a mapping")
        if self.fingerprint and not isinstance(self.fingerprint, str):
            raise TypeError("Resource.fingerprint must be a string")
        if self.provenance is not None and not isinstance(self.provenance, ResourceProvenance):
            raise TypeError("Resource.provenance must be ResourceProvenance or None")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not self.fingerprint:
            digest = hashlib.sha256(self.raw).hexdigest()
            object.__setattr__(self, "fingerprint", f"sha256:{digest}")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        uri: str = "memory://text",
        media_type: str = "text/plain",
        metadata: Mapping[str, Any] | None = None,
    ) -> Resource:
        """Create an already-resolved in-memory text resource."""
        if not isinstance(text, str):
            raise TypeError("Resource.from_text text must be a string")
        raw = text.encode("utf-8")
        ref = ResourceRef(uri=uri, media_type=media_type)
        provenance = ResourceProvenance(
            uri=uri,
            resolved_uri=uri,
            loader="memory",
            codec="text",
            media_type=media_type,
            byte_length=len(raw),
        )
        return cls(
            ref=ref,
            raw=raw,
            media_type=media_type,
            data=text,
            text=text,
            provenance=provenance,
            metadata=metadata or {},
        )

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        uri: str = "memory://bytes",
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> Resource:
        """Create an already-resolved in-memory binary resource."""
        if not isinstance(raw, bytes):
            raise TypeError("Resource.from_bytes raw must be bytes")
        ref = ResourceRef(uri=uri, media_type=media_type)
        provenance = ResourceProvenance(
            uri=uri,
            resolved_uri=uri,
            loader="memory",
            codec="binary",
            media_type=media_type,
            byte_length=len(raw),
        )
        return cls(
            ref=ref,
            raw=raw,
            media_type=media_type,
            data=raw,
            provenance=provenance,
            metadata=metadata or {},
        )


__all__ = ["Resource", "ResourceProvenance", "ResourceRef"]
