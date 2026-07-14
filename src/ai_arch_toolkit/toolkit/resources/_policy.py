"""Security and size policies for resource loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_arch_toolkit.toolkit.resources._errors import (
    ResourcePolicyError,
    ResourceTooLargeError,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourcePolicy:
    """Restrictions applied before a resource is read or decoded."""

    allowed_roots: tuple[Path, ...] = ()
    max_bytes: int = 5 * 1024 * 1024
    allow_remote: bool = False
    allow_symlinks: bool = True
    allow_absolute_paths: bool = True
    allowed_media_types: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        roots = tuple(Path(root).expanduser().resolve() for root in self.allowed_roots)
        object.__setattr__(self, "allowed_roots", roots)
        if (
            not isinstance(self.max_bytes, int)
            or isinstance(self.max_bytes, bool)
            or self.max_bytes <= 0
        ):
            raise ValueError("ResourcePolicy.max_bytes must be a positive integer")
        if not isinstance(self.allow_remote, bool):
            raise TypeError("ResourcePolicy.allow_remote must be a boolean")
        if not isinstance(self.allow_symlinks, bool):
            raise TypeError("ResourcePolicy.allow_symlinks must be a boolean")
        if not isinstance(self.allow_absolute_paths, bool):
            raise TypeError("ResourcePolicy.allow_absolute_paths must be a boolean")
        media_types = frozenset(self.allowed_media_types)
        if not all(isinstance(media_type, str) and media_type for media_type in media_types):
            raise ValueError("ResourcePolicy.allowed_media_types must contain non-empty strings")
        object.__setattr__(self, "allowed_media_types", media_types)

    def check_path(self, path: Path) -> Path:
        """Validate and return the canonical path."""
        expanded = path.expanduser()
        if not self.allow_absolute_paths and expanded.is_absolute():
            raise ResourcePolicyError(f"absolute resource paths are not allowed: {expanded}")
        if not self.allow_symlinks and expanded.is_symlink():
            raise ResourcePolicyError(f"symbolic links are not allowed: {expanded}")
        resolved = expanded.resolve()
        if self.allowed_roots and not any(
            resolved == root or resolved.is_relative_to(root) for root in self.allowed_roots
        ):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            raise ResourcePolicyError(
                f"resource path {resolved} is outside allowed roots: {roots}"
            )
        return resolved

    def check_size(self, size: int, *, uri: str) -> None:
        """Reject resources larger than the configured limit."""
        if size > self.max_bytes:
            raise ResourceTooLargeError(
                f"resource {uri!r} is {size} bytes; maximum is {self.max_bytes} bytes"
            )

    def check_media_type(self, media_type: str, *, uri: str) -> None:
        """Reject media types outside an optional allowlist."""
        if self.allowed_media_types and media_type not in self.allowed_media_types:
            allowed = ", ".join(sorted(self.allowed_media_types))
            raise ResourcePolicyError(
                f"resource {uri!r} has media type {media_type!r}; allowed: {allowed}"
            )


__all__ = ["ResourcePolicy"]
