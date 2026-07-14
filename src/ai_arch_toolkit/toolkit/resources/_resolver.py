"""Composable resource resolution facade."""

from __future__ import annotations

import mimetypes
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from ai_arch_toolkit.toolkit.resources._codecs import (
    BinaryCodec,
    JsonCodec,
    ResourceCodec,
    TextCodec,
    TomlCodec,
    YamlCodec,
)
from ai_arch_toolkit.toolkit.resources._errors import ResourceDecodeError, ResourceLoadError
from ai_arch_toolkit.toolkit.resources._loaders import (
    FileResourceLoader,
    PackageResourceLoader,
    ResourceLoader,
)
from ai_arch_toolkit.toolkit.resources._policy import ResourcePolicy
from ai_arch_toolkit.toolkit.resources._serializers import (
    ResourceSerializer,
    SerializerRegistry,
)
from ai_arch_toolkit.toolkit.resources._types import Resource, ResourceProvenance, ResourceRef

_MEDIA_TYPE_BY_EXTENSION = {
    ".json": "application/json",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".toml": "application/toml",
    ".txt": "text/plain",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


class ResourceResolver:
    """Resolve references using configurable origin loaders and content codecs."""

    __slots__ = ("_codecs", "_extension_media_types", "_loaders", "policy", "serializers")

    def __init__(self, *, policy: ResourcePolicy | None = None) -> None:
        self.policy = policy or ResourcePolicy()
        self.serializers = SerializerRegistry()
        self._loaders: dict[str, ResourceLoader] = {
            "": FileResourceLoader(),
            "file": FileResourceLoader(),
            "package": PackageResourceLoader(),
        }
        text = TextCodec()
        self._codecs: dict[str, ResourceCodec] = {
            "application/json": JsonCodec(),
            "application/octet-stream": BinaryCodec(),
            "application/toml": TomlCodec(),
            "application/x-yaml": YamlCodec(),
            "application/yaml": YamlCodec(),
            "text/markdown": text,
            "text/plain": text,
            "text/yaml": YamlCodec(),
        }
        self._extension_media_types = dict(_MEDIA_TYPE_BY_EXTENSION)

    def register_loader(self, scheme: str, loader: ResourceLoader) -> None:
        """Register or replace an origin loader for a URI scheme."""
        if not isinstance(scheme, str):
            raise TypeError("resource loader scheme must be a string")
        self._loaders[scheme.lower()] = loader

    def register_codec(
        self,
        media_type: str,
        codec: ResourceCodec,
        *,
        extensions: Iterable[str] = (),
    ) -> None:
        """Register or replace a codec and optional extension mappings."""
        if not isinstance(media_type, str) or not media_type:
            raise ValueError("resource codec media type must be a non-empty string")
        normalized = media_type.lower()
        self._codecs[normalized] = codec
        for extension in extensions:
            if not isinstance(extension, str) or not extension:
                raise ValueError("resource codec extensions must be non-empty strings")
            suffix = extension if extension.startswith(".") else f".{extension}"
            self._extension_media_types[suffix.lower()] = normalized

    def register_serializer(self, name: str, serializer: ResourceSerializer) -> None:
        """Register or replace a serializer for prompt-ready resource values."""
        self.serializers.register(name, serializer)

    def resolve(self, ref: ResourceRef | str | Path) -> Resource:
        """Load and decode a resource reference."""
        resolved_ref = ref if isinstance(ref, ResourceRef) else ResourceRef.from_path(ref)
        parsed = urlparse(resolved_ref.uri)
        scheme = parsed.scheme.lower()
        if scheme in {"http", "https"} and not self.policy.allow_remote:
            raise ResourceLoadError(
                f"remote resources are disabled by policy: {resolved_ref.uri!r}"
            )
        try:
            loader = self._loaders[scheme]
        except KeyError:
            choices = ", ".join(repr(value) for value in sorted(self._loaders))
            raise ResourceLoadError(
                f"no resource loader is registered for scheme {scheme!r}; available: {choices}"
            ) from None
        raw, canonical_uri = loader.load(resolved_ref, self.policy)
        media_type = resolved_ref.media_type or self._infer_media_type(resolved_ref.uri)
        self.policy.check_media_type(media_type, uri=resolved_ref.uri)
        try:
            codec = self._codecs[media_type]
        except KeyError:
            raise ResourceDecodeError(
                f"no resource codec is registered for media type {media_type!r} "
                f"({resolved_ref.uri!r})"
            ) from None
        decoded = codec.decode(raw, resolved_ref)
        provenance = ResourceProvenance(
            uri=resolved_ref.uri,
            resolved_uri=canonical_uri,
            loader=loader.name,
            codec=codec.name,
            media_type=media_type,
            byte_length=len(raw),
        )
        return Resource(
            ref=resolved_ref,
            raw=raw,
            media_type=media_type,
            data=decoded.data,
            text=decoded.text,
            provenance=provenance,
        )

    def load_directory(
        self,
        directory: str | Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        extensions: set[str] | None = None,
    ) -> tuple[Resource, ...]:
        """Resolve known files in deterministic relative-path order."""
        root = self.policy.check_path(Path(directory))
        paths = root.rglob(pattern) if recursive else root.glob(pattern)
        allowed = (
            {extension.lower() for extension in extensions}
            if extensions is not None
            else set(self._extension_media_types)
        )
        files = sorted(
            (path for path in paths if path.is_file() and path.suffix.lower() in allowed),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        return tuple(self.resolve(path) for path in files)

    def _infer_media_type(self, uri: str) -> str:
        parsed = urlparse(uri)
        suffix = Path(parsed.path or uri).suffix.lower()
        if suffix in self._extension_media_types:
            return self._extension_media_types[suffix]
        guessed, _encoding = mimetypes.guess_type(parsed.path or uri)
        return guessed or "application/octet-stream"


def load_resource(
    ref: ResourceRef | str | Path,
    *,
    policy: ResourcePolicy | None = None,
    resolver: ResourceResolver | None = None,
    media_type: str | None = None,
    encoding: str = "utf-8",
) -> Resource:
    """Load one resource using a default or supplied resolver."""
    if resolver is not None and policy is not None:
        raise ValueError("pass either resolver or policy, not both")
    active = resolver or ResourceResolver(policy=policy)
    resolved_ref = (
        ref
        if isinstance(ref, ResourceRef)
        else ResourceRef.from_path(ref, media_type=media_type, encoding=encoding)
    )
    if isinstance(ref, ResourceRef) and (media_type is not None or encoding != "utf-8"):
        raise ValueError("media_type and encoding overrides cannot be used with ResourceRef")
    return active.resolve(resolved_ref)


def load_resources(
    directory: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    extensions: set[str] | None = None,
    policy: ResourcePolicy | None = None,
    resolver: ResourceResolver | None = None,
) -> tuple[Resource, ...]:
    """Load known resources from a directory."""
    if resolver is not None and policy is not None:
        raise ValueError("pass either resolver or policy, not both")
    active = resolver or ResourceResolver(policy=policy)
    return active.load_directory(
        directory,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
    )


__all__ = ["ResourceResolver", "load_resource", "load_resources"]
