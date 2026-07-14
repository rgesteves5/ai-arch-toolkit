"""Resource origin loaders."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from ai_arch_toolkit.toolkit.resources._errors import ResourceError, ResourceLoadError
from ai_arch_toolkit.toolkit.resources._policy import ResourcePolicy
from ai_arch_toolkit.toolkit.resources._types import ResourceRef


class ResourceLoader(Protocol):
    """Load bytes for a resource reference."""

    name: str

    def load(self, ref: ResourceRef, policy: ResourcePolicy) -> tuple[bytes, str]:
        """Return resource bytes and a canonical URI."""
        ...


class FileResourceLoader:
    """Load local filesystem resources."""

    name = "file"

    def load(self, ref: ResourceRef, policy: ResourcePolicy) -> tuple[bytes, str]:
        parsed = urlparse(ref.uri)
        raw_path = unquote(parsed.path) if parsed.scheme == "file" else ref.uri
        path = policy.check_path(Path(raw_path))
        try:
            size = path.stat().st_size
            policy.check_size(size, uri=ref.uri)
            raw = path.read_bytes()
        except ResourceError:
            raise
        except OSError as exc:
            raise ResourceLoadError(f"could not read resource {ref.uri!r}: {exc}") from exc
        policy.check_size(len(raw), uri=ref.uri)
        return raw, str(path)


class PackageResourceLoader:
    """Load ``package://module/path`` resources through importlib.resources."""

    name = "package"

    def load(self, ref: ResourceRef, policy: ResourcePolicy) -> tuple[bytes, str]:
        parsed = urlparse(ref.uri)
        package = parsed.netloc
        resource_path = unquote(parsed.path.lstrip("/"))
        if not package or not resource_path:
            raise ResourceLoadError("package resources must use package://module/path syntax")
        if any(part in {"", ".", ".."} for part in resource_path.split("/")):
            raise ResourceLoadError("package resource paths cannot contain '.' or '..' segments")
        try:
            target = importlib.resources.files(package).joinpath(resource_path)
            raw = target.read_bytes()
        except (ImportError, FileNotFoundError, ModuleNotFoundError, OSError) as exc:
            raise ResourceLoadError(f"could not read package resource {ref.uri!r}: {exc}") from exc
        policy.check_size(len(raw), uri=ref.uri)
        return raw, ref.uri


__all__ = ["FileResourceLoader", "PackageResourceLoader", "ResourceLoader"]
