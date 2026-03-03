"""File loaders for KnowledgeRegistry."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from ai_arch_toolkit.toolkit.knowledge._registry import KnowledgeEntry, KnowledgeRegistry

_FORMAT_BY_EXT: dict[str, str] = {
    ".txt": "text",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def load_text(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load a plain text file into the registry."""
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return registry.register(key, content, format="text", source=str(p), **kw)


def load_json(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load and validate a JSON file, storing formatted JSON."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = json.loads(raw)  # validates
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return registry.register(key, content, format="json", source=str(p), **kw)


def load_toml(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load and validate a TOML file, storing raw TOML text."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    tomllib.loads(raw)  # validates
    return registry.register(key, raw, format="toml", source=str(p), **kw)


def load_markdown(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load a Markdown file into the registry."""
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return registry.register(key, content, format="markdown", source=str(p), **kw)


def load_yaml(
    registry: KnowledgeRegistry,
    key: str,
    path: str | Path,
    **kw: Any,
) -> KnowledgeEntry:
    """Load and validate a YAML file. Requires pyyaml."""
    try:
        import yaml
    except ImportError:
        msg = "pyyaml is required for YAML loading: pip install pyyaml"
        raise ImportError(msg) from None

    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    yaml.safe_load(raw)  # validates
    return registry.register(key, raw, format="yaml", source=str(p), **kw)


_LOADER_BY_EXT: dict[str, Any] = {
    ".txt": load_text,
    ".json": load_json,
    ".toml": load_toml,
    ".md": load_markdown,
    ".yaml": load_yaml,
    ".yml": load_yaml,
}


def load_directory(
    registry: KnowledgeRegistry,
    directory: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    prefix: str = "",
    extensions: set[str] | None = None,
    category: str = "",
    tags: tuple[str, ...] = (),
) -> int:
    """Load files from a directory into the registry.

    Returns the number of entries loaded.
    """
    d = Path(directory)
    allowed_exts = extensions or set(_FORMAT_BY_EXT.keys())

    if recursive:
        files = sorted(d.rglob(pattern), key=lambda p: p.name)
    else:
        files = sorted(d.glob(pattern), key=lambda p: p.name)

    # Filter to files with known extensions
    files = [f for f in files if f.is_file() and f.suffix in allowed_exts]

    # Check for stem collisions in non-recursive mode
    if not recursive:
        stems: dict[str, Path] = {}
        for f in files:
            if f.stem in stems:
                msg = f"Stem collision: {f.stem!r} from {stems[f.stem]} and {f}"
                raise ValueError(msg)
            stems[f.stem] = f

    count = 0
    for f in files:
        if recursive:
            rel = f.relative_to(d)
            key = prefix + str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
        else:
            key = prefix + f.stem

        loader = _LOADER_BY_EXT.get(f.suffix)
        if loader:
            loader(registry, key, f, category=category, tags=tags)
            count += 1

    return count
