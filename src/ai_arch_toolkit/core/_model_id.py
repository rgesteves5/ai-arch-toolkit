"""One grammar for model ids: an id's own entry, a dated snapshot of one, a family, or unknown.

Every table keyed by model id — prices, provider routing, the local tokenizer, the adapters'
per-model profiles — resolves through :func:`lookup`. Nothing else in ``core`` compares a model
id's prefix, so a variant (``o3-pro``, ``gpt-4o-audio-preview``) never inherits the entry of the
id it starts with.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

__all__ = ["ModelMatch", "family", "lookup", "snapshot_base"]

# Snapshot suffixes the providers append to a model id: claude-haiku-4-5-20251001 (Anthropic),
# gpt-4.1-2025-04-14 (OpenAI), claude-opus-4-5@20251101 (Vertex AI), gemini-2.0-flash-001 and
# gemini-2.5-flash-preview-05-20 / -preview-09-2025 (Gemini), grok-4-0709 and gpt-3.5-turbo-0125
# (xAI and older OpenAI), and the moving -latest pointer.
_SNAPSHOT = re.compile(
    r"(?:-\d{8}|-\d{4}-\d{2}-\d{2}|@\d{8}|-\d{3,4}|-preview-\d{2}-\d{2}(?:\d{2})?|-latest)$"
)


@dataclass(frozen=True, slots=True)
class ModelMatch[T]:
    """How a model id found its entry: its own, a dated snapshot of one, or a family's."""

    kind: Literal["exact", "snapshot", "family"]
    key: str
    value: T


def snapshot_base(model: str) -> str | None:
    """The id without its snapshot suffix, or ``None`` when it carries none."""
    match = _SNAPSHOT.search(model)
    return model[: match.start()] if match and match.start() > 0 else None


def _family_key(model: str, families: Mapping[str, object]) -> str | None:
    return max((prefix for prefix in families if model.startswith(prefix)), key=len, default=None)


def family[T](model: str, families: Mapping[str, T]) -> T | None:
    """The value of the longest family prefix of ``model``, or ``None``."""
    key = _family_key(model, families)
    return families[key] if key is not None else None


def lookup[T](
    model: str,
    exact: Mapping[str, T],
    families: Mapping[str, T] | None = None,
) -> ModelMatch[T] | None:
    """Resolve ``model`` against a table, in a fixed order; ``None`` means unknown.

    The id's own entry wins; then the entry of the id without its snapshot suffix; then, only
    when the caller passes ``families``, the longest family prefix.
    """
    if model in exact:
        return ModelMatch("exact", model, exact[model])
    base = snapshot_base(model)
    if base is not None and base in exact:
        return ModelMatch("snapshot", base, exact[base])
    if families and (key := _family_key(model, families)) is not None:
        return ModelMatch("family", key, families[key])
    return None
