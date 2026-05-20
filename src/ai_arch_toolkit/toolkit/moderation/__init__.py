"""Content moderation — OpenAI API and LLM-based classifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from ai_arch_toolkit.toolkit.moderation._llm import LLMModerator
from ai_arch_toolkit.toolkit.moderation._middleware import ModerationMiddleware

if _TYPE_CHECKING:
    # Surfaced lazily via __getattr__ to keep the openai dep optional.
    from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

__all__ = ["LLMModerator", "ModerationMiddleware", "OpenAIModerator"]


def __getattr__(name: str) -> object:
    if name == "OpenAIModerator":
        from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

        return OpenAIModerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
