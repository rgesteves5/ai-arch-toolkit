"""Content moderation — OpenAI API and LLM-based classifiers."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.moderation._llm import LLMModerator
from ai_arch_toolkit.toolkit.moderation._middleware import ModerationMiddleware

__all__ = ["LLMModerator", "ModerationMiddleware", "OpenAIModerator"]


def __getattr__(name: str) -> object:
    if name == "OpenAIModerator":
        from ai_arch_toolkit.toolkit.moderation._openai import OpenAIModerator

        return OpenAIModerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
