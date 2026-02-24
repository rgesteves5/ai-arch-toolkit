"""Conversation memory utilities with optional token-window trimming."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ai_arch_toolkit._legacy.llm._tokens import estimate_conversation_tokens, estimate_item_tokens
from ai_arch_toolkit._legacy.llm._types import ConversationItem, Message


@dataclass(slots=True)
class ConversationMemory:
    """Stores conversation history with basic append/retrieve operations."""

    items: list[ConversationItem] = field(default_factory=list)

    def add(self, item: ConversationItem) -> None:
        """Append a single conversation item."""
        self.items.append(item)

    def extend(self, items: Sequence[ConversationItem]) -> None:
        """Append multiple conversation items."""
        self.items.extend(items)

    def add_user(self, content: str) -> None:
        """Append a user message."""
        self.items.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant message."""
        self.items.append(Message(role="assistant", content=content))

    def history(self) -> list[ConversationItem]:
        """Return a shallow copy of the current history."""
        return list(self.items)

    def clear(self) -> None:
        """Clear all stored items."""
        self.items.clear()

    def token_count(self) -> int:
        """Estimate the token count for the current history."""
        return estimate_conversation_tokens(self.items)


@dataclass(slots=True)
class SlidingWindowMemory(ConversationMemory):
    """Conversation memory that trims oldest items to fit ``max_tokens``."""

    max_tokens: int = 4096

    def add(self, item: ConversationItem) -> None:
        ConversationMemory.add(self, item)
        self.trim_to_budget()

    def extend(self, items: Sequence[ConversationItem]) -> None:
        ConversationMemory.extend(self, items)
        self.trim_to_budget()

    def add_user(self, content: str) -> None:
        ConversationMemory.add_user(self, content)
        self.trim_to_budget()

    def add_assistant(self, content: str) -> None:
        ConversationMemory.add_assistant(self, content)
        self.trim_to_budget()

    def trim_to_budget(self) -> None:
        """Drop oldest messages until token budget is satisfied."""
        if not self.items:
            return
        total_tokens = self.token_count()
        if total_tokens <= self.max_tokens:
            return

        remove_count = 0
        for item in self.items:
            total_tokens -= estimate_item_tokens(item)
            remove_count += 1
            if total_tokens <= self.max_tokens:
                break
        if remove_count:
            del self.items[:remove_count]
