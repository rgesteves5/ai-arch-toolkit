"""Tests for conversation memory utilities."""

from __future__ import annotations

from ai_arch_toolkit._legacy.llm._memory import ConversationMemory, SlidingWindowMemory
from ai_arch_toolkit._legacy.llm._types import Message


def test_conversation_memory_add_extend_history_clear() -> None:
    memory = ConversationMemory()
    memory.add_user("hello")
    memory.add_assistant("hi")
    memory.extend([Message(role="user", content="next")])

    history = memory.history()
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[1].role == "assistant"

    memory.clear()
    assert memory.history() == []


def test_conversation_memory_token_count() -> None:
    memory = ConversationMemory()
    memory.add_user("a" * 40)
    assert memory.token_count() > 0


def test_sliding_window_memory_trims_oldest() -> None:
    memory = SlidingWindowMemory(max_tokens=8)
    memory.add_user("a" * 20)
    memory.add_assistant("b" * 20)

    history = memory.history()
    assert len(history) <= 1
