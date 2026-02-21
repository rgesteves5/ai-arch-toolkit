"""Tests for prompt and chat templates."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.llm._templates import ChatTemplate, PromptTemplate
from ai_arch_toolkit.llm._types import Message, TextPart


def test_prompt_template_formats_with_str_format() -> None:
    template = PromptTemplate("Hello {name}, task={task}.")
    assert template.format(name="Ana", task="summarize") == "Hello Ana, task=summarize."


def test_prompt_template_raises_for_missing_key() -> None:
    template = PromptTemplate("Hello {name}")
    with pytest.raises(KeyError):
        template.format()


def test_chat_template_from_tuples_and_format_messages() -> None:
    template = ChatTemplate.from_tuples(
        [
            ("system", "You are a {role}."),
            ("user", "Summarize {topic}."),
        ]
    )
    messages = template.format_messages(role="helper", topic="logs")

    assert messages == [
        Message(role="system", content="You are a helper."),
        Message(role="user", content="Summarize logs."),
    ]


def test_chat_template_preserves_non_string_content() -> None:
    template = ChatTemplate(messages=(Message(role="user", content=(TextPart(text="x"),)),))
    messages = template.format_messages(name="ignored")
    assert messages[0].content == (TextPart(text="x"),)
