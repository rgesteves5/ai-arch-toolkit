"""Structured prompt message and Content composition tests."""

from __future__ import annotations

import pytest

from ai_arch_toolkit import document, image
from ai_arch_toolkit.toolkit.prompts import (
    Prompt,
    PromptConversation,
    PromptMessage,
    PromptTemplate,
    PromptTemplateSection,
    PromptVariable,
)


def test_conversation_renders_system_user_and_assistant_messages() -> None:
    template = PromptTemplate(
        sections=(
            PromptTemplateSection.literal(
                name="request", content="Write ${topic}.", engine="string-template"
            ),
        ),
        variables=(PromptVariable(name="topic", required=True),),
    )
    conversation = PromptConversation(
        messages=(
            PromptMessage(role="system", content=Prompt.from_text("You are a writer.")),
            PromptMessage(role="user", content=template),
            PromptMessage(role="assistant", content="Understood."),
        )
    )
    rendered = conversation.render(topic="chapter one", unused="ignored")
    messages, system = rendered.to_llm_request()
    assert system == "You are a writer."
    assert messages == [
        {"role": "user", "content": "Write chapter one."},
        {"role": "assistant", "content": "Understood."},
    ]
    assert rendered.messages[1].rendered_prompt is not None


def test_conversation_preserves_multimodal_content_and_fingerprints_bytes() -> None:
    conversation = PromptConversation(
        messages=(
            PromptMessage(
                role="user",
                content=[
                    "Describe these inputs.",
                    image(b"image", "image/png"),
                    document(b"pdf", name="guide.pdf"),
                ],
            ),
        )
    )
    first = conversation.render()
    second = conversation.render()
    messages, system = first.to_llm_request()
    assert system is None
    assert messages[0]["content"][1].source == b"image"
    assert first.fingerprint == second.fingerprint


def test_conversation_rejects_invalid_roles_parts_and_multiple_systems() -> None:
    with pytest.raises(ValueError, match="invalid prompt message role"):
        PromptMessage(role="tool", content="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unsupported part"):
        PromptMessage(role="user", content=[object()])  # type: ignore[list-item]
    rendered = PromptConversation(
        messages=(
            PromptMessage(role="system", content="one"),
            PromptMessage(role="system", content="two"),
        )
    ).render()
    with pytest.raises(ValueError, match="at most one system"):
        rendered.to_llm_request()


def test_message_fingerprint_changes_with_role_content_and_order() -> None:
    user = PromptConversation(messages=(PromptMessage(role="user", content="x"),)).render()
    assistant = PromptConversation(
        messages=(PromptMessage(role="assistant", content="x"),)
    ).render()
    changed = PromptConversation(messages=(PromptMessage(role="user", content="y"),)).render()
    assert len({user.fingerprint, assistant.fingerprint, changed.fingerprint}) == 3
