"""Prompt and chat templating utilities."""

from __future__ import annotations

from dataclasses import dataclass

from ai_arch_toolkit.llm._types import Message


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A plain string template rendered via ``str.format``."""

    template: str

    def format(self, **kwargs: object) -> str:
        """Render the template with ``str.format`` semantics."""
        return self.template.format(**kwargs)


@dataclass(frozen=True, slots=True)
class ChatTemplate:
    """A reusable template for structured chat message lists."""

    messages: tuple[Message, ...]

    @classmethod
    def from_tuples(cls, items: list[tuple[str, str]]) -> ChatTemplate:
        """Create a template from ``(role, content_template)`` items."""
        return cls(messages=tuple(Message(role=role, content=content) for role, content in items))

    def format_messages(self, **kwargs: object) -> list[Message]:
        """Render all templated messages using ``str.format``."""
        rendered: list[Message] = []
        for message in self.messages:
            content = message.content
            if isinstance(content, str):
                rendered.append(Message(role=message.role, content=content.format(**kwargs)))
            else:
                rendered.append(message)
        return rendered

