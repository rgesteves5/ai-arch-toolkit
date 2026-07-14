"""Prompt composition across provider-agnostic conversation messages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from ai_arch_toolkit.core._content import CachePart, Content, ContentPart, DocumentPart, ImagePart
from ai_arch_toolkit.toolkit.prompts._templates import PromptTemplate
from ai_arch_toolkit.toolkit.prompts._types import Prompt, RenderedPrompt

type PromptMessageRole = Literal["system", "user", "assistant"]
type PromptMessageContent = Prompt | PromptTemplate | Content | tuple[ContentPart, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedPromptMessage:
    """One fully resolved message with fingerprint and prompt provenance."""

    role: PromptMessageRole
    content: str | tuple[ContentPart, ...]
    fingerprint: str
    rendered_prompt: RenderedPrompt | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"invalid prompt message role {self.role!r}")
        content = tuple(self.content) if isinstance(self.content, list) else self.content
        if not isinstance(content, str | tuple):
            raise TypeError("RenderedPromptMessage.content must be text or content parts")
        _validate_parts(content)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_message(self) -> dict[str, Any]:
        """Return a core-compatible message mapping."""
        content: Content = list(self.content) if isinstance(self.content, tuple) else self.content
        return {"role": self.role, "content": content}


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptMessage:
    """A message backed by literal Content, Prompt, or PromptTemplate."""

    role: PromptMessageRole
    content: PromptMessageContent
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"invalid prompt message role {self.role!r}")
        content = tuple(self.content) if isinstance(self.content, list) else self.content
        if not isinstance(content, Prompt | PromptTemplate | str | tuple):
            raise TypeError("PromptMessage.content must be Content, Prompt, or PromptTemplate")
        if isinstance(content, tuple):
            _validate_parts(content)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("PromptMessage.metadata must be a mapping")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def render(self, **variables: Any) -> RenderedPromptMessage:
        """Resolve this message without calling an LLM."""
        rendered_prompt: RenderedPrompt | None = None
        if isinstance(self.content, PromptTemplate):
            supplied = {
                name: variables[name] for name in self.content.variable_names if name in variables
            }
            rendered_prompt = self.content.render(**supplied)
            content: str | tuple[ContentPart, ...] = rendered_prompt.text
        elif isinstance(self.content, Prompt):
            rendered_prompt = self.content.render()
            content = rendered_prompt.text
        elif isinstance(self.content, str):
            content = self.content
        else:
            content = tuple(self.content)
        fingerprint = _message_fingerprint(self.role, content)
        provenance = {
            "metadata": dict(self.metadata),
            "prompt_fingerprint": rendered_prompt.fingerprint if rendered_prompt else None,
        }
        return RenderedPromptMessage(
            role=self.role,
            content=content,
            fingerprint=fingerprint,
            rendered_prompt=rendered_prompt,
            provenance=provenance,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedPromptConversation:
    """Fully rendered ordered messages ready for an LLM request."""

    messages: tuple[RenderedPromptMessage, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not all(isinstance(message, RenderedPromptMessage) for message in messages):
            raise TypeError("messages must contain RenderedPromptMessage values")
        object.__setattr__(self, "messages", messages)

    def to_llm_request(self) -> tuple[list[dict[str, Any]], str | None]:
        """Split one textual system message from user/assistant messages."""
        systems = [message for message in self.messages if message.role == "system"]
        if len(systems) > 1:
            raise ValueError("LLM requests support at most one system prompt")
        system: str | None = None
        if systems:
            if not isinstance(systems[0].content, str):
                raise TypeError("system prompt content must be text")
            system = systems[0].content
        messages = [message.to_message() for message in self.messages if message.role != "system"]
        return messages, system


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptConversation:
    """Ordered system/user/assistant prompt definitions rendered together."""

    messages: tuple[PromptMessage, ...]

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not all(isinstance(message, PromptMessage) for message in messages):
            raise TypeError("messages must contain PromptMessage values")
        object.__setattr__(self, "messages", messages)

    def render(self, **variables: Any) -> RenderedPromptConversation:
        """Render every message and fingerprint their exact ordered content."""
        rendered = tuple(message.render(**variables) for message in self.messages)
        encoded = json.dumps(
            [{"role": message.role, "fingerprint": message.fingerprint} for message in rendered],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return RenderedPromptConversation(
            messages=rendered,
            fingerprint="sha256:" + hashlib.sha256(encoded).hexdigest(),
        )


def _validate_parts(content: str | tuple[ContentPart, ...]) -> None:
    if isinstance(content, str):
        return
    if not all(isinstance(part, str | ImagePart | DocumentPart | CachePart) for part in content):
        raise TypeError("prompt message content contains an unsupported part")


def _message_fingerprint(
    role: PromptMessageRole,
    content: str | tuple[ContentPart, ...],
) -> str:
    payload = {"role": role, "content": _content_payload(content)}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _content_payload(content: str | tuple[ContentPart, ...]) -> Any:
    if isinstance(content, str):
        return content
    result: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            result.append({"type": "text", "content": part})
        elif isinstance(part, CachePart):
            result.append({"type": "cache", "content": part.content, "ttl": part.ttl})
        else:
            source = part.source
            if isinstance(source, bytes):
                source = "sha256:" + hashlib.sha256(source).hexdigest()
            item: dict[str, Any] = {
                "type": type(part).__name__,
                "source": source,
                "media_type": part.media_type,
            }
            if isinstance(part, DocumentPart):
                item["name"] = part.name
            result.append(item)
    return result


__all__ = [
    "PromptConversation",
    "PromptMessage",
    "PromptMessageContent",
    "PromptMessageRole",
    "RenderedPromptConversation",
    "RenderedPromptMessage",
]
