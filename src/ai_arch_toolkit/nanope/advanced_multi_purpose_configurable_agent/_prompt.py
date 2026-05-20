"""System prompt rendering for configurable agents."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._config import (
    AgentConfig,
    PromptSection,
)

# Built-in section positions. Multiples of 100 leave room for custom sections
# to slot in between (e.g. position=350 lands between goals and tasks).
_POSITION_IDENTITY = 100
_POSITION_ROLE = 200
_POSITION_GOALS = 300
_POSITION_TASKS = 400
_POSITION_STYLE = 500
_POSITION_CONSTRAINTS = 600
_POSITION_MEMORY = 700


type SectionProvider = Callable[[AgentConfig], PromptSection | None]
"""Callable that produces a prompt section from runtime config, or None to skip."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RenderedPrompt:
    """Rendered system prompt with metadata for debugging."""

    system: str
    sections: tuple[PromptSection, ...]
    fingerprint: str

    @property
    def section_names(self) -> tuple[str, ...]:
        """Names of included prompt sections, in render order."""
        return tuple(section.name for section in self.sections)


def render_system_prompt(
    config: AgentConfig,
    *,
    providers: Sequence[SectionProvider] = (),
) -> RenderedPrompt:
    """Render the agent system prompt from a resolved config.

    Args:
        config: Resolved agent configuration.
        providers: Optional callables that produce additional sections from
            runtime data. Each returns a ``PromptSection`` or ``None``.
            Combined with ``config.context.extra_sections`` and sorted by
            ``PromptSection.position`` (ascending, insertion-stable on ties).
    """
    sections: list[PromptSection] = [
        PromptSection(
            name="identity",
            position=_POSITION_IDENTITY,
            content=(
                f"Agent name: {config.identity.name}\n"
                f"Agent description: {config.identity.description}"
            ),
        )
    ]

    if config.context.role:
        sections.append(
            PromptSection(
                name="role",
                position=_POSITION_ROLE,
                content=f"Role: {config.context.role}",
            )
        )
    if config.context.goals:
        sections.append(
            PromptSection(
                name="goals",
                position=_POSITION_GOALS,
                content=_bullet_section("Goals", config.context.goals),
            )
        )
    if config.context.tasks:
        sections.append(
            PromptSection(
                name="tasks",
                position=_POSITION_TASKS,
                content=_bullet_section("Tasks", config.context.tasks),
            )
        )
    if config.context.style:
        sections.append(
            PromptSection(
                name="style",
                position=_POSITION_STYLE,
                content=f"Style: {config.context.style}",
            )
        )
    if config.context.constraints:
        sections.append(
            PromptSection(
                name="constraints",
                position=_POSITION_CONSTRAINTS,
                content=_bullet_section("Behavior constraints", config.context.constraints),
            )
        )
    if config.memory.private_enabled or config.memory.read or config.memory.write:
        memory_lines = [
            f"Private memory enabled: {config.memory.private_enabled}",
            f"Memory read allowed: {config.memory.read}",
            f"Memory write allowed: {config.memory.write}",
        ]
        if config.memory.read:
            memory_lines.append(
                "Use memory inspection/search tools before claiming what is or is not stored."
            )
        if config.memory.write:
            memory_lines.append(
                "Persist stable user facts, preferences, and durable project context when useful."
            )
            memory_lines.append(
                "Before saving a memory, check whether an equivalent memory already exists."
            )
        sections.append(
            PromptSection(
                name="memory",
                position=_POSITION_MEMORY,
                content="\n".join(memory_lines),
            )
        )

    sections.extend(config.context.extra_sections)

    for provider in providers:
        section = provider(config)
        if section is not None:
            sections.append(section)

    ordered = _sort_sections(sections)
    system = "\n\n".join(section.content for section in ordered)
    fingerprint = hashlib.sha256(system.encode("utf-8")).hexdigest()
    return RenderedPrompt(system=system, sections=tuple(ordered), fingerprint=fingerprint)


def _sort_sections(sections: Sequence[PromptSection]) -> list[PromptSection]:
    """Sort ascending by position; insertion order preserved on ties."""
    indexed = sorted(enumerate(sections), key=lambda pair: (pair[1].position, pair[0]))
    return [section for _index, section in indexed]


def _bullet_section(title: str, values: tuple[str, ...]) -> str:
    return "\n".join([f"{title}:"] + [f"- {value}" for value in values])
