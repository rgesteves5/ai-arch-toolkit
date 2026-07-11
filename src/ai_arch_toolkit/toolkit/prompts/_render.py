"""Deterministic prompt rendering and validation."""

from __future__ import annotations

import hashlib

from ai_arch_toolkit.toolkit.prompts._types import Prompt, PromptSection, RenderedPrompt

_STABILITY_RANK = {"static": 0, "session": 1, "request": 2}


def render_prompt(prompt: Prompt) -> RenderedPrompt:
    """Validate and render a structured prompt.

    Sections are ordered by ``order`` and retain insertion order when values
    tie. Names must be unique. Stability must progress from static to session
    to request content so volatile content cannot silently split a reusable
    prefix.
    """
    ordered = _ordered_sections(prompt.sections)
    _validate_unique_names(ordered)
    _validate_stability_order(ordered)

    text = prompt.separator.join(section.content for section in ordered)
    fingerprint = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    stable_prefix_end = _stable_prefix_end(ordered, prompt.separator)
    return RenderedPrompt(
        text=text,
        sections=ordered,
        fingerprint=fingerprint,
        stable_prefix_end=stable_prefix_end,
    )


def _ordered_sections(sections: tuple[PromptSection, ...]) -> tuple[PromptSection, ...]:
    indexed = sorted(enumerate(sections), key=lambda pair: (pair[1].order, pair[0]))
    return tuple(section for _index, section in indexed)


def _validate_unique_names(sections: tuple[PromptSection, ...]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for section in sections:
        if section.name in seen and section.name not in duplicates:
            duplicates.append(section.name)
        seen.add(section.name)
    if duplicates:
        names = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"prompt section names must be unique; duplicates: {names}")


def _validate_stability_order(sections: tuple[PromptSection, ...]) -> None:
    previous: PromptSection | None = None
    for section in sections:
        if previous is not None and (
            _STABILITY_RANK[section.stability] < _STABILITY_RANK[previous.stability]
        ):
            raise ValueError(
                "prompt stability must progress from static to session to request; "
                f"section {section.name!r} ({section.stability}) follows "
                f"{previous.name!r} ({previous.stability})"
            )
        previous = section


def _stable_prefix_end(sections: tuple[PromptSection, ...], separator: str) -> int | None:
    static_contents: list[str] = []
    for section in sections:
        if section.stability != "static":
            break
        static_contents.append(section.content)
    if not static_contents:
        return None
    return len(separator.join(static_contents))


__all__ = ["render_prompt"]
