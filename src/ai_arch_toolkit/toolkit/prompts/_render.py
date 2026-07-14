"""Deterministic prompt rendering and validation."""

from __future__ import annotations

import hashlib

from ai_arch_toolkit.toolkit.prompts._layouts import (
    PromptLayout,
    SectionSpan,
    TextLayout,
    layout_from_name,
)
from ai_arch_toolkit.toolkit.prompts._types import Prompt, PromptSection, RenderedPrompt

_STABILITY_RANK = {"static": 0, "session": 1, "request": 2}


def render_prompt(
    prompt: Prompt,
    *,
    layout: str | PromptLayout | None = None,
) -> RenderedPrompt:
    """Validate and render a structured prompt.

    Sections are ordered by ``order`` and retain insertion order when values
    tie. Names must be unique. ``order`` is the only semantic ordering key;
    stability describes cache layout without changing or rejecting the prompt.
    """
    ordered = _ordered_sections(prompt.sections)
    _validate_unique_names(ordered)

    active_layout: PromptLayout
    if layout is None:
        active_layout = TextLayout(separator=prompt.separator)
    elif isinstance(layout, str):
        active_layout = layout_from_name(layout, separator=prompt.separator)
    else:
        active_layout = layout
    layout_result = active_layout.render(ordered)
    _validate_layout_spans(ordered, layout_result.text, layout_result.spans)
    text = layout_result.text
    fingerprint = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    stable_prefix_end = _stable_prefix_end(ordered, layout_result.spans, len(text))
    return RenderedPrompt(
        text=text,
        sections=ordered,
        fingerprint=fingerprint,
        stable_prefix_end=stable_prefix_end,
        section_spans=layout_result.spans,
        layout=layout_result.layout,
    )


def validate_cache_layout(prompt: Prompt) -> None:
    """Raise when ordered sections do not form a cache-optimized stability layout.

    This validation is opt-in because cache layout must never change or reject
    an otherwise valid prompt during normal rendering.
    """
    _validate_stability_order(_ordered_sections(prompt.sections))


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


def _stable_prefix_end(
    sections: tuple[PromptSection, ...],
    spans: tuple[SectionSpan, ...],
    text_length: int,
) -> int | None:
    static_count = 0
    for section in sections:
        if section.stability != "static":
            break
        static_count += 1
    if static_count == 0:
        return None
    if static_count == len(sections):
        return text_length
    return spans[static_count - 1].end


def _validate_layout_spans(
    sections: tuple[PromptSection, ...],
    text: str,
    spans: tuple[SectionSpan, ...],
) -> None:
    if len(spans) != len(sections):
        raise ValueError(f"prompt layout returned {len(spans)} spans for {len(sections)} sections")
    previous_end = 0
    for section, span in zip(sections, spans, strict=True):
        if span.name != section.name:
            raise ValueError(
                f"prompt layout span {span.name!r} does not match section {section.name!r}"
            )
        if span.start < previous_end or span.end > len(text):
            raise ValueError(f"prompt layout returned invalid span for section {section.name!r}")
        previous_end = span.end


__all__ = ["render_prompt", "validate_cache_layout"]
