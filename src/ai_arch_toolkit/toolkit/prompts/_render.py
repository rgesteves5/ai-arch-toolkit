"""Deterministic prompt rendering and validation."""

from __future__ import annotations

import hashlib

from ai_arch_toolkit.toolkit.prompts._layouts import (
    PromptLayout,
    SectionSpan,
    TextLayout,
    layout_from_name,
)
from ai_arch_toolkit.toolkit.prompts._types import (
    Prompt,
    PromptSection,
    RenderedPrompt,
    _ordered_sections,
    _walk_sections,
)

_STABILITY_RANK = {"static": 0, "session": 1, "request": 2}


def render_prompt(
    prompt: Prompt,
    *,
    layout: str | PromptLayout | None = None,
) -> RenderedPrompt:
    """Validate and render a structured prompt.

    Sections are ordered by ``order`` within each sibling level and retain
    insertion order when values tie; subsections render after their parent's
    own content (canonical preorder). Names must be unique across the whole
    tree. ``order`` is the only semantic ordering key; stability describes
    cache layout without changing or rejecting the prompt.
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


def _validate_unique_names(sections: tuple[PromptSection, ...]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for section, _depth in _walk_sections(sections):
        if section.name in seen and section.name not in duplicates:
            duplicates.append(section.name)
        seen.add(section.name)
    if duplicates:
        names = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"prompt section names must be unique; duplicates: {names}")


def _validate_stability_order(sections: tuple[PromptSection, ...]) -> None:
    previous: PromptSection | None = None
    for section, _depth in _walk_sections(sections):
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
    walk = tuple(_walk_sections(sections))
    cut = next(
        (index for index, (section, _depth) in enumerate(walk) if section.stability != "static"),
        None,
    )
    if not walk or cut == 0:
        return None
    if cut is None:
        return text_length
    cut_depth = walk[cut][1]
    if cut_depth > walk[cut - 1][1]:
        # The first dynamic node is a child: stable bytes end with the parent's
        # own block; without content offsets fall back to the child's start.
        parent_span = spans[cut - 1]
        if parent_span.content_end is not None:
            return parent_span.content_end
        return spans[cut].start
    for index in range(cut - 1, -1, -1):
        if walk[index][1] == cut_depth:
            return spans[index].end
    return None


def _validate_layout_spans(
    sections: tuple[PromptSection, ...],
    text: str,
    spans: tuple[SectionSpan, ...],
) -> None:
    walk = tuple(_walk_sections(sections))
    if len(spans) != len(walk):
        raise ValueError(f"prompt layout returned {len(spans)} spans for {len(walk)} sections")
    ancestors: list[SectionSpan] = []
    sibling_floor: dict[int, int] = {}
    for (section, depth), span in zip(walk, spans, strict=True):
        if span.name != section.name:
            raise ValueError(
                f"prompt layout span {span.name!r} does not match section {section.name!r}"
            )
        if span.depth != depth:
            raise ValueError(
                f"prompt layout span {span.name!r} has depth {span.depth}; expected {depth}"
            )
        del ancestors[depth:]
        for level in [level for level in sibling_floor if level > depth]:
            del sibling_floor[level]
        parent = ancestors[-1] if ancestors else None
        lower = sibling_floor.get(depth, 0)
        if parent is not None:
            parent_floor = parent.content_end if parent.content_end is not None else parent.start
            lower = max(lower, parent_floor)
        upper = parent.end if parent is not None else len(text)
        if span.start < lower or span.end > upper:
            raise ValueError(f"prompt layout returned invalid span for section {section.name!r}")
        sibling_floor[depth] = span.end
        ancestors.append(span)


__all__ = ["render_prompt", "validate_cache_layout"]
