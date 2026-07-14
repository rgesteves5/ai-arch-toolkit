"""41 — Text, Markdown, XML, and JSON prompt layouts. No API key required."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.prompts import (
    JsonLayout,
    MarkdownLayout,
    Prompt,
    PromptSection,
    SeparatorPolicy,
    TextLayout,
    XmlLayout,
)

prompt = Prompt(
    sections=(
        PromptSection(name="role", content="You are a story architect.", order=100),
        PromptSection(name="rules", content="Keep characters consistent.", order=200),
        PromptSection(
            name="request",
            content="Write chapter one.",
            order=900,
            stability="request",
        ),
    )
)

layouts = (
    TextLayout(
        separator=SeparatorPolicy(
            default="\n\n",
            between={("rules", "request"): "\n\n--- REQUEST ---\n\n"},
        )
    ),
    MarkdownLayout(),
    XmlLayout(root_tag="instructions"),
    JsonLayout(include_stability=True),
)

for layout in layouts:
    rendered = prompt.render(layout=layout)
    print(f"\n=== {layout.name.upper()} ===")
    print(rendered.text)
