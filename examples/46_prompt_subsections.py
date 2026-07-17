"""46 — Prompt Subsections.

Nest sections into a tree: each section renders its own content first and
then its subsections, and layouts translate depth (Markdown deepens headings,
XML nests elements). No API call needed.
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection, render_prompt

prompt = Prompt(
    sections=(
        PromptSection(
            name="context",
            content="You are a senior software architect.",
            order=100,
            sections=(
                PromptSection(
                    name="rules",
                    content=(
                        "Be concise and state important trade-offs.\n"
                        "Prefer provider-agnostic architecture advice."
                    ),
                    order=100,
                ),
                PromptSection(
                    name="examples",
                    content="Q: Monolith or services? A: Depends on team size.",
                    order=200,
                ),
            ),
        ),
        PromptSection(
            name="task",
            content="Review the public API design.",
            order=200,
            stability="request",
        ),
    )
)

markdown = render_prompt(prompt, layout="markdown")
xml = render_prompt(prompt, layout="xml")

print("=== Sections (preorder) ===")
print(markdown.section_names)
print("\n=== Markdown (headings deepen per level) ===")
print(markdown.text)
print("\n=== XML (elements nest) ===")
print(xml.text)
print("\n=== Subtree slice ===")
print(markdown.section_text("context"))
print("\n=== Provenance ===")
print(f"fingerprint: {markdown.fingerprint}")
print(f"stable prefix characters: {markdown.stable_prefix_end}")
