"""40 — Explicit templates and typed variables. No API key required."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.toolkit.prompts import PromptTemplate, PromptVariable

ASSETS = Path(__file__).parent / "assets/prompts/story_writer"

template = PromptTemplate.from_file(
    ASSETS / "request.template.md",
    name="request",
    variables=(
        PromptVariable(name="genre", value_type="string", required=True),
        PromptVariable(name="audience", value_type="string", default="general readers"),
        PromptVariable(name="task", value_type="string", required=True),
    ),
)

rendered = template.render(genre="mystery", task="Write chapter one")
print(rendered.text)
print(f"Definition: {template.fingerprint}")
print(f"Rendered:   {rendered.fingerprint}")
