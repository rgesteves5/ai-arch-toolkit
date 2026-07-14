"""42 — Load, inspect, and render a declarative prompt manifest. No API key required."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.toolkit.prompts import load_prompt

MANIFEST = Path(__file__).parent / "assets/prompts/story_writer/story-writer.prompt.yaml"

template = load_prompt(MANIFEST)
template.validate()

print(f"Prompt: {template.name}")
print(f"Variables: {template.variable_names}")
print(f"Sources: {template.sources}")

rendered = template.render(
    genre="mystery",
    audience="young adults",
    task="Write chapter one",
)

print("\n=== RENDERED ===")
print(rendered.text)
