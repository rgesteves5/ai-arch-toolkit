"""39 — Prompt files and resource selectors. No API key required."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection
from ai_arch_toolkit.toolkit.resources import MarkdownHeading

ASSETS = Path(__file__).parent / "assets/prompts/story_writer"

prompt = Prompt(
    sections=(
        PromptSection.from_file(ASSETS / "role.md", name="role", order=100),
        PromptSection.from_file(
            ASSETS / "rules.yaml",
            name="rules",
            selector="/writing/rules",
            serialize_as="markdown",
            order=200,
        ),
        PromptSection.from_file(
            ASSETS / "guide.md",
            name="quality",
            selector=MarkdownHeading(heading="Quality Checklist"),
            order=300,
        ),
    )
)

rendered = prompt.render()
print(rendered.text)
print(f"\nFingerprint: {rendered.fingerprint}")
