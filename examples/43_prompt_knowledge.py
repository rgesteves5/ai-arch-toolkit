"""43 — Resources, Knowledge, and prompts without manual concatenation. No API key required."""

from __future__ import annotations

from pathlib import Path

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection

ASSETS = Path(__file__).parent / "assets/prompts/story_writer"

knowledge = KnowledgeRegistry()
knowledge.load(
    "writing.rules",
    ASSETS / "rules.yaml",
    selector="/writing/rules",
    serialize_as="markdown",
    category="writing",
    tags=("story", "style"),
)
knowledge.load("writing.role", ASSETS / "role.md", category="writing")

prompt = Prompt(
    sections=(
        PromptSection.from_knowledge(
            knowledge,
            ["writing.role", "writing.rules"],
            include_names=True,
        ),
    )
)

print(prompt.render(layout="markdown").text)
