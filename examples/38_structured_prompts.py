"""38 — Structured Prompts.

Compose deterministic prompt sections, inject registered knowledge, and keep
an exact fingerprint for experiment tracking without making an API call.
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection, render_prompt

knowledge = KnowledgeRegistry()
knowledge.register("style", "Be concise and state important trade-offs.")
knowledge.register("domain", "Prefer provider-agnostic architecture advice.")

prompt = Prompt(
    sections=(
        PromptSection(
            name="role",
            content="You are a senior software architect.",
            order=100,
        ),
        PromptSection(
            name="knowledge",
            content=knowledge.as_context("style", "domain", separator="\n"),
            order=200,
        ),
        PromptSection(
            name="session_context",
            content="The project is a Python 3.13 library.",
            order=300,
            stability="session",
        ),
        PromptSection(
            name="request_context",
            content="Review the public API design.",
            order=400,
            stability="request",
        ),
    )
)

rendered = render_prompt(prompt)

print("=== Sections ===")
print(rendered.section_names)
print("\n=== Prompt ===")
print(rendered.text)
print("\n=== Provenance ===")
print(f"fingerprint: {rendered.fingerprint}")
print(f"stable prefix characters: {rendered.stable_prefix_end}")
