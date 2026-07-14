"""35 — Knowledge Loaders.

Load knowledge from files (text, JSON, TOML, Markdown) and directories
into a KnowledgeRegistry.

No API keys needed. Creates temporary files for demonstration.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection

# --- 1. Create temporary files to load ---

tmp = Path(tempfile.mkdtemp())

(tmp / "style_guide.txt").write_text(
    "Use active voice. Keep paragraphs under 4 sentences. Avoid adverbs."
)

(tmp / "api_config.json").write_text(
    json.dumps({"base_url": "https://api.example.com", "version": "v2", "timeout": 30}, indent=2)
)

(tmp / "settings.toml").write_text(
    '[model]\nname = "gpt-4.1-nano"\ntemperature = 0.7\nmax_tokens = 1024\n'
)

(tmp / "readme.md").write_text(
    "# Project Overview\n\nThis project implements a task management API.\n\n"
    "## Goals\n- Simple REST interface\n- Token-based auth\n- Real-time updates\n"
)

print(f"Created temp files in: {tmp}")
print()

# --- 2. Load individual files ---

registry = KnowledgeRegistry()

entry = registry.load("style", tmp / "style_guide.txt", category="guides")
print(f"Loaded text: key={entry.key}, format={entry.format}, source={entry.source}")

entry = registry.load(
    "api_config", tmp / "api_config.json", serialize_as="json", category="config"
)
print(f"Loaded JSON: key={entry.key}, format={entry.format}")
print(f"  Content (formatted):\n{entry.content}")

entry = registry.load("model_settings", tmp / "settings.toml", category="config")
print(f"Loaded TOML: key={entry.key}, format={entry.format}")

entry = registry.load("overview", tmp / "readme.md", category="docs")
print(f"Loaded Markdown: key={entry.key}, format={entry.format}")
print()

print(f"Registry now has {len(registry)} entries")
print(f"Categories: {registry.categories()}")
print()

# --- 3. Load an entire directory at once ---

registry2 = KnowledgeRegistry.from_directory(
    tmp,
    prefix="project.",
    category="project-files",
    tags=("auto-loaded",),
)

print(f"=== from_directory loaded {len(registry2)} files ===")
for key in registry2.keys():  # noqa: SIM118
    e = registry2.require(key)
    print(f"  {key}: format={e.format}, tags={e.tags}, source={Path(e.source).name}")
print()

# --- 4. Load a directory tree recursively ---

# Create subdirectories
(tmp / "prompts").mkdir()
(tmp / "prompts" / "system.txt").write_text("You are a helpful assistant.")
(tmp / "prompts" / "review.txt").write_text("Review the following text for clarity.")
(tmp / "schemas").mkdir()
user_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
(tmp / "schemas" / "user.json").write_text(json.dumps(user_schema))

registry3 = KnowledgeRegistry.from_directory(tmp, recursive=True, prefix="kb.")

print(f"=== Recursive load: {len(registry3)} files ===")
for key in registry3.keys():  # noqa: SIM118
    print(f"  {key}")
print()

# --- 5. Filter loaded entries and build context ---

# Combine specific entries through a prompt section
prompt = Prompt.from_sections(
    PromptSection(name="role", content="You are a project assistant.", order=100),
    PromptSection.from_knowledge(
        registry,
        ["overview", "style"],
        include_names=True,
        order=200,
    ),
)
print("=== Combined Prompt Context ===")
print(prompt.render(layout="markdown").text)

# Cleanup
shutil.rmtree(tmp)
print(f"\nCleaned up {tmp}")
