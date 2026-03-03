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

from ai_arch_toolkit.toolkit.knowledge import (
    KnowledgeRegistry,
    load_directory,
    load_json,
    load_markdown,
    load_text,
    load_toml,
)

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

entry = load_text(registry, "style", tmp / "style_guide.txt", category="guides")
print(f"Loaded text: key={entry.key}, format={entry.format}, source={entry.source}")

entry = load_json(registry, "api_config", tmp / "api_config.json", category="config")
print(f"Loaded JSON: key={entry.key}, format={entry.format}")
print(f"  Content (formatted):\n{entry.content}")

entry = load_toml(registry, "model_settings", tmp / "settings.toml", category="config")
print(f"Loaded TOML: key={entry.key}, format={entry.format}")

entry = load_markdown(registry, "overview", tmp / "readme.md", category="docs")
print(f"Loaded Markdown: key={entry.key}, format={entry.format}")
print()

print(f"Registry now has {len(registry)} entries")
print(f"Categories: {registry.categories()}")
print()

# --- 3. Load an entire directory at once ---

registry2 = KnowledgeRegistry()
count = load_directory(
    registry2,
    tmp,
    prefix="project.",
    category="project-files",
    tags=("auto-loaded",),
)

print(f"=== load_directory loaded {count} files ===")
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

registry3 = KnowledgeRegistry()
count = load_directory(registry3, tmp, recursive=True, prefix="kb.")

print(f"=== Recursive load: {count} files ===")
for key in registry3.keys():  # noqa: SIM118
    print(f"  {key}")
print()

# --- 5. Filter loaded entries and build context ---

# Combine specific entries into a prompt context
context = registry.as_context(
    "overview",
    "style",
    transform=lambda k, c: f"## {k.replace('_', ' ').title()}\n{c}",
)
print("=== Combined Context ===")
print(context)

# Cleanup
shutil.rmtree(tmp)
print(f"\nCleaned up {tmp}")
