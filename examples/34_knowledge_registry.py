"""34 — Knowledge Registry.

Register reference data, filter by category and tags, and build
prompt context strings for injection into LLM calls.

No API keys needed.
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry

# --- 1. Create a registry and register entries ---

registry = KnowledgeRegistry()

registry.register(
    "tone_guide",
    "Write in a friendly, professional tone. Avoid jargon. Use short sentences.",
    category="constraints",
    tags=("writing", "style"),
)

registry.register(
    "audience",
    "Target audience: software developers with 2-5 years of experience.",
    category="constraints",
    tags=("writing", "audience"),
)

registry.register(
    "api_schema",
    '{"endpoint": "/api/v1/users", "methods": ["GET", "POST"], "auth": "bearer"}',
    format="json",
    category="schemas",
    tags=("api", "reference"),
)

registry.register(
    "error_codes",
    "400: Bad Request\n401: Unauthorized\n404: Not Found\n500: Internal Server Error",
    category="reference",
    tags=("api", "errors"),
)

registry.register(
    "project_context",
    "We are building a REST API for a task management application.",
    category="context",
    tags=("project",),
)

print(f"Registered {len(registry)} entries")
print(f"Keys: {registry.keys()}")
print(f"Categories: {registry.categories()}")
print()

# --- 2. Retrieve entries ---

entry = registry.require("tone_guide")
print(f"Entry: key={entry.key}, format={entry.format}, category={entry.category}")
print(f"Tags: {entry.tags}")
print()

# --- 3. Filter by category ---

print("=== Constraints ===")
for e in registry.by_category("constraints"):
    print(f"  [{e.key}] {e.content[:60]}...")
print()

# --- 4. Filter by tags ---

print("=== API-related (match_all=False) ===")
for e in registry.by_tags("api", match_all=False):
    print(f"  [{e.key}] {e.content[:60]}...")
print()

print("=== Writing + Style (match_all=True) ===")
for e in registry.by_tags("writing", "style", match_all=True):
    print(f"  [{e.key}] {e.content[:60]}...")
print()

# --- 5. Build prompt context with as_context() ---

# Basic: join content with separator
context = registry.as_context("tone_guide", "audience")
print("=== Basic Context ===")
print(context)
print()

# Custom separator
context = registry.as_context("tone_guide", "audience", separator="\n\n")
print("=== Custom Separator ===")
print(context)
print()

# Custom transform: wrap each entry in XML-style tags
context = registry.as_context(
    "project_context",
    "tone_guide",
    "api_schema",
    transform=lambda key, content: f"<{key}>\n{content}\n</{key}>",
)
print("=== XML-Tagged Context ===")
print(context)
print()

# --- 6. Use in a prompt template ---

system_prompt = f"""You are a technical writer.

{
    registry.as_context(
        "project_context",
        "tone_guide",
        "audience",
        separator="\n\n",
        transform=lambda k, c: f"[{k.upper().replace('_', ' ')}]\n{c}",
    )
}

Use the above guidelines for all responses."""

print("=== System Prompt ===")
print(system_prompt)
