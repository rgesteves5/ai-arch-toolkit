# Knowledge Registry

A sync, in-memory registry for prompt-injectable reference data. Use it for domain knowledge, style guides, few-shot examples, or any structured context you want to weave into a system prompt.

`KnowledgeRegistry` owns reusable content; [Structured Prompts](prompts.md) owns section
order, stability, rendering, and fingerprints. Use `as_context()` as the content of a
`PromptSection` when both are needed.

```python
from ai_arch_toolkit import KnowledgeRegistry

registry = KnowledgeRegistry()

# Register entries
registry.register(
    "company_style",
    content="Always use Oxford commas. Avoid passive voice.",
    category="style",
    tags=frozenset({"writing", "formatting"}),
)

registry.register(
    "api_reference",
    content='{"endpoints": ["/users", "/posts"]}',
    format="json",
    category="technical",
)

# Query
style_guides = registry.by_category("style")
writing_docs = registry.by_tags("writing", "formatting")

# Inject into prompts
context = registry.as_context("company_style", "api_reference", separator="\n---\n")
# → "Always use Oxford commas. Avoid passive voice.\n---\n{\"endpoints\": ...}"

response = await llm.complete(
    "Write API documentation",
    system=f"Follow these guidelines:\n{context}",
)
```

`register(key, content, *, format="text", category="", tags=(), metadata=None, source="")` returns a `KnowledgeEntry`. `as_context(*keys, separator=..., transform=...)` builds the combined prompt string, optionally transforming each entry.

---

## Loaders

Load knowledge from files:

```python
from ai_arch_toolkit.toolkit.knowledge import (
    load_text, load_json, load_yaml, load_toml, load_markdown, load_directory,
)

# Single file
entries = load_text("style-guide.txt")

# Structured data (nested keys become separate entries)
entries = load_json("api-spec.json")
entries = load_yaml("config.yaml")
entries = load_toml("settings.toml")

# Markdown (sections become entries)
entries = load_markdown("docs/reference.md")

# Bulk load a directory
entries = load_directory("knowledge/", recursive=True)
```

---

For long-lived, agent-managed memory (as opposed to static reference data), see [Memory](memory.md).
