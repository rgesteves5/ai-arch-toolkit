# Knowledge Registry

`KnowledgeRegistry` gives application-level keys, categories, tags, and lookup behavior to
reusable reference content. File parsing belongs to `toolkit.resources`; Knowledge consumes
loaded resources and focuses on its domain.

## Register literal knowledge

```python
from ai_arch_toolkit import KnowledgeRegistry

knowledge = KnowledgeRegistry()
knowledge.register(
    "company.style",
    "Use short sentences and active voice.",
    category="style",
    tags=("writing", "company"),
)

style = knowledge.require("company.style")
print(style.content)
print(style.fingerprint)
```

Duplicate keys are rejected. Replacement is explicit:

```python
knowledge.register("company.style", "New guide", overwrite=True)
```

## Load knowledge from a file

```python
knowledge.load(
    "story.rules",
    "knowledge/story.yaml",
    selector="/writing/rules",
    serialize_as="markdown",
    category="writing",
    tags=("story", "rules"),
)
```

`KnowledgeEntry` exposes model-ready `content`, parsed `data`, `media_type`, source
`fingerprint`, category, tags, metadata, and source.

## Load a directory

```python
knowledge = KnowledgeRegistry.from_directory(
    "knowledge/",
    recursive=True,
    prefix="kb.",
)
```

Nested paths become deterministic dotted keys such as `kb.product.rules`.

## Query

```python
writing = knowledge.by_category("writing")
story_rules = knowledge.by_tags("story", "rules")
anything_story_or_style = knowledge.by_tags("story", "style", match_all=False)
```

For a small deterministic registry, `search()` provides explainable lexical ranking without
embeddings or a network dependency:

```python
matches = knowledge.search("short writing rules", limit=5, category="style")
for match in matches:
    print(match.entry.key, match.score, match.matched_terms)
```

Scores weight matches in keys and tags above categories and content. Ties are resolved by key,
so the same registry snapshot produces the same result. This is intentionally a lightweight
domain query; use [Memory](memory.md) for durable agent memories or a specialized vector index
for large corpora.

## Use knowledge in a prompt

```python
from ai_arch_toolkit import Prompt, PromptSection

prompt = Prompt(
    sections=(
        PromptSection.from_knowledge(
            knowledge,
            ["company.style", "story.rules"],
            name="knowledge",
            include_names=True,
        ),
    )
)
```

Manifests can use Knowledge when a registry is supplied to `load_prompt()`:

```yaml
- name: knowledge
  knowledge:
    keys: [company.style, story.rules]
    include_names: true
```

```python
template = load_prompt("writer.prompt.yaml", knowledge=knowledge)
```

## Compatibility loaders

The original functions remain available and delegate to Resources:

```python
load_text(registry, "style", "style.txt")
load_json(registry, "schema", "schema.json")
load_yaml(registry, "rules", "rules.yaml")
load_toml(registry, "settings", "settings.toml")
load_markdown(registry, "guide", "guide.md")
load_directory(registry, "knowledge/", recursive=True)
```

Their required `registry`, `key`, and `path` arguments are retained. `as_context()` also
retains its exact legacy joining behavior, but new prompt code should prefer
`PromptSection.from_knowledge()` or a manifest Knowledge source.

For long-lived, agent-managed information, use [Memory](memory.md), not Knowledge.
