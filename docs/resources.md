# Resources & File Loading

`toolkit.resources` loads content independently from Prompts and Knowledge. A `Resource`
preserves bytes, decoded text, parsed data, media type, fingerprint, and provenance.

```python
from ai_arch_toolkit.toolkit.resources import load_resource

resource = load_resource("prompts/rules.yaml")
print(resource.text)
print(resource.data)
print(resource.fingerprint)
```

Built-in formats are TXT, Markdown, JSON, TOML, YAML, and raw bytes. YAML requires the
`yaml` or `prompts` extra.

Resources can also be created without touching the filesystem. This is useful when an
application receives a generated fragment, a database value, or bytes from another adapter:

```python
from ai_arch_toolkit.toolkit.resources import Resource

text_resource = Resource.from_text("generated rules", uri="memory://rules")
binary_resource = Resource.from_bytes(
    pdf_bytes,
    uri="memory://brief.pdf",
    media_type="application/pdf",
)
```

Use `Prompt.from_resource()` / `PromptSection.from_resource()` or
`KnowledgeRegistry.register_resource()` to consume these snapshots.

## Select fragments

JSON, YAML, and TOML use RFC 6901 JSON Pointer:

```python
from ai_arch_toolkit.toolkit.resources import select_resource

rules = select_resource(resource, "/writing/rules/0")
```

Text selectors are explicit:

```python
from ai_arch_toolkit.toolkit.resources import MarkdownHeading, LineRange, NamedBlock

Prompt.from_file("guide.md", selector=MarkdownHeading(heading="Rules"))
Prompt.from_file("guide.txt", selector=LineRange(start=10, end=20))
```

## Serialize selected data

```python
Prompt.from_file(
    "rules.yaml",
    selector="/writing/rules",
    serialize_as="markdown",
)
```

Built-in serializers are `text`, `json`, `yaml`, and `markdown`. With no selector or
serializer, the original decoded source text is preserved.

Custom serializers are isolated per `ResourceResolver`:

```python
from ai_arch_toolkit import PromptSection
from ai_arch_toolkit.toolkit.resources import ResourceResolver

resolver = ResourceResolver()
resolver.register_serializer("compact", CompactSerializer())
section = PromptSection.from_file(
    "rules.json",
    name="rules",
    selector="/rules",
    serialize_as="compact",
    resolver=resolver,
)
```

## Directories

```python
from ai_arch_toolkit.toolkit.resources import load_resources

resources = load_resources("knowledge/", recursive=True)
```

Results are sorted by full relative path.

## Policies

```python
from pathlib import Path
from ai_arch_toolkit.toolkit.resources import ResourcePolicy, load_resource

policy = ResourcePolicy(
    allowed_roots=(Path("prompts"),),
    max_bytes=1_000_000,
    allow_remote=False,
    allowed_media_types=frozenset({"text/plain", "text/markdown"}),
)
resource = load_resource("prompts/system.md", policy=policy)
```

Prompt manifests restrict relative resources to the manifest directory by default. Remote
resources are disabled; custom loaders must be registered explicitly.
