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
`yaml` or `prompts` extra. Raw bytes cover files of unknown type; a file whose extension maps
to another media type (a `.pdf` is `application/pdf`) needs a registered codec or
`media_type="application/octet-stream"`.

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
`KnowledgeRegistry.register_resource()` to consume these snapshots. The default `text`
serializer refuses bytes, so a binary resource needs a selector or serializer that turns it
into text.

## Select fragments

JSON, YAML, and TOML use RFC 6901 JSON Pointer:

```python
from ai_arch_toolkit.toolkit.resources import select_resource

rules = select_resource(resource, "/writing/rules/0")
```

Text selectors are explicit:

```python
from ai_arch_toolkit import Prompt
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

class CompactSerializer:
    name = "compact"

    def serialize(self, value):
        return ";".join(str(item) for item in value)

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

Only files with a known extension are loaded (`.txt`, `.md`, `.markdown`, `.json`, `.toml`,
`.yaml`, `.yml`, and any registered with a codec); `extensions={".md", ...}` sets the list
instead. Results are sorted by full relative path.

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

### Bounded parsing

An imported manifest or resource costs time and memory in proportion to its size. The JSON, TOML
and YAML codecs, and with them prompt manifests, knowledge and agent manifests, refuse a document
that nests deeper than 100 levels of mappings and lists. In YAML, anchors and merge keys
(`<<: *base`) work, but a document's aliases may add at most 10,000 nodes and 1,000,000
characters, or as many as the document holds if that is more: an alias bomb (a few hundred bytes
that expand to millions of values, or an alias that copies one long string thousands of times) is
refused before it is built, as is an alias inside its own anchor. A merged mapping's keys count
beside the mapping's own, not as a level, and a chain of merge keys may have at most 100 links.
A refused file raises `ResourceDecodeError` (and `PromptLoadError` or `AgentManifestError` where
a manifest was loaded). A manifest that several others extend or include is read once per load.

This covers what the toolkit loads as manifests and resources. Files that are the application's
own data (a saved memory graph, a price table) are parsed as they are.
