# Extending the Prompt System

Most applications only need `load_prompt()` and built-in layouts. Extension protocols live
in specialized namespaces so the convenience API stays small.

## Custom codec

```python
from ai_arch_toolkit.toolkit.resources import DecodedResource, ResourceResolver

class UpperCodec:
    name = "upper"

    def decode(self, raw, ref):
        text = raw.decode(ref.encoding).upper()
        return DecodedResource(data=text, text=text)

resolver = ResourceResolver()
resolver.register_codec("text/x-upper", UpperCodec(), extensions=("upper",))
prompt = Prompt.from_file("rules.upper", resolver=resolver)
```

The same registry supports custom URI-scheme loaders. Custom selectors implement
`select(resource)`. Serializers implement `serialize(value)`. Prompt layouts implement
`render(sections)` and return `LayoutResult` with one `SectionSpan` per section. Template
engines implement strict `render()` and variable discovery.

Custom Python callables can be wrapped in `CallableSource`; they are intentionally not
serializable in manifests.

## Serializer registry

Serializers are isolated by resolver, so an application can add or replace a name without
changing global behavior:

```python
from ai_arch_toolkit import Prompt
from ai_arch_toolkit.toolkit.resources import ResourceResolver

class CompactSerializer:
    name = "compact"

    def serialize(self, value):
        return ";".join(str(item) for item in value)

resolver = ResourceResolver()
resolver.register_serializer("compact", CompactSerializer())
prompt = Prompt.from_file(
    "rules.json",
    selector="/rules",
    serialize_as="compact",
    resolver=resolver,
)
```

The same resolver registry is used by `KnowledgeRegistry.load()` and prompt manifests. A
serializer must return a string; it should not perform I/O or mutate the selected value.

Extensions must remain deterministic, avoid hidden network or model calls, report accurate
spans, and respect `ResourcePolicy`.
