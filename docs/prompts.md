# Prompts

`toolkit.prompts` turns literal text, files, structured data, knowledge, and runtime
variables into deterministic model-visible text. Start with the smallest API that fits;
the same resolved `Prompt` and `RenderedPrompt` contracts are used at every level.

## Literal prompts

```python
from ai_arch_toolkit import Prompt, PromptSection

prompt = Prompt(
    sections=(
        PromptSection(name="role", content="You are an architect.", order=100),
        PromptSection(name="rules", content="Explain trade-offs.", order=200),
    )
)

rendered = prompt.render()
print(rendered.text)
print(rendered.fingerprint)
```

For one section:

```python
prompt = Prompt.from_text("You are a helpful assistant.")
```

## Prompts from files

```python
prompt = Prompt.from_file("prompts/system.md")
```

```python
section = PromptSection.from_file(
    "prompts/rules.yaml",
    name="rules",
    selector="/writing/rules",
    serialize_as="markdown",
    order=200,
)
```

See [Resources & File Loading](resources.md) for supported formats and selectors.

## Templates and manifests

```python
from ai_arch_toolkit import load_prompt

template = load_prompt("prompts/story-writer.prompt.yaml")
rendered = template.render(genre="mystery", task="Write chapter one")
```

Content remains literal unless a section explicitly selects a template engine. See
[Templates & Variables](prompt-templates.md) and
[Declarative Manifests](prompt-manifests.md).

## Rendering and layouts

The default output is section content joined with two newlines. Other layouts are
explicit:

```python
rendered = prompt.render(layout="xml")
rendered = prompt.render(layout="json")
rendered = prompt.render(layout="markdown")
```

See [Layouts & Separators](prompt-layouts.md).

## Determinism and provenance

Sections are ordered by `order`; ties preserve insertion order. Names must be unique.
The fingerprint is SHA-256 over the exact UTF-8 bytes of `RenderedPrompt.text`, including
whitespace and layout wrappers.

`RenderedPrompt` exposes:

- `text` and the compatibility alias `system`;
- ordered `sections` and `section_names`;
- `section_spans` and `section_text(name)`;
- `fingerprint`;
- `stable_prefix` and `stable_prefix_end`;
- `layout` and non-sensitive `provenance`.

The fingerprint identifies prompt text, not the whole generation. Model, tools, messages,
parameters, and output schema must be tracked separately for full replay.

## Deliberate scope boundary

This delivery stops at the stable prompt/resource contracts and the real consumer integrations.
It does not add generic `compose_prompts`, `PromptProvider`, `PromptMiddleware`, or
`AgentDefinition` abstractions yet. Those APIs should be designed only after the Story Creator
and Nanope have concrete repeated composition/lifecycle needs; adding them now would create a
second orchestration layer without a validated consumer.

## Stability and cache layout

`PromptSection.stability` describes the expected content lifetime:

| Value | Intended lifetime |
|---|---|
| `static` | Shared role, policies, examples |
| `session` | Tenant or conversation context |
| `request` | Current request data |

`order` remains the only semantic ordering key. Stability never reorders or rejects a
normal render. `validate_cache_layout(prompt)` is an opt-in check for a monotonic
`static → session → request` arrangement.

Stable-prefix diagnostics do not activate provider caching. Provider cache behavior remains
provider-specific; see [Content & Messages](content.md).

## Sending to an LLM

```python
response = llm.complete_sync(
    "Review this design.",
    system=rendered.text,
)
```

The LLM core accepts strings and `Content`; it deliberately does not depend on toolkit
prompt objects.

When the workflow needs ordered system/user/assistant turns, use
[Prompt Messages & Content](prompt-messages.md). It preserves multimodal `Content` parts and
returns the plain `messages, system` pair expected by `LLM`.

## Next steps

- [Context Model](context-model.md): Prompt vs Resource vs Knowledge vs Memory.
- [Resources & File Loading](resources.md): files and fragments.
- [Prompt Messages & Content](prompt-messages.md): ordered and multimodal conversations.
- [Templates & Variables](prompt-templates.md): explicit substitution.
- [Layouts & Separators](prompt-layouts.md): Text, Markdown, XML, JSON.
- [Declarative Manifests](prompt-manifests.md): versioned configuration.
- [Knowledge Registry](knowledge.md): reusable reference content.
- [Extending the Prompt System](prompt-extensibility.md): custom protocols.
- [Migration Guide](prompt-migration.md): existing APIs and replacements.
