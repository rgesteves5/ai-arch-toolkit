# Structured Prompts

`toolkit.prompts` composes named prompt sections into deterministic text and records an
exact fingerprint of what was rendered. It is provider-agnostic: the result can be passed
to `LLM.complete(..., system=...)`, `ReasoningSpec.system`, or another application.

```python
from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection, render_prompt

prompt = Prompt(
    sections=(
        PromptSection(
            name="role",
            content="You are a senior software architect.",
            order=100,
        ),
        PromptSection(
            name="rules",
            content="Be concise and explain trade-offs.",
            order=200,
        ),
    )
)

rendered = render_prompt(prompt)

print(rendered.text)
print(rendered.section_names)
print(rendered.fingerprint)  # sha256:...
```

## Deterministic rendering

Sections are sorted by `order`. Sections with the same order retain insertion order.
Names must be unique; duplicate names raise `ValueError` instead of silently overriding
or duplicating instructions. The configured separator defaults to two newlines.

Metadata is recursively frozen for built-in containers. It participates in object equality
but is excluded from hashing and from the rendered-text fingerprint, so prompts and sections
remain hashable without conflating metadata with model-visible content.

The fingerprint is SHA-256 over the exact UTF-8 bytes of `RenderedPrompt.text`, including
whitespace. It identifies the rendered prompt, not the entire generation: model, tools,
messages, parameters, and output schema must be tracked separately for complete replay.

```python
response = llm.complete_sync(
    "Review this design.",
    system=rendered.text,
)
```

## Stability and cache layout

`PromptSection.stability` describes how often content may change:

| Value | Intended lifetime |
|---|---|
| `static` | Shared by requests, such as role, policies, and examples |
| `session` | Shared within one session or tenant context |
| `request` | Specific to the current request |

```python
prompt = Prompt(
    sections=(
        PromptSection(name="rules", content=rules, order=100),
        PromptSection(
            name="tenant",
            content=tenant_context,
            order=200,
            stability="session",
        ),
        PromptSection(
            name="current_request",
            content=request_context,
            order=300,
            stability="request",
        ),
    )
)
```

`order` is always the only semantic ordering key. Stability never reorders or rejects a
prompt: a static section after session/request content is valid, but it cannot be part of the
initial reusable prefix. `RenderedPrompt.stable_prefix` and `stable_prefix_end` expose that
initial static text for diagnostics.

Applications that require a cache-optimized `static → session → request` layout can validate
it explicitly:

```python
from ai_arch_toolkit.toolkit.prompts import validate_cache_layout

validate_cache_layout(prompt)  # raises ValueError for a non-monotonic layout
```

This is a layout diagnostic, not provider cache activation. Provider caching still depends
on each API and the current core content contract. Use `cache()` for the Anthropic content
blocks supported by the LLM facade; see [Content & Messages](content.md).

## Knowledge Registry

The two APIs have separate responsibilities:

- `KnowledgeRegistry` stores and selects reusable content.
- `Prompt` determines structure, order, stability, and provenance.

```python
from ai_arch_toolkit import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import Prompt, PromptSection, render_prompt

knowledge = KnowledgeRegistry()
knowledge.register("style", "Use short sentences.")
knowledge.register("domain", "Use architecture terminology.")

prompt = Prompt(
    sections=(
        PromptSection(name="role", content="You are an architect.", order=100),
        PromptSection(
            name="knowledge",
            content=knowledge.as_context("style", "domain"),
            order=200,
        ),
    )
)

rendered = render_prompt(prompt)
```

## Template scope

The first public API intentionally treats every section as literal text. It does not perform
variable substitution, so JSON, shell, code, Mermaid, and other brace- or dollar-heavy
content needs no escaping. A template contract will only be added after its syntax is
validated against real prompts from multiple projects.

## API

```python
Prompt(
    *,
    sections: tuple[PromptSection, ...] = (),
    separator: str = "\n\n",
)

PromptSection(
    *,
    name: str,
    content: str,
    order: int = 0,
    stability: Literal["static", "session", "request"] = "static",
    metadata: Mapping[str, Any] | None = None,
)

render_prompt(prompt: Prompt) -> RenderedPrompt
validate_cache_layout(prompt: Prompt) -> None
prompt_from_sections(sections, *, separator="\n\n") -> Prompt
```

`position=` remains a temporary compatibility alias for Nanope's experimental configurable
agent. New toolkit code should use `order=`. Nanope config mappings still place an extra
section without either field at order `1000`; direct `PromptSection` construction uses the
toolkit default `0`. Duplicate names, including collisions with Nanope built-ins, are rejected.
