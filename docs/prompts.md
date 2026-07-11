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

Stability must progress `static → session → request`. Rendering fails if volatile content
appears before a more stable section because that layout silently breaks reusable prompt
prefixes. `RenderedPrompt.stable_prefix` and `stable_prefix_end` expose the initial static
text for diagnostics.

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
prompt_from_sections(sections, *, separator="\n\n") -> Prompt
```

`position=` remains a temporary compatibility alias for Nanope's experimental configurable
agent. New toolkit code should use `order=`.
