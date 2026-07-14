# Prompt and Resource System Design

Status: accepted for implementation.

## Goals

The prompt subsystem must make common operations small while keeping every important
stage replaceable. Consumers should be able to load a prompt from a file in one call,
but advanced consumers must be able to supply custom loaders, codecs, selectors,
serializers, template engines, and layouts.

The system preserves a strict distinction between:

- `Resource`: content loaded from an origin, with raw and parsed representations.
- `Knowledge`: reusable content indexed by application-level keys and metadata.
- `PromptTemplate`: an unresolved prompt definition with sources and variables.
- `Prompt`: an immutable collection of fully resolved literal sections.
- `PromptLayout`: a serializer from resolved sections to model-visible text.
- `RenderedPrompt`: exact model-visible text plus spans, fingerprints, and provenance.

`core` remains independent from these toolkit facilities. The output boundary is a
plain string passed to `LLM` or an agent.

## Dependency Direction

```text
toolkit.resources  <- toolkit.knowledge
toolkit.resources  <- toolkit.prompts
toolkit.knowledge  <- toolkit.prompts (adapter/source only)
toolkit.prompts    <- toolkit.agents and nanope
core               <- rendered text only
```

No dependency may point from `core` to `toolkit` or from resources to knowledge or
prompts.

## Public API Levels

The convenience level consists of `Prompt.from_text()`, `Prompt.from_file()`,
`PromptSection.from_file()`, `Prompt.render()`, `PromptTemplate.from_file()`, and
`load_prompt()`.

The composition level exposes `Prompt`, `PromptSection`, `PromptTemplate`, variables,
and built-in layouts.

The extension level exposes resource loaders/codecs/selectors/serializers, prompt
sources, layouts, and template-engine protocols from their specialized namespaces.

## Resolution Lifecycle

```text
manifest or Python definition
  -> load resources
  -> select fragments
  -> serialize structured values
  -> validate variables
  -> render explicitly templated sections
  -> create a literal Prompt
  -> apply a PromptLayout
  -> produce RenderedPrompt
```

`render_prompt(Prompt)` is pure and performs no I/O. `load_prompt()` and the `from_file`
constructors perform I/O eagerly. A loaded `PromptTemplate` therefore represents a
stable snapshot; reloading is explicit.

## Formats

The API uses separate terms for separate transformations:

- source format/media type: how bytes are decoded and parsed;
- selector: which value or fragment is chosen;
- `serialize_as`: how a selected value becomes section text;
- layout: how sections become final prompt text;
- response format: how a provider constrains the model response (outside this system).

Structured resources use RFC 6901 JSON Pointer. Markdown headings and text ranges use
explicit selector types. Whole resources use the identity selector.

## Templates

All content is literal unless a template engine is explicitly selected. The stdlib
engine uses strict `${name}` substitution. Jinja is optional, uses a sandbox and strict
undefined values, and is never described as safe for untrusted templates.

Missing required variables and type mismatches are errors. Provenance records variable
names, not values. Rendered text and its fingerprint necessarily contain the actual
model-visible values.

## Layouts and Spans

The default layout is text joined with `Prompt.separator`, preserving the current output
byte-for-byte. Built-in layouts are text, Markdown, XML, and JSON. XML and JSON use
stdlib serializers rather than string concatenation.

Layouts return text and a span for every section. Stable-prefix diagnostics are derived
from those spans, so they remain meaningful when a layout adds wrappers.

## Manifest v1

Prompt definitions use explicit `.prompt.yaml`, `.prompt.yml`, `.prompt.json`, or
`.prompt.toml` names. Ordinary structured files remain content resources.

Paths are relative to the manifest. The manifest parent is the default allowed root.
Unknown fields are rejected. Includes and inheritance are opt-in, cycle checked, depth
limited, and use explicit duplicate/replace behavior rather than implicit deep merges.

The canonical machine-readable contract is
`toolkit/prompts/schemas/prompt-manifest-v1.schema.json`.

## Security Defaults

- local and package resources only by default;
- remote access disabled;
- UTF-8 and bounded resource sizes;
- manifest paths constrained to allowed roots after symlink resolution;
- YAML parsed with `safe_load`;
- no Python evaluation in manifests or built-in templates;
- XML/JSON escaping delegated to serializers;
- include cycles and excessive depth rejected;
- error and provenance objects avoid template-variable values.

## Compatibility

`Prompt`, `PromptSection`, `Prompt.separator`, `render_prompt()`,
`validate_cache_layout()`, and `prompt_from_sections()` retain their existing behavior.
Default rendered text and fingerprints are golden-tested.

Knowledge file-loader functions remain compatibility wrappers over resources and retain
their return types. `KnowledgeRegistry.as_context()` retains its exact output. New code
uses `KnowledgeRegistry.load()` or prompt knowledge sources.

Nanope `position=`, built-in prompt sections, and `extra_sections` remain supported.

## Non-goals for v1

- implicit network fetching;
- arbitrary code execution in templates or manifests;
- automatic provider cache activation;
- model-specific prompt optimization;
- a prompt registry service or hosted prompt management platform;
- hidden file watching or hot reload;
- forcing the LLM core to understand toolkit prompt objects.
- speculative orchestration APIs (`compose_prompts`, `PromptProvider`, `PromptMiddleware`, or
  `AgentDefinition`) before repeated consumer requirements justify their contracts.
