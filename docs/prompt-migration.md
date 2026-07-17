# Prompt and Knowledge Migration

The existing literal prompt API remains supported. Default rendering and fingerprints are
byte-compatible. Subsections (`PromptSection.sections`) are additive: prompts without them
render the exact same bytes, keep the same fingerprints (text and definition), and produce
the same spans.

| Existing use | Preferred addition |
|---|---|
| `Prompt(sections=..., separator=...)` | Still supported |
| `render_prompt(prompt)` | Still supported; `prompt.render()` is the convenience form |
| manual `Path.read_text()` | `Prompt.from_file()` or `PromptSection.from_file()` |
| Knowledge `load_json()` etc. | `KnowledgeRegistry.load()` or `toolkit.resources` |
| `knowledge.as_context()` into a section | `PromptSection.from_knowledge()` |
| manual XML/JSON concatenation | `XmlLayout` or `JsonLayout` |
| application-owned YAML prompt config | `.prompt.yaml` + `load_prompt()` |

Compatibility loader signatures remain `load_text(registry, key, path, ...)` and equivalent
for other formats. They delegate to Resources and retain legacy return types.

Knowledge duplicate registration now requires `overwrite=True`; this prevents accidental
replacement. Directory stem collisions remain errors.

Nanope retains built-in and `extra_sections` prompts and additionally accepts:

```yaml
prompt:
  manifest: prompts/agent.prompt.yaml
  variables:
    domain: fiction
  layout: markdown
  mode: append  # or replace
```
