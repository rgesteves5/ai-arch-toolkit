# Templates & Variables

Prompt content is literal unless a template engine is selected explicitly. This keeps JSON,
shell, code, braces, and dollar-heavy content safe from accidental substitution.

## Stdlib templates

```python
from ai_arch_toolkit import PromptTemplate, PromptVariable

template = PromptTemplate.from_file(
    "request.template.md",
    variables=(
        PromptVariable(name="topic", value_type="string", required=True),
        PromptVariable(name="audience", value_type="string", default="general"),
    ),
)

rendered = template.render(topic="graphs")
```

The built-in engine uses strict `${name}` syntax. Missing variables are errors; optional
variables are omitted unless a default exists.

Supported types are `string`, `integer`, `number`, `boolean`, `array`, `object`, and `any`.
Optional JSON Schema validation is available with the `prompts` extra.

## Jinja

Install `ai-arch-toolkit[templates]` or `[prompts]`, then select `jinja2` explicitly:

```yaml
template:
  path: examples.template.md
  engine: jinja2
```

Jinja uses `SandboxedEnvironment` and `StrictUndefined`. Sandboxing is defence in depth,
not permission to execute untrusted templates.

## Provenance

Rendered provenance records variable names and the template engine, not variable values.
The final rendered text and its fingerprint necessarily contain values visible to the model.
