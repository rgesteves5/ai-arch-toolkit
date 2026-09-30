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

The built-in engine is the stdlib `string.Template`, used strictly: `${name}` or `$name`
substitutes, and `$$` writes a literal `$`. Missing variables are errors; optional variables are
omitted unless a default exists.

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

`RenderedPrompt.provenance` records the supplied variable names, and each templated section's
metadata records its engine and variable names; neither records variable values, except that a
source selector written with `${name}` is recorded as resolved. The final rendered text and its
fingerprint necessarily contain values visible to the model.
