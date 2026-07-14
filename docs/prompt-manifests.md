# Declarative Prompt Manifests

Manifests keep prompt structure, variables, sources, and layout in version control. Use
explicit `.prompt.yaml`, `.prompt.yml`, `.prompt.json`, or `.prompt.toml` filenames.

```yaml
version: 1
name: story-writer

layout:
  type: markdown

variables:
  genre:
    type: string
    required: true
  audience:
    type: string
    default: general readers

sections:
  - name: role
    source: role.md
    order: 100

  - name: rules
    source:
      path: rules.yaml
      select: /writing/rules
      serialize_as: markdown
    order: 200

  - name: request
    template: request.template.md
    order: 900
    stability: request
```

```python
from ai_arch_toolkit import load_prompt

template = load_prompt("story-writer.prompt.yaml")
rendered = template.render(genre="mystery")
```

Paths are relative to the manifest and constrained to its directory by default. Unknown
fields, invalid selectors, missing variables, duplicate sections, and cycles fail before an
LLM call.

Package manifests can be loaded without copying them to the current working directory:

```python
template = load_prompt("package://my_prompts/manifests/story.prompt.yaml")
```

The package URI is restricted to the named import package and rejects `.` / `..` path
segments. Relative includes and section sources remain constrained to the extracted manifest
directory.

## Includes and inheritance

`include` adds sections from another manifest; duplicates are errors. `extends` inherits a
base definition. Child sections use explicit operations:

```yaml
extends: base.prompt.yaml
sections:
  - name: rules
    replace: true
    content: New rules
  - name: legacy
    remove: true
```

Includes and inheritance are cycle checked and depth limited. Variables in a child override
base declarations; included manifests may not introduce duplicate variables.

The packaged JSON Schema is
`ai_arch_toolkit/toolkit/prompts/schemas/prompt-manifest-v1.schema.json`.

## CLI

```bash
ai-arch prompt validate prompts/story-writer.prompt.yaml
ai-arch prompt inspect prompts/story-writer.prompt.yaml
ai-arch prompt render prompts/story-writer.prompt.yaml --var genre=mystery
```

Knowledge-backed sections can be supplied to the CLI with `--knowledge-dir DIR` (and
`--knowledge-recursive`) or repeated `--knowledge KEY=FILE` options.
