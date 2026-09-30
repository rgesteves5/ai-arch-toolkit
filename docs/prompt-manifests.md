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

## Subsections

A section may declare nested `sections`. A parent with content renders it before its
subsections; a parent with only `sections` is a pure container. Names must be unique
across the whole tree.

```yaml
sections:
  - name: context
    content: General rules.
    sections:
      - name: rules
        source: rules.md
      - name: examples
        content: Examples.
```

The TOML equivalent nests arrays of tables (`[[sections.sections]]`); JSON nests the
`sections` key. Nested definitions cannot carry `remove`/`replace`/`merge` flags — those
are merge operations, described below.

Package manifests can be loaded without copying them to the current working directory:

```python
template = load_prompt("package://my_prompts/manifests/story.prompt.yaml")
```

The part after the package name, without its leading slashes, is a path under the package root that
`importlib.resources.files()` returns; empty, `.`, and `..` segments in it are rejected. A package
in a regular directory is read in place. Any other package, such as a zipped one, is copied to a
temporary directory that is removed once the manifest has loaded; every included manifest and file
source is read during the load. From there the manifest loads like a file path: relative includes
and section sources resolve against it and, by default, stay inside its directory.

## Includes and inheritance

`include` adds sections from another manifest; duplicates are errors. `extends` inherits a
base definition. Child sections use explicit operations:

```yaml
version: 1
extends: base.prompt.yaml
sections:
  - name: rules
    replace: true
    content: New rules
  - name: legacy
    remove: true
```

Because names are unique across the tree, `replace` and `remove` address sections at any
nesting level; `replace` swaps the whole subtree and `remove` prunes it. To operate on a
section's subsections without restating the parent, use the explicit `merge` flag — a
merge entry may only define `sections`, whose entries are themselves operations:

```yaml
version: 1
extends: base.prompt.yaml
sections:
  - name: context
    merge: true
    sections:
      - name: examples
        content: New nested section
      - name: rules
        replace: true
        content: Replaced nested rules
```

Includes and inheritance are cycle checked and depth limited. Variables in a child override
base declarations; included manifests may not introduce duplicate variables.

The loader checks every manifest file (extended and included ones too) against one declared
shape, and the packaged JSON Schema,
`ai_arch_toolkit/toolkit/prompts/schemas/prompt-manifest-v1.schema.json`, is generated from the
same declaration, so an editor that uses it flags the same shape errors. Only the loader refuses
`1.0` where an integer goes, since JSON cannot tell `1` from `1.0`. Rules that involve
several fields, files, or a registry (one source per section, `remove`/`replace`/`merge`, a
template's path or content, paths, serializer and engine names, knowledge) are checked by the
loader alone. Errors name the field's path, as in
`sections[0].source.select.start must be a positive integer`. An inline template
(`template: {content: ...}`) takes no `select` or `serialize_as`: those read a template file.

## CLI

```bash
ai-arch prompt validate prompts/story-writer.prompt.yaml
ai-arch prompt inspect prompts/story-writer.prompt.yaml
ai-arch prompt render prompts/story-writer.prompt.yaml --var genre=mystery
```

`render` also takes `--vars FILE` (a JSON, YAML, or TOML object of variables) and `--layout`
(`json`, `markdown`, `text`, or `xml`). A `--var` value that parses as JSON is passed decoded, so
`--var n=3` is the integer `3`.

Knowledge-backed sections can be supplied to the CLI with `--knowledge-dir DIR` (and
`--knowledge-recursive`) or repeated `--knowledge KEY=FILE` options.
