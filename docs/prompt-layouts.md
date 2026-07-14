# Layouts & Separators

Layouts serialize resolved sections. They do not parse source files and do not constrain the
model response.

```python
prompt.render(layout="text")
prompt.render(layout="markdown")
prompt.render(layout="xml")
prompt.render(layout="json")
```

## Text and boundary separators

```python
from ai_arch_toolkit.toolkit.prompts import SeparatorPolicy, TextLayout

layout = TextLayout(
    separator=SeparatorPolicy(
        default="\n\n",
        between={("examples", "request"): "\n\n--- REQUEST ---\n\n"},
    )
)
rendered = prompt.render(layout=layout)
```

`Prompt.separator` remains the shorthand used by the default text layout.

For reusable boundary rules, `SeparatorPolicy` also supports separators before or after a
named section and a Python resolver:

```python
policy = SeparatorPolicy(
    default="\n\n",
    before={"request": "\n\n<REQUEST>\n"},
    after={"request": "\n</REQUEST>"},
    resolver=lambda previous, current: f"\n\n[{previous.name} -> {current.name}]\n",
)
```

`resolver` is a Python-only extension and takes precedence over `between`; manifests use the
serializable `separator`, `between`, `before`, and `after` forms.

## Markdown

`MarkdownLayout` adds configurable headings. A section can set `metadata={"title": "Rules"}`
without changing its stable machine name.

## XML

`XmlLayout` uses the stdlib XML serializer for escaping. Section names are attributes rather
than arbitrary element names. Selected scalar metadata can be copied to attributes:

```python
XmlLayout(metadata_attributes=("audience", "version"), include_stability=True)
```

## JSON

`JsonLayout` emits an ordered array so section order is unambiguous. Unicode, indentation,
and stability fields are configurable. Use `mode="object"` when section names should be JSON
keys (names are still validated as unique by the prompt renderer):

```python
prompt.render(layout=JsonLayout(mode="object", include_stability=True))
```

Every built-in layout produces section spans. `RenderedPrompt.section_text(name)` returns
the exact layout-visible slice, and stable-prefix diagnostics use those offsets.
