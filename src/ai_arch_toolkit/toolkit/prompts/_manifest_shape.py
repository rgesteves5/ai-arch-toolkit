"""The shape of a prompt manifest (``*.prompt.*``), declared once.

The loader checks every manifest file it reads against ``PROMPT_MANIFEST`` (extended and included
ones too), and the packaged JSON Schema (``schemas/prompt-manifest-v1.schema.json``) is written
from it. Values that belong to a runtime type take its choices (``PromptVariableType``,
``PromptStability``, the XML tag rule, the JSON layout modes). What needs more than one field, the
file system or a registry stays with the loader: one source per section, ``remove`` / ``replace``
/ ``merge``, a template's path or content, paths, serializers, template engines, knowledge.
"""

from __future__ import annotations

from ai_arch_toolkit.toolkit._shape import (
    Anything,
    Choice,
    Const,
    Either,
    Fields,
    Flag,
    Free,
    Items,
    Maybe,
    Names,
    Ref,
    Shape,
    Tagged,
    Text,
    Whole,
    json_schema,
)
from ai_arch_toolkit.toolkit.prompts._layouts import JSON_LAYOUT_MODES, XML_TAG
from ai_arch_toolkit.toolkit.prompts._types import PromptStability
from ai_arch_toolkit.toolkit.prompts._variables import PromptVariableType

__all__ = ["LAYOUTS", "PROMPT_MANIFEST", "schema"]

_ANY_TEXT = Text(empty=True)


_SEPARATORS = Names(_ANY_TEXT)
_SPACING: dict[str, Shape] = {
    "separator": _ANY_TEXT,
    "between": Items(
        Fields(
            {"from": _ANY_TEXT, "to": _ANY_TEXT, "separator": _ANY_TEXT},
            required=frozenset({"from", "to", "separator"}),
        )
    ),
    "before": _SEPARATORS,
    "after": _SEPARATORS,
}

LAYOUTS = Tagged(
    "type",
    {
        "text": Fields(_SPACING),
        "markdown": Fields(
            {
                "heading_level": Whole(minimum=1, maximum=6),
                "include_headings": Flag(),
                **_SPACING,
            }
        ),
        "xml": Fields(
            {
                "root_tag": Text(pattern=XML_TAG.pattern),
                "section_tag": Text(pattern=XML_TAG.pattern),
                "separator": _ANY_TEXT,
                "include_stability": Flag(),
                "metadata_attributes": Items(Text()),
            }
        ),
        "json": Fields(
            {
                "indent": Maybe(Whole(minimum=0)),
                "include_stability": Flag(),
                "ensure_ascii": Flag(),
                "mode": Choice(*JSON_LAYOUT_MODES),
            }
        ),
    },
)

_SELECT = Maybe(
    Either(
        _ANY_TEXT,
        Ref("selector").bind(
            Tagged(
                "type",
                {
                    "json_pointer": Fields({"value": _ANY_TEXT}),
                    "heading": Fields(
                        {
                            "heading": Text(),
                            "occurrence": Maybe(Whole(minimum=1)),
                            "include_heading": Flag(),
                        },
                        required=frozenset({"heading"}),
                    ),
                    "lines": Fields(
                        {"start": Whole(minimum=1), "end": Maybe(Whole(minimum=1))},
                        required=frozenset({"start"}),
                    ),
                    "block": Fields(
                        {"start_marker": Text(), "end_marker": Text(), "include_markers": Flag()},
                        required=frozenset({"start_marker", "end_marker"}),
                    ),
                },
            )
        ),
    )
)

_SOURCE = Fields(
    {
        "path": Text(),
        "media_type": Maybe(_ANY_TEXT),
        "select": _SELECT,
        "serialize_as": Text(),
    },
    required=frozenset({"path"}),
)

_TEMPLATE = Fields(
    {
        "path": Text(),
        "content": _ANY_TEXT,
        "engine": Text(),
        "select": _SELECT,
        "serialize_as": Text(),
    }
)

_KNOWLEDGE = Fields(
    {"keys": Items(Text()), "separator": _ANY_TEXT, "include_names": Flag()},
    required=frozenset({"keys"}),
)

_SECTION = Ref("section")
_SECTION.bind(
    Fields(
        {
            "name": Text(),
            "order": Whole(),
            "stability": Choice.of(PromptStability),
            "metadata": Free(),
            "replace": Flag(),
            "remove": Flag(),
            "merge": Flag(),
            "content": _ANY_TEXT,
            "source": Either(Text(), _SOURCE),
            "template": Either(Text(), _TEMPLATE),
            "knowledge": Either(Text(), Items(Text()), _KNOWLEDGE),
            "sections": Items(_SECTION),
        },
        required=frozenset({"name"}),
    )
)

_VARIABLE = Either(
    Choice.of(PromptVariableType),
    Fields(
        {
            "type": Choice.of(PromptVariableType),
            "required": Flag(),
            "default": Anything(),
            "description": _ANY_TEXT,
            "json_schema": Maybe(Free()),
        }
    ),
)

PROMPT_MANIFEST = Fields(
    {
        "version": Const(1),
        "name": _ANY_TEXT,
        "description": _ANY_TEXT,
        "extends": Maybe(Text()),
        "include": Either(Text(), Items(Text())),
        "separator": _ANY_TEXT,
        "layout": Maybe(Either(Choice(*LAYOUTS.variants), Ref("layout").bind(LAYOUTS))),
        "metadata": Free(),
        "variables": Names(_VARIABLE),
        "sections": Items(_SECTION),
    },
    required=frozenset({"version"}),
)


def schema() -> dict[str, object]:
    """``PROMPT_MANIFEST`` as the JSON Schema in ``schemas/prompt-manifest-v1.schema.json``.

    To rewrite that file after changing the declaration::

        uv run python -c "import json; from ai_arch_toolkit.toolkit.prompts._manifest_shape \\
            import schema; print(json.dumps(schema(), indent=2))" \\
            > src/ai_arch_toolkit/toolkit/prompts/schemas/prompt-manifest-v1.schema.json
    """
    return json_schema(
        PROMPT_MANIFEST,
        title="ai-arch-toolkit prompt manifest v1",
        schema_id="https://ai-arch-toolkit.dev/schemas/prompt-manifest-v1.schema.json",
    )
