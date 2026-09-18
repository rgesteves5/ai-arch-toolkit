"""The shape of an agent manifest (``*.agent.*``), declared once.

The loader checks every manifest file against ``AGENT_MANIFEST``, and the packaged JSON Schema
(``schemas/agent-manifest-v1.schema.json``) is written from it. A ``null`` value means "not set",
as if the field were absent. Values that belong to a runtime type take its choices
(``TraceCapture``, ``Reserve``, ``Unpriced``). What needs more than one field or the file system
stays with the loader: a phase prompt given twice, a prompt file that is a manifest, paths under
the allowed roots, secrets.
"""

from __future__ import annotations

from ai_arch_toolkit.core._trace import TraceCapture
from ai_arch_toolkit.toolkit._shape import (
    Choice,
    Const,
    Either,
    Fields,
    Flag,
    Free,
    Items,
    Maybe,
    Names,
    Number,
    Ref,
    Shape,
    Text,
    Whole,
    json_schema,
)
from ai_arch_toolkit.toolkit.budget._policy import Reserve, Unpriced

__all__ = ["AGENT_MANIFEST", "REACT_KNOBS", "schema"]

# ReAct's knobs, also accepted directly under ``strategy`` (``reasoning_spec`` folds them in).
REACT_KNOBS = (
    "final_answer_hint",
    "parallel_tool_calls",
    "show_turn_counter",
    "strip_tools_on_final",
)

_TEXT = Maybe(Text())
_TEXTS = Items(Text())

_MODEL = Ref("model").bind(
    Fields(
        {
            "provider": _TEXT,
            "model": _TEXT,
            "profile": _TEXT,
            "base_url": _TEXT,
            "structured_output_mode": _TEXT,
            "max_tokens": Maybe(Whole(minimum=1)),
            "temperature": Number(minimum=0, maximum=2),
        }
    )
)

_STRATEGY = Ref("strategy").bind(
    Fields(
        {
            "name": _TEXT,
            "system": Maybe(Text(empty=True)),
            "max_iterations": Maybe(Whole(minimum=1)),
            "timeout": Maybe(Number(above=0)),
            "trace_capture": Maybe(Choice.of(TraceCapture)),
            "knobs": Free(),
            "llm_kwargs": Free(),
            "phases": Maybe(
                Names(Fields({"system": _TEXT, "system_file": _TEXT, "model": Maybe(_MODEL)}))
            ),
            **dict.fromkeys(REACT_KNOBS, Maybe(Flag())),
        }
    )
)

_PROMPTS = Ref("prompts").bind(
    Fields(
        {
            "system": _TEXT,
            "system_manifest": _TEXT,
            "user": _TEXT,
            "request_template": _TEXT,
            "input_adapter": _TEXT,
            "request_variables": Maybe(Fields({"required": _TEXTS, "optional": _TEXTS})),
        }
    )
)

_LIMITS = Ref("limits").bind(
    Fields(
        {
            "timeout_seconds": Maybe(Number(above=0)),
            "max_wall_s": Maybe(Number(above=0)),
            "max_llm_calls": Maybe(Whole(minimum=0)),
            "max_tool_calls": Maybe(Whole(minimum=0)),
            "max_input_tokens": Maybe(Whole(minimum=0)),
            "max_output_tokens": Maybe(Whole(minimum=0)),
            "max_total_tokens": Maybe(Whole(minimum=0)),
            "max_cost": Maybe(Number(minimum=0)),
            "reserve": Maybe(Choice.of(Reserve)),
            "unpriced": Maybe(Choice.of(Unpriced)),
        }
    )
)

# What a profile may set: everything but the manifest's identity and inheritance.
_BODY: dict[str, Shape] = {
    "description": _TEXT,
    "phase": _TEXT,
    "result_adapter": _TEXT,
    "metadata": Free(),
    "strategy": Maybe(_STRATEGY),
    "model": Maybe(_MODEL),
    "prompts": Maybe(_PROMPTS),
    "output": Maybe(Fields({"schema": _TEXT})),
    "tools": Maybe(Fields({"factory": _TEXT, "manifest": _TEXT})),
    "limits": Maybe(_LIMITS),
    "override_policy": Maybe(Fields({"allow": _TEXTS, "deny": _TEXTS})),
}

AGENT_MANIFEST = Fields(
    {
        "version": Const(1),
        "id": _TEXT,
        "extends": Either(Text(), _TEXTS),
        "profiles": Names(Ref("profile").bind(Fields(_BODY))),
        **_BODY,
    },
    required=frozenset({"version"}),
)


def schema() -> dict[str, object]:
    """``AGENT_MANIFEST`` as the JSON Schema in ``schemas/agent-manifest-v1.schema.json``.

    To rewrite that file after changing the declaration::

        uv run python -c "import json; from ai_arch_toolkit.toolkit.agents._manifest_shape \\
            import schema; print(json.dumps(schema(), indent=2))" \\
            > src/ai_arch_toolkit/toolkit/agents/schemas/agent-manifest-v1.schema.json
    """
    return json_schema(
        AGENT_MANIFEST,
        title="ai-arch-toolkit agent manifest v1",
        schema_id="https://ai-arch-toolkit.dev/schemas/agent-manifest-v1.schema.json",
    )
