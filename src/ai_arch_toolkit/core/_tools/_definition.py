"""Canonical tool definition types.

Splits a tool into three concerns:

- ``ToolSchema``: the provider-facing contract (``name``/``description``/
  ``input_schema``). This — and only this — is sent to LLM APIs.
- ``ToolRuntimePolicy``: declarative governance metadata for a tool
  (capability, risk level, approval requirement, output and time bounds). Never reaches a
  provider.
- ``ToolDefinition``: the runtime object binding a callable to its schema and
  policy. Produced by ``@tool`` and stored as ``fn.__tool_definition__``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

type RiskLevel = Literal["low", "medium", "high", "critical"]

DEFAULT_MAX_OUTPUT_CHARS = 200_000
DEFAULT_TIMEOUT_S = 120.0

# The names every provider and MCP accept, their rules intersected: Anthropic's
# ^[a-zA-Z0-9_-]{1,128}$ (https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools),
# OpenAI's letters, digits, "_" and "-" up to 64 (``openai`` FunctionDefinition.name), Gemini's
# first character a letter or "_" (``google-genai`` FunctionDeclaration.name), and MCP's
# [A-Za-z0-9_.-]{1,128} (https://modelcontextprotocol.io/specification/2026-07-28/server/tools).
_PORTABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")


def check_bounds(max_output_chars: int | None, timeout_s: float | None) -> None:
    """Raise ``ValueError`` unless each bound is positive or ``None``."""
    for name, value in (("max_output_chars", max_output_chars), ("timeout_s", timeout_s)):
        if value is not None and value <= 0:
            msg = f"{name} must be positive or None, got {value!r}"
            raise ValueError(msg)


def check_tool_name(name: object) -> None:
    """Raise ``ValueError`` unless ``name`` is a tool name every provider takes.

    ``LLM(fallback=...)`` may move a run to another provider, so a name holds for all of them.
    """
    if not isinstance(name, str) or not _PORTABLE_NAME.fullmatch(name):
        msg = (
            f"Tool name {name!r} is not portable: use 1 to 64 letters, digits, '_' or '-', "
            "starting with a letter or '_', the names every provider and MCP accept"
        )
        raise ValueError(msg)


def one_per_name[T](entries: Iterable[tuple[str | None, object, T]]) -> list[T]:
    """The items of ``(name, identity, item)`` entries, one per name, in their order.

    An identity met again under its name is the same tool and counts once; identities compare
    by equality, so a bound method read twice is one tool. An entry without a name (a server
    tool) is always kept.

    Raises:
        ValueError: Two different identities share a name: the model could not choose between
            them, and a provider refuses the request.
    """
    held: dict[str, object] = {}
    kept: list[T] = []
    for name, identity, item in entries:
        if name is not None:
            if name in held:
                if held[name] != identity:
                    raise ValueError(name_clash(name))
                continue
            held[name] = identity
        kept.append(item)
    return kept


def name_clash(name: str) -> str:
    """Why two tools cannot share ``name``, and how to fix it."""
    return (
        f"Two different tools are named {name!r}: under one name, the model could not choose "
        "between them. Give one another name with @tool(name=...) or tool_from_schema(name=...)."
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSchema:
    """Provider-facing tool contract. The only part sent to LLM APIs.

    Raises:
        ValueError: ``name`` is not portable: 1 to 64 letters, digits, ``_`` or ``-``, starting
            with a letter or ``_``.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_tool_name(self.name)

    def to_provider_dict(self) -> dict[str, Any]:
        """Return the provider-safe dict (no governance metadata)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolRuntimePolicy:
    """Declarative governance metadata for a tool. Never sent to a provider.

    Attributes:
        capability: What the tool reaches (``"network"``, ``"filesystem"``, ...), for gates and
            approval handlers.
        risk_level: How much harm a wrong call can do.
        requires_approval: Whether a human or handler must approve each call.
        approval_reason: Why approval is needed, shown to the approver.
        max_output_chars: Most characters of the result the model receives; the executor cuts the
            rest and marks the cut. ``None`` lets any size through.
        timeout_s: Seconds the executor waits for the tool before the call fails with
            ``timeout``. ``None`` waits as long as it takes. A synchronous tool cannot be killed:
            it runs in a daemon thread, which the executor stops waiting for.
    """

    capability: str | None = None
    risk_level: RiskLevel = "low"
    requires_approval: bool = False
    approval_reason: str = ""
    max_output_chars: int | None = DEFAULT_MAX_OUTPUT_CHARS
    timeout_s: float | None = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        check_bounds(self.max_output_chars, self.timeout_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolDefinition:
    """Canonical runtime object: callable + schema + runtime policy."""

    fn: Callable[..., Any]
    schema: ToolSchema
    policy: ToolRuntimePolicy = field(default_factory=ToolRuntimePolicy)
