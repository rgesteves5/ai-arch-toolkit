"""Typed variable declarations and validation for prompt templates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from ai_arch_toolkit.toolkit.prompts._errors import (
    MissingPromptVariableError,
    PromptVariableError,
)

type PromptVariableType = Literal[
    "any", "array", "boolean", "integer", "number", "object", "string"
]


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()

_TYPE_CHECKS = {
    "array": lambda value: isinstance(value, list | tuple),
    "boolean": lambda value: isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, int | float) and not isinstance(value, bool),
    "object": lambda value: isinstance(value, Mapping),
    "string": lambda value: isinstance(value, str),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptVariable:
    """Declaration for one runtime prompt variable."""

    name: str
    value_type: PromptVariableType = "any"
    required: bool = False
    default: Any = field(default=MISSING, hash=False)
    description: str = ""
    json_schema: Mapping[str, Any] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("PromptVariable.name is required")
        if self.value_type not in {"any", *_TYPE_CHECKS}:
            raise ValueError(f"invalid prompt variable type {self.value_type!r}")
        if not isinstance(self.required, bool):
            raise TypeError("PromptVariable.required must be a boolean")
        if not isinstance(self.description, str):
            raise TypeError("PromptVariable.description must be a string")
        if self.json_schema is not None:
            if not isinstance(self.json_schema, Mapping):
                raise TypeError("PromptVariable.json_schema must be a mapping or None")
            object.__setattr__(self, "json_schema", MappingProxyType(dict(self.json_schema)))
        if self.default is not MISSING:
            self.validate(self.default)

    @property
    def has_default(self) -> bool:
        """Whether the variable declaration supplies a default."""
        return self.default is not MISSING

    def validate(self, value: Any) -> None:
        """Validate a runtime or default value."""
        if self.value_type != "any" and not _TYPE_CHECKS[self.value_type](value):
            raise PromptVariableError(
                f"prompt variable {self.name!r} must be {self.value_type}; "
                f"got {type(value).__name__}"
            )
        if self.json_schema is not None:
            try:
                import jsonschema
            except ImportError:
                raise ImportError(
                    "jsonschema is required to validate PromptVariable.json_schema: "
                    "pip install 'ai-arch-toolkit[prompts]'"
                ) from None
            try:
                jsonschema.validate(value, dict(self.json_schema))
            except jsonschema.ValidationError as exc:
                raise PromptVariableError(
                    f"prompt variable {self.name!r} does not match its JSON Schema: {exc.message}"
                ) from exc


def resolve_variables(
    declarations: tuple[PromptVariable, ...],
    supplied: Mapping[str, Any],
    *,
    allow_extra: bool = False,
) -> dict[str, Any]:
    """Validate and merge supplied values with declared defaults."""
    by_name: dict[str, PromptVariable] = {}
    duplicates: list[str] = []
    for variable in declarations:
        if variable.name in by_name:
            duplicates.append(variable.name)
        by_name[variable.name] = variable
    if duplicates:
        names = ", ".join(sorted(set(duplicates)))
        raise PromptVariableError(f"duplicate prompt variable declarations: {names}")

    unknown = sorted(set(supplied) - set(by_name))
    if unknown and not allow_extra:
        raise PromptVariableError(
            "unknown prompt variables: " + ", ".join(repr(name) for name in unknown)
        )

    resolved = dict(supplied) if allow_extra else {}
    missing: list[str] = []
    for name, variable in by_name.items():
        if name in supplied:
            value = supplied[name]
        elif variable.has_default:
            value = variable.default
        elif variable.required:
            missing.append(name)
            continue
        else:
            continue
        variable.validate(value)
        resolved[name] = value
    if missing:
        raise MissingPromptVariableError(
            "missing required prompt variables: "
            + ", ".join(repr(name) for name in sorted(missing))
        )
    return resolved


__all__ = ["MISSING", "PromptVariable", "PromptVariableType", "resolve_variables"]
