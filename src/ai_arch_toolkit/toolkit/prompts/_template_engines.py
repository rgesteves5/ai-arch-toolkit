"""Explicit, pluggable template engines."""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any, Protocol

from ai_arch_toolkit.toolkit.prompts._errors import PromptTemplateError


class TemplateEngine(Protocol):
    """Render explicitly templated prompt content."""

    name: str

    def render(self, template: str, variables: Mapping[str, Any]) -> str:
        """Render one template string."""
        ...

    def variables(self, template: str) -> frozenset[str]:
        """Return statically discoverable variable names."""
        ...


class StringTemplateEngine:
    """Strict stdlib `${name}` template rendering without code execution."""

    name = "string-template"

    def render(self, template: str, variables: Mapping[str, Any]) -> str:
        normalized = {name: _stringify(value) for name, value in variables.items()}
        try:
            return string.Template(template).substitute(normalized)
        except KeyError as exc:
            raise PromptTemplateError(
                f"template requires missing variable {exc.args[0]!r}"
            ) from exc
        except ValueError as exc:
            raise PromptTemplateError(f"invalid string template: {exc}") from exc

    def variables(self, template: str) -> frozenset[str]:
        names: set[str] = set()
        for match in string.Template.pattern.finditer(template):
            if match.group("invalid") is not None:
                raise PromptTemplateError(f"invalid string template near offset {match.start()}")
            name = match.group("named") or match.group("braced")
            if name:
                names.add(name)
        return frozenset(names)


class JinjaTemplateEngine:
    """Optional sandboxed Jinja rendering with strict undefined variables."""

    name = "jinja2"

    def _environment(self):
        try:
            from jinja2 import StrictUndefined
            from jinja2.sandbox import SandboxedEnvironment
        except ImportError:
            raise ImportError(
                "jinja2 is required for Jinja prompt templates: "
                "pip install 'ai-arch-toolkit[templates]'"
            ) from None
        return SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)

    def render(self, template: str, variables: Mapping[str, Any]) -> str:
        environment = self._environment()
        try:
            return environment.from_string(template).render(dict(variables))
        except Exception as exc:
            raise PromptTemplateError(f"Jinja template rendering failed: {exc}") from exc

    def variables(self, template: str) -> frozenset[str]:
        try:
            from jinja2 import meta
        except ImportError as exc:
            self._environment()
            raise AssertionError("unreachable") from exc
        environment = self._environment()
        try:
            parsed = environment.parse(template)
        except Exception as exc:
            raise PromptTemplateError(f"invalid Jinja template: {exc}") from exc
        return frozenset(meta.find_undeclared_variables(parsed))


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    if isinstance(value, int | float):
        return str(value)
    import json

    return json.dumps(value, ensure_ascii=False)


def template_engine(
    engine: str | TemplateEngine,
) -> TemplateEngine:
    """Resolve a built-in template-engine name or return an engine object."""
    if not isinstance(engine, str):
        return engine
    normalized = engine.lower()
    if normalized in {"string", "string-template", "stdlib"}:
        return StringTemplateEngine()
    if normalized in {"jinja", "jinja2"}:
        return JinjaTemplateEngine()
    raise ValueError("unknown template engine; expected one of: jinja2, string-template")


__all__ = [
    "JinjaTemplateEngine",
    "StringTemplateEngine",
    "TemplateEngine",
    "template_engine",
]
