"""Reusable prompt templates compiled into literal structured prompts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ai_arch_toolkit.toolkit.prompts._layouts import PromptLayout
from ai_arch_toolkit.toolkit.prompts._render import render_prompt
from ai_arch_toolkit.toolkit.prompts._sources import (
    LiteralSource,
    PromptSource,
    ResourceSource,
)
from ai_arch_toolkit.toolkit.prompts._template_engines import TemplateEngine, template_engine
from ai_arch_toolkit.toolkit.prompts._types import (
    Prompt,
    PromptSection,
    PromptStability,
    RenderedPrompt,
)
from ai_arch_toolkit.toolkit.prompts._variables import PromptVariable, resolve_variables
from ai_arch_toolkit.toolkit.resources import ResourcePolicy, ResourceResolver


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptTemplateSection:
    """An unresolved section backed by a source and optional template engine.

    ``sections`` holds optional unresolved subsections compiled into the
    resulting :class:`PromptSection` tree.
    """

    name: str
    source: PromptSource
    order: int = 0
    stability: PromptStability = "static"
    engine: str | TemplateEngine | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    sections: tuple[PromptTemplateSection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("PromptTemplateSection.name is required")
        if not isinstance(self.order, int):
            raise TypeError("PromptTemplateSection.order must be an integer")
        if self.stability not in {"static", "session", "request"}:
            raise ValueError(f"invalid prompt stability {self.stability!r}")
        if not hasattr(self.source, "resolve") or not hasattr(self.source, "describe"):
            raise TypeError("PromptTemplateSection.source must implement PromptSource")
        if (
            self.engine is not None
            and not isinstance(self.engine, str)
            and not all(
                hasattr(self.engine, attribute) for attribute in ("name", "render", "variables")
            )
        ):
            raise TypeError("PromptTemplateSection.engine must implement TemplateEngine")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("PromptTemplateSection.metadata must be a mapping")
        sections = tuple(self.sections)
        if not all(isinstance(section, PromptTemplateSection) for section in sections):
            raise TypeError(
                "PromptTemplateSection.sections must contain PromptTemplateSection values"
            )
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def literal(
        cls,
        *,
        name: str,
        content: str,
        order: int = 0,
        stability: PromptStability = "static",
        engine: str | TemplateEngine | None = None,
        metadata: Mapping[str, Any] | None = None,
        sections: Sequence[PromptTemplateSection] = (),
    ) -> PromptTemplateSection:
        """Create a template section from inline content."""
        return cls(
            name=name,
            source=LiteralSource(content),
            order=order,
            stability=stability,
            engine=engine,
            metadata=metadata or {},
            sections=tuple(sections),
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        name: str,
        selector: str | Any | None = None,
        serialize_as: str | Any | None = None,
        order: int = 0,
        stability: PromptStability = "static",
        engine: str | TemplateEngine | None = None,
        metadata: Mapping[str, Any] | None = None,
        sections: Sequence[PromptTemplateSection] = (),
        policy: ResourcePolicy | None = None,
        resolver: ResourceResolver | None = None,
    ) -> PromptTemplateSection:
        """Eagerly load a template section from a file resource."""
        return cls(
            name=name,
            source=ResourceSource.from_file(
                path,
                selector=selector,
                serialize_as=serialize_as,
                policy=policy,
                resolver=resolver,
            ),
            order=order,
            stability=stability,
            engine=engine,
            metadata=metadata or {},
            sections=tuple(sections),
        )


def _walk_template_sections(
    sections: Sequence[PromptTemplateSection],
) -> Iterator[PromptTemplateSection]:
    """Yield template sections in definition-order preorder."""
    for section in sections:
        yield section
        yield from _walk_template_sections(section.sections)


def _inspect_section(section: PromptTemplateSection) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": section.name,
        "order": section.order,
        "stability": section.stability,
        "engine": getattr(section.engine, "name", section.engine),
        "source": dict(section.source.describe()),
    }
    if section.sections:
        payload["sections"] = [_inspect_section(child) for child in section.sections]
    return payload


def _fingerprint_section(section: PromptTemplateSection) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": section.name,
        "order": section.order,
        "stability": section.stability,
        "engine": (
            section.engine if isinstance(section.engine, str) else type(section.engine).__name__
        ),
        "provenance": dict(section.source.describe()),
        "metadata": dict(section.metadata),
    }
    if section.sections:
        payload["sections"] = [_fingerprint_section(child) for child in section.sections]
    return payload


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptTemplate:
    """Reusable unresolved prompt definition."""

    sections: tuple[PromptTemplateSection, ...] = ()
    variables: tuple[PromptVariable, ...] = ()
    name: str = ""
    description: str = ""
    separator: str = "\n\n"
    layout: str | PromptLayout | None = None
    allow_extra_variables: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        sections = tuple(self.sections)
        variables = tuple(self.variables)
        if not all(isinstance(section, PromptTemplateSection) for section in sections):
            raise TypeError("PromptTemplate.sections must contain PromptTemplateSection values")
        if not all(isinstance(variable, PromptVariable) for variable in variables):
            raise TypeError("PromptTemplate.variables must contain PromptVariable values")
        if not isinstance(self.separator, str):
            raise TypeError("PromptTemplate.separator must be a string")
        if not isinstance(self.name, str) or not isinstance(self.description, str):
            raise TypeError("PromptTemplate.name and description must be strings")
        if (
            self.layout is not None
            and not isinstance(self.layout, str)
            and not hasattr(self.layout, "render")
        ):
            raise TypeError("PromptTemplate.layout must be a layout name, PromptLayout, or None")
        if not isinstance(self.allow_extra_variables, bool):
            raise TypeError("PromptTemplate.allow_extra_variables must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("PromptTemplate.metadata must be a mapping")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        name: str = "prompt",
        engine: str | TemplateEngine = "string-template",
        variables: Sequence[PromptVariable] | None = None,
        policy: ResourcePolicy | None = None,
        resolver: ResourceResolver | None = None,
    ) -> PromptTemplate:
        """Load one explicitly templated file and infer stdlib variables when possible."""
        section = PromptTemplateSection.from_file(
            path,
            name=name,
            engine=engine,
            policy=policy,
            resolver=resolver,
        )
        declared = tuple(variables or ())
        if variables is None and isinstance(section.source, ResourceSource):
            text = section.source.resolve({}).content
            inferred = template_engine(engine).variables(text)
            declared = tuple(
                PromptVariable(name=variable_name, required=True)
                for variable_name in sorted(inferred)
            )
        return cls(sections=(section,), variables=declared, name=name)

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load a versioned prompt manifest."""
        from ai_arch_toolkit.toolkit.prompts._manifest import load_prompt

        return load_prompt(path, **kwargs)

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Declared variable names in definition order."""
        return tuple(variable.name for variable in self.variables)

    @property
    def sources(self) -> tuple[Mapping[str, Any], ...]:
        """Return non-sensitive source provenance snapshots in preorder."""
        return tuple(
            section.source.describe() for section in _walk_template_sections(self.sections)
        )

    def inspect(self) -> Mapping[str, Any]:
        """Return a non-sensitive, serializable definition summary."""
        return {
            "name": self.name,
            "description": self.description,
            "fingerprint": self.fingerprint,
            "layout": getattr(self.layout, "name", self.layout or "text"),
            "sections": [_inspect_section(section) for section in self.sections],
            "variables": [
                {
                    "name": variable.name,
                    "type": variable.value_type,
                    "required": variable.required,
                    "has_default": variable.has_default,
                    "description": variable.description,
                }
                for variable in self.variables
            ],
        }

    def validate(self) -> None:
        """Validate names, variable declarations, and configured template engines."""
        section_names = [section.name for section in _walk_template_sections(self.sections)]
        duplicates = sorted({name for name in section_names if section_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "prompt section names must be unique; duplicates: "
                + ", ".join(repr(name) for name in duplicates)
            )
        variable_names = [variable.name for variable in self.variables]
        duplicate_variables = sorted(
            {name for name in variable_names if variable_names.count(name) > 1}
        )
        if duplicate_variables:
            raise ValueError(
                "prompt variable names must be unique; duplicates: "
                + ", ".join(repr(name) for name in duplicate_variables)
            )
        for section in _walk_template_sections(self.sections):
            if section.engine is not None:
                template_engine(section.engine)

    def compile(self, **variables: Any) -> Prompt:
        """Resolve variables and sources into an immutable literal prompt."""
        self.validate()
        resolved_variables = resolve_variables(
            self.variables,
            variables,
            allow_extra=self.allow_extra_variables,
        )

        def build(template_section: PromptTemplateSection) -> PromptSection:
            resolution = template_section.source.resolve(resolved_variables)
            content = resolution.content
            engine_name: str | None = None
            if template_section.engine is not None:
                active_engine = template_engine(template_section.engine)
                content = active_engine.render(content, resolved_variables)
                engine_name = active_engine.name
            metadata = {
                **dict(template_section.metadata),
                "source_provenance": dict(resolution.provenance),
            }
            if engine_name:
                metadata["template_engine"] = engine_name
                metadata["template_variables"] = tuple(sorted(resolved_variables))
            return PromptSection(
                name=template_section.name,
                content=content,
                order=template_section.order,
                stability=template_section.stability,
                metadata=metadata,
                sections=tuple(build(child) for child in template_section.sections),
            )

        sections = tuple(build(template_section) for template_section in self.sections)
        prompt = Prompt(sections=sections, separator=self.separator)
        # Validate duplicate section names at compile time, before consumers call an LLM.
        render_prompt(prompt)
        return prompt

    def render(
        self,
        *,
        layout: str | PromptLayout | None = None,
        **variables: Any,
    ) -> RenderedPrompt:
        """Compile and render the prompt definition."""
        prompt = self.compile(**variables)
        rendered = render_prompt(prompt, layout=layout if layout is not None else self.layout)
        provenance = {
            "template": self.name,
            "description": self.description,
            "variables": tuple(sorted(variables)),
            "definition_fingerprint": self.fingerprint,
            "metadata": dict(self.metadata),
        }
        return replace(rendered, provenance=MappingProxyType(provenance))

    @property
    def fingerprint(self) -> str:
        """Fingerprint the serializable definition without runtime variable values."""
        payload = {
            "name": self.name,
            "description": self.description,
            "separator": self.separator,
            "layout": self.layout if isinstance(self.layout, str) else repr(self.layout),
            "allow_extra_variables": self.allow_extra_variables,
            "metadata": dict(self.metadata),
            "sections": [_fingerprint_section(section) for section in self.sections],
            "variables": [
                {
                    "name": variable.name,
                    "type": variable.value_type,
                    "required": variable.required,
                    "has_default": variable.has_default,
                    "default": variable.default if variable.has_default else None,
                    "description": variable.description,
                    "json_schema": dict(variable.json_schema) if variable.json_schema else None,
                }
                for variable in self.variables
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["PromptTemplate", "PromptTemplateSection"]
