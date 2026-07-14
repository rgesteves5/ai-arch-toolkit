"""Versioned declarative prompt-manifest loading."""

from __future__ import annotations

import difflib
import importlib.resources
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from ai_arch_toolkit.toolkit.prompts._errors import (
    PromptIncludeCycleError,
    PromptLoadError,
    PromptValidationError,
)
from ai_arch_toolkit.toolkit.prompts._layouts import (
    JsonLayout,
    MarkdownLayout,
    PromptLayout,
    SeparatorPolicy,
    TextLayout,
    XmlLayout,
)
from ai_arch_toolkit.toolkit.prompts._sources import (
    KnowledgeSource,
    LiteralSource,
    ResourceSource,
)
from ai_arch_toolkit.toolkit.prompts._templates import PromptTemplate, PromptTemplateSection
from ai_arch_toolkit.toolkit.prompts._variables import PromptVariable
from ai_arch_toolkit.toolkit.resources import (
    JsonPointer,
    LineRange,
    MarkdownHeading,
    NamedBlock,
    ResourceError,
    ResourcePolicy,
    ResourceRef,
    ResourceResolver,
)

if TYPE_CHECKING:
    from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry

_TOP_LEVEL_FIELDS = frozenset(
    {
        "description",
        "extends",
        "include",
        "layout",
        "metadata",
        "name",
        "sections",
        "separator",
        "variables",
        "version",
    }
)
_SECTION_FIELDS = frozenset(
    {
        "content",
        "knowledge",
        "metadata",
        "name",
        "order",
        "remove",
        "replace",
        "source",
        "stability",
        "template",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _SectionOperation:
    name: str
    action: str
    section: PromptTemplateSection | None = None


def load_prompt(
    path: str | Path,
    *,
    resolver: ResourceResolver | None = None,
    knowledge: KnowledgeRegistry | None = None,
    max_include_depth: int = 16,
) -> PromptTemplate:
    """Load a versioned YAML, JSON, or TOML prompt manifest."""
    raw_path = str(path)
    if urlparse(raw_path).scheme == "package":
        return _load_package_prompt(
            raw_path,
            resolver=resolver,
            knowledge=knowledge,
            max_include_depth=max_include_depth,
        )
    return _load_prompt_path(
        Path(path),
        resolver=resolver,
        knowledge=knowledge,
        max_include_depth=max_include_depth,
    )


def _load_prompt_path(
    path: Path,
    *,
    resolver: ResourceResolver | None,
    knowledge: KnowledgeRegistry | None,
    max_include_depth: int,
) -> PromptTemplate:
    manifest_path = Path(path).expanduser().resolve()
    valid_suffixes = (".prompt.yaml", ".prompt.yml", ".prompt.json", ".prompt.toml")
    if not manifest_path.name.endswith(valid_suffixes):
        raise PromptLoadError(
            "prompt manifests must use .prompt.yaml, .prompt.yml, .prompt.json, "
            "or .prompt.toml filenames"
        )
    if max_include_depth < 1:
        raise ValueError("max_include_depth must be at least 1")
    active_resolver = resolver or ResourceResolver(
        policy=ResourcePolicy(allowed_roots=(manifest_path.parent,))
    )
    return _load_prompt(
        manifest_path,
        resolver=active_resolver,
        knowledge=knowledge,
        stack=(),
        max_include_depth=max_include_depth,
    )


def _load_package_prompt(
    uri: str,
    *,
    resolver: ResourceResolver | None,
    knowledge: KnowledgeRegistry | None,
    max_include_depth: int,
) -> PromptTemplate:
    parsed = urlparse(uri)
    package = parsed.netloc
    resource_path = unquote(parsed.path.lstrip("/"))
    if not package or not resource_path:
        raise PromptLoadError("package manifests must use package://module/path syntax")
    if any(part in {"", ".", ".."} for part in resource_path.split("/")):
        raise PromptLoadError("package manifest paths cannot contain '.' or '..' segments")
    try:
        package_root = importlib.resources.files(package)
        with importlib.resources.as_file(package_root) as root:
            template = _load_prompt_path(
                root / resource_path,
                resolver=resolver,
                knowledge=knowledge,
                max_include_depth=max_include_depth,
            )
    except (ImportError, FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise PromptLoadError(f"could not load package prompt manifest {uri!r}: {exc}") from exc
    return template


def _load_prompt(
    path: Path,
    *,
    resolver: ResourceResolver,
    knowledge: KnowledgeRegistry | None,
    stack: tuple[Path, ...],
    max_include_depth: int,
) -> PromptTemplate:
    canonical = path.expanduser().resolve()
    if canonical in stack:
        cycle = " -> ".join(str(item) for item in (*stack, canonical))
        raise PromptIncludeCycleError(f"prompt manifest cycle detected: {cycle}")
    if len(stack) >= max_include_depth:
        raise PromptValidationError(
            f"prompt manifest include depth exceeds maximum {max_include_depth}: {canonical}"
        )
    try:
        resource = resolver.resolve(canonical)
    except ImportError:
        raise
    except ResourceError as exc:
        raise PromptLoadError(f"could not load prompt manifest {canonical}: {exc}") from exc
    data = resource.data
    if not isinstance(data, Mapping):
        raise PromptValidationError(f"prompt manifest {canonical} must contain an object")
    _reject_unknown(data, _TOP_LEVEL_FIELDS, context=f"prompt manifest {canonical}")
    if data.get("version") != 1:
        raise PromptValidationError(
            f"prompt manifest {canonical} must declare version: 1; got {data.get('version')!r}"
        )

    next_stack = (*stack, canonical)
    base: PromptTemplate | None = None
    extends = data.get("extends")
    if extends is not None:
        if not isinstance(extends, str) or not extends:
            raise PromptValidationError("prompt manifest extends must be a non-empty path string")
        base = _load_prompt(
            _relative_path(canonical, extends),
            resolver=resolver,
            knowledge=knowledge,
            stack=next_stack,
            max_include_depth=max_include_depth,
        )

    includes_value = data.get("include", ())
    if isinstance(includes_value, str):
        includes = (includes_value,)
    elif isinstance(includes_value, Sequence) and not isinstance(includes_value, bytes):
        includes = tuple(includes_value)
    else:
        raise PromptValidationError("prompt manifest include must be a path or list of paths")
    included_templates: list[PromptTemplate] = []
    for include in includes:
        if not isinstance(include, str) or not include:
            raise PromptValidationError("prompt manifest include paths must be non-empty strings")
        included_templates.append(
            _load_prompt(
                _relative_path(canonical, include),
                resolver=resolver,
                knowledge=knowledge,
                stack=next_stack,
                max_include_depth=max_include_depth,
            )
        )

    variables = _merge_variables(base, included_templates)
    local_variables = _parse_variables(data.get("variables", {}), path=canonical)
    for variable in local_variables:
        variables[variable.name] = variable

    sections = list(base.sections if base else ())
    section_names = {section.name for section in sections}
    for included in included_templates:
        for section in included.sections:
            if section.name in section_names:
                raise PromptValidationError(
                    f"included prompt section {section.name!r} is duplicated in {canonical}"
                )
            sections.append(section)
            section_names.add(section.name)

    operations = _parse_sections(
        data.get("sections", ()),
        manifest_path=canonical,
        resolver=resolver,
        knowledge=knowledge,
    )
    for operation in operations:
        existing_index = next(
            (index for index, section in enumerate(sections) if section.name == operation.name),
            None,
        )
        if operation.action == "remove":
            if existing_index is None:
                raise PromptValidationError(
                    f"cannot remove unknown prompt section {operation.name!r} in {canonical}"
                )
            sections.pop(existing_index)
            section_names.remove(operation.name)
        elif operation.action == "replace":
            if existing_index is None:
                raise PromptValidationError(
                    f"cannot replace unknown prompt section {operation.name!r} in {canonical}"
                )
            assert operation.section is not None
            sections[existing_index] = operation.section
        else:
            if existing_index is not None:
                raise PromptValidationError(
                    f"prompt section {operation.name!r} is duplicated in {canonical}; "
                    "use replace: true when extending a base manifest"
                )
            assert operation.section is not None
            sections.append(operation.section)
            section_names.add(operation.name)

    try:
        _infer_manifest_variables(sections, variables)
    except (TypeError, ValueError) as exc:
        raise PromptValidationError(f"invalid prompt template configuration: {exc}") from exc
    metadata = {
        **(dict(base.metadata) if base else {}),
        **_mapping(data.get("metadata", {}), "metadata"),
    }
    try:
        layout = (
            _parse_layout(data["layout"])
            if "layout" in data
            else base.layout
            if base is not None
            else None
        )
    except PromptValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise PromptValidationError(f"invalid prompt layout: {exc}") from exc
    separator = data.get("separator", base.separator if base else "\n\n")
    if not isinstance(separator, str):
        raise PromptValidationError("prompt manifest separator must be a string")
    name = data.get("name", base.name if base else canonical.stem.removesuffix(".prompt"))
    description = data.get("description", base.description if base else "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise PromptValidationError("prompt manifest name and description must be strings")
    template = PromptTemplate(
        sections=tuple(sections),
        variables=tuple(variables.values()),
        name=name,
        description=description,
        separator=separator,
        layout=layout,
        metadata={
            **metadata,
            "manifest": str(canonical),
            "manifest_fingerprint": resource.fingerprint,
        },
    )
    try:
        template.validate()
    except (TypeError, ValueError) as exc:
        raise PromptValidationError(f"invalid prompt manifest {canonical}: {exc}") from exc
    return template


def _parse_variables(value: Any, *, path: Path) -> tuple[PromptVariable, ...]:
    if not isinstance(value, Mapping):
        raise PromptValidationError(f"prompt manifest variables must be an object in {path}")
    variables: list[PromptVariable] = []
    for name, config in value.items():
        if not isinstance(name, str) or not name:
            raise PromptValidationError("prompt variable names must be non-empty strings")
        if isinstance(config, str):
            config = {"type": config}
        if not isinstance(config, Mapping):
            raise PromptValidationError(f"prompt variable {name!r} must be an object or type name")
        allowed = {"default", "description", "json_schema", "required", "type"}
        _reject_unknown(config, allowed, context=f"prompt variable {name!r}")
        kwargs: dict[str, Any] = {
            "name": name,
            "value_type": config.get("type", "any"),
            "required": config.get("required", False),
            "description": config.get("description", ""),
            "json_schema": config.get("json_schema"),
        }
        if "default" in config:
            kwargs["default"] = config["default"]
        try:
            variables.append(PromptVariable(**kwargs))
        except (TypeError, ValueError) as exc:
            raise PromptValidationError(f"invalid prompt variable {name!r}: {exc}") from exc
    return tuple(variables)


def _parse_sections(
    value: Any,
    *,
    manifest_path: Path,
    resolver: ResourceResolver,
    knowledge: KnowledgeRegistry | None,
) -> tuple[_SectionOperation, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise PromptValidationError("prompt manifest sections must be a list")
    operations: list[_SectionOperation] = []
    for index, config in enumerate(value):
        if not isinstance(config, Mapping):
            raise PromptValidationError(f"prompt section at index {index} must be an object")
        _reject_unknown(config, _SECTION_FIELDS, context=f"prompt section at index {index}")
        name = config.get("name")
        if not isinstance(name, str) or not name:
            raise PromptValidationError(f"prompt section at index {index} requires a name")
        remove = config.get("remove", False)
        replace = config.get("replace", False)
        if not isinstance(remove, bool) or not isinstance(replace, bool):
            raise PromptValidationError(f"section {name!r} remove/replace flags must be booleans")
        if remove and replace:
            raise PromptValidationError(f"section {name!r} cannot both remove and replace")
        if remove:
            content_fields = {"content", "knowledge", "source", "template"} & set(config)
            if content_fields:
                raise PromptValidationError(f"removed section {name!r} cannot define content")
            operations.append(_SectionOperation(name=name, action="remove"))
            continue
        source_fields = {"content", "knowledge", "source", "template"} & set(config)
        if len(source_fields) != 1:
            raise PromptValidationError(
                f"section {name!r} must define exactly one of content, knowledge, source, template"
            )
        try:
            engine: str | None = None
            if "content" in config:
                content = config["content"]
                if not isinstance(content, str):
                    raise PromptValidationError(f"section {name!r} content must be a string")
                source = LiteralSource(content)
            elif "source" in config:
                source = _parse_resource_source(
                    config["source"], manifest_path=manifest_path, resolver=resolver
                )
            elif "template" in config:
                source, engine = _parse_template_source(
                    config["template"], manifest_path=manifest_path, resolver=resolver
                )
            else:
                if knowledge is None:
                    raise PromptValidationError(
                        f"section {name!r} uses knowledge but load_prompt() received no registry"
                    )
                source = _parse_knowledge_source(config["knowledge"], registry=knowledge)
            section = PromptTemplateSection(
                name=name,
                source=source,
                order=config.get("order", 0),
                stability=config.get("stability", "static"),
                engine=engine,
                metadata=_mapping(config.get("metadata", {}), f"section {name!r} metadata"),
            )
        except (PromptLoadError, PromptValidationError):
            raise
        except (TypeError, ValueError) as exc:
            raise PromptValidationError(f"invalid prompt section {name!r}: {exc}") from exc
        operations.append(
            _SectionOperation(name=name, action="replace" if replace else "add", section=section)
        )
    return tuple(operations)


def _parse_resource_source(
    value: Any,
    *,
    manifest_path: Path,
    resolver: ResourceResolver,
) -> ResourceSource:
    if isinstance(value, str):
        config: Mapping[str, Any] = {"path": value}
    elif isinstance(value, Mapping):
        config = value
    else:
        raise PromptValidationError("prompt section source must be a path or object")
    _reject_unknown(
        config,
        {"media_type", "path", "select", "serialize_as"},
        context="prompt section source",
    )
    path = config.get("path")
    if not isinstance(path, str) or not path:
        raise PromptValidationError("prompt section source requires a non-empty path")
    selector = _parse_selector(config.get("select"))
    try:
        source_ref = (
            ResourceRef(
                uri=path,
                media_type=config.get("media_type"),
            )
            if urlparse(path).scheme
            else ResourceRef.from_path(
                _relative_path(manifest_path, path),
                media_type=config.get("media_type"),
            )
        )
        resource = resolver.resolve(source_ref)
    except ResourceError as exc:
        raise PromptLoadError(
            f"could not load section source {path!r} from {manifest_path}: {exc}"
        ) from exc
    return ResourceSource(
        resource=resource,
        selector=selector,
        serialize_as=(
            resolver.serializers.resolve(config["serialize_as"])
            if "serialize_as" in config
            else None
        ),
    )


def _parse_template_source(
    value: Any,
    *,
    manifest_path: Path,
    resolver: ResourceResolver,
) -> tuple[LiteralSource | ResourceSource, str]:
    if isinstance(value, str):
        config: Mapping[str, Any] = {"path": value}
    elif isinstance(value, Mapping):
        config = value
    else:
        raise PromptValidationError("prompt section template must be a path or object")
    _reject_unknown(
        config,
        {"content", "engine", "path", "select", "serialize_as"},
        context="prompt section template",
    )
    engine = config.get("engine", "string-template")
    if not isinstance(engine, str):
        raise PromptValidationError("prompt template engine must be a string")
    has_path = "path" in config
    has_content = "content" in config
    if has_path == has_content:
        raise PromptValidationError("prompt template must define exactly one of path or content")
    if has_content:
        content = config["content"]
        if not isinstance(content, str):
            raise PromptValidationError("inline prompt template content must be a string")
        return LiteralSource(content), engine
    return (
        _parse_resource_source(
            {
                "path": config["path"],
                **({"select": config["select"]} if "select" in config else {}),
                **({"serialize_as": config["serialize_as"]} if "serialize_as" in config else {}),
            },
            manifest_path=manifest_path,
            resolver=resolver,
        ),
        engine,
    )


def _parse_knowledge_source(value: Any, *, registry: KnowledgeRegistry) -> KnowledgeSource:
    if isinstance(value, str):
        config: Mapping[str, Any] = {"keys": [value]}
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        config = {"keys": value}
    elif isinstance(value, Mapping):
        config = value
    else:
        raise PromptValidationError("knowledge source must be a key, key list, or object")
    _reject_unknown(
        config,
        {"include_names", "keys", "separator"},
        context="knowledge source",
    )
    keys = config.get("keys")
    if not isinstance(keys, Sequence) or isinstance(keys, str | bytes):
        raise PromptValidationError("knowledge source keys must be a list")
    return KnowledgeSource(
        registry=registry,
        keys=tuple(keys),
        separator=config.get("separator", "\n\n---\n\n"),
        include_names=config.get("include_names", False),
    )


def _parse_selector(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        raise PromptValidationError("resource selector must be a string or object")
    selector_type = value.get("type")
    if selector_type == "json_pointer":
        _reject_unknown(value, {"type", "value"}, context="JSON Pointer selector")
        return JsonPointer(value.get("value", ""))
    if selector_type == "heading":
        _reject_unknown(
            value,
            {"heading", "include_heading", "occurrence", "type"},
            context="Markdown heading selector",
        )
        return MarkdownHeading(
            heading=value.get("heading", ""),
            occurrence=value.get("occurrence"),
            include_heading=value.get("include_heading", False),
        )
    if selector_type == "lines":
        _reject_unknown(value, {"end", "start", "type"}, context="line selector")
        return LineRange(start=value.get("start", 0), end=value.get("end"))
    if selector_type == "block":
        _reject_unknown(
            value,
            {"end_marker", "include_markers", "start_marker", "type"},
            context="named block selector",
        )
        return NamedBlock(
            start_marker=value.get("start_marker", ""),
            end_marker=value.get("end_marker", ""),
            include_markers=value.get("include_markers", False),
        )
    raise PromptValidationError(
        "unknown resource selector type; expected one of: block, heading, json_pointer, lines"
    )


def _parse_layout(value: Any) -> str | PromptLayout | None:
    if value is None:
        return value
    if isinstance(value, str):
        if value not in {"json", "markdown", "text", "xml"}:
            raise PromptValidationError(
                "unknown prompt layout; expected one of: json, markdown, text, xml"
            )
        return value
    if not isinstance(value, Mapping):
        raise PromptValidationError("prompt layout must be a name or object")
    layout_type = value.get("type")
    if not isinstance(layout_type, str):
        raise PromptValidationError("prompt layout object requires a type")
    if layout_type == "text":
        _reject_unknown(
            value,
            {"after", "before", "between", "separator", "type"},
            context="text layout",
        )
        return TextLayout(separator=_parse_separator_policy(value))
    if layout_type == "markdown":
        _reject_unknown(
            value,
            {
                "after",
                "before",
                "between",
                "heading_level",
                "include_headings",
                "separator",
                "type",
            },
            context="Markdown layout",
        )
        return MarkdownLayout(
            heading_level=value.get("heading_level", 2),
            separator=_parse_separator_policy(value),
            include_headings=value.get("include_headings", True),
        )
    if layout_type == "xml":
        _reject_unknown(
            value,
            {
                "include_stability",
                "metadata_attributes",
                "root_tag",
                "section_tag",
                "separator",
                "type",
            },
            context="XML layout",
        )
        return XmlLayout(
            root_tag=value.get("root_tag", "prompt"),
            section_tag=value.get("section_tag", "section"),
            separator=value.get("separator", "\n"),
            include_stability=value.get("include_stability", False),
            metadata_attributes=tuple(value.get("metadata_attributes", ())),
        )
    if layout_type == "json":
        _reject_unknown(
            value,
            {"ensure_ascii", "include_stability", "indent", "mode", "type"},
            context="JSON layout",
        )
        return JsonLayout(
            indent=value.get("indent", 2),
            include_stability=value.get("include_stability", False),
            ensure_ascii=value.get("ensure_ascii", False),
            mode=value.get("mode", "array"),
        )
    raise PromptValidationError(
        "unknown prompt layout type; expected one of: json, markdown, text, xml"
    )


def _parse_separator_policy(config: Mapping[str, Any]) -> str | SeparatorPolicy:
    separator = config.get("separator", "\n\n")
    between_value = config.get("between", ())
    before = _parse_named_separators(config.get("before", {}), context="layout before")
    after = _parse_named_separators(config.get("after", {}), context="layout after")
    if not isinstance(between_value, Sequence) or isinstance(between_value, str | bytes):
        raise PromptValidationError("layout between must be a list of boundary objects")
    if not between_value and not before and not after:
        return separator
    between: dict[tuple[str, str], str] = {}
    for boundary in between_value:
        if not isinstance(boundary, Mapping):
            raise PromptValidationError("layout boundary must be an object")
        _reject_unknown(boundary, {"from", "separator", "to"}, context="layout boundary")
        previous = boundary.get("from")
        current = boundary.get("to")
        boundary_separator = boundary.get("separator")
        if not all(isinstance(item, str) for item in (previous, current, boundary_separator)):
            raise PromptValidationError("layout boundary from, to, and separator must be strings")
        assert isinstance(previous, str)
        assert isinstance(current, str)
        assert isinstance(boundary_separator, str)
        between[(previous, current)] = boundary_separator
    return SeparatorPolicy(default=separator, between=between, before=before, after=after)


def _parse_named_separators(value: Any, *, context: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise PromptValidationError(f"{context} must be an object")
    if not all(
        isinstance(name, str) and isinstance(separator, str) for name, separator in value.items()
    ):
        raise PromptValidationError(f"{context} must map section names to strings")
    return dict(value)


def _merge_variables(
    base: PromptTemplate | None,
    included: Sequence[PromptTemplate],
) -> dict[str, PromptVariable]:
    merged = {variable.name: variable for variable in base.variables} if base else {}
    for template in included:
        for variable in template.variables:
            if variable.name in merged:
                raise PromptValidationError(
                    f"included prompt variable {variable.name!r} is duplicated"
                )
            merged[variable.name] = variable
    return merged


def _infer_manifest_variables(
    sections: Sequence[PromptTemplateSection],
    variables: dict[str, PromptVariable],
) -> None:
    from ai_arch_toolkit.toolkit.prompts._template_engines import template_engine

    inferred: set[str] = set()
    for section in sections:
        if section.engine is not None:
            source = section.source
            if isinstance(source, LiteralSource):
                inferred.update(template_engine(section.engine).variables(source.content))
            elif isinstance(source, ResourceSource) and source.selector is None:
                text = source.resource.text
                if text is not None:
                    inferred.update(template_engine(section.engine).variables(text))
        if isinstance(section.source, ResourceSource) and isinstance(section.source.selector, str):
            inferred.update(template_engine("string-template").variables(section.source.selector))
    for name in sorted(inferred - set(variables)):
        variables[name] = PromptVariable(name=name, required=True)


def _relative_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptValidationError(f"{context} must be an object")
    return value


def _reject_unknown(
    value: Mapping[Any, Any],
    allowed: set[str] | frozenset[str],
    *,
    context: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if not unknown:
        return
    details: list[str] = []
    for name in unknown:
        match = difflib.get_close_matches(name, allowed, n=1)
        details.append(f"{name!r}" + (f" (did you mean {match[0]!r}?)" if match else ""))
    raise PromptValidationError(f"unknown fields in {context}: {', '.join(details)}")


__all__ = ["load_prompt"]
