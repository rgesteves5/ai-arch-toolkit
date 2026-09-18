"""Versioned declarative prompt-manifest loading."""

from __future__ import annotations

import importlib.resources
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from ai_arch_toolkit.toolkit._shape import ShapeError
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
from ai_arch_toolkit.toolkit.prompts._manifest_shape import PROMPT_MANIFEST
from ai_arch_toolkit.toolkit.prompts._sources import (
    KnowledgeSource,
    LiteralSource,
    ResourceSource,
)
from ai_arch_toolkit.toolkit.prompts._templates import (
    PromptTemplate,
    PromptTemplateSection,
    _walk_template_sections,
)
from ai_arch_toolkit.toolkit.prompts._variables import PromptVariable
from ai_arch_toolkit.toolkit.resources import (
    JsonPointer,
    LineRange,
    MarkdownHeading,
    NamedBlock,
    Resource,
    ResourceError,
    ResourcePolicy,
    ResourceRef,
    ResourceResolver,
)

if TYPE_CHECKING:
    from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry


@dataclass(frozen=True, slots=True, kw_only=True)
class _SectionOperation:
    name: str
    action: str
    section: PromptTemplateSection | None = None
    operations: tuple[_SectionOperation, ...] = ()


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
    resource = _manifest_resource(canonical, resolver, stack, max_include_depth)
    data: Mapping[str, Any] = resource.data

    def load(value: str) -> PromptTemplate:
        return _load_prompt(
            _relative_path(canonical, value),
            resolver=resolver,
            knowledge=knowledge,
            stack=(*stack, canonical),
            max_include_depth=max_include_depth,
        )

    extends = data.get("extends")
    base = load(extends) if extends is not None else None
    include = data.get("include", ())
    included = [load(item) for item in ((include,) if isinstance(include, str) else include)]
    variables = _merge_variables(base, included)
    for variable in _parse_variables(data.get("variables", {})):
        variables[variable.name] = variable
    operations = _parse_sections(
        data.get("sections", ()),
        manifest_path=canonical,
        resolver=resolver,
        knowledge=knowledge,
    )
    sections = _apply_section_operations(
        _inherited_sections(base, included, canonical), operations, canonical
    )
    try:
        _infer_manifest_variables(sections, variables)
    except (TypeError, ValueError) as exc:
        raise PromptValidationError(f"invalid prompt template configuration: {exc}") from exc
    return _template(data, base, sections, variables, canonical, resource.fingerprint)


def _manifest_resource(
    canonical: Path,
    resolver: ResourceResolver,
    stack: tuple[Path, ...],
    max_include_depth: int,
) -> Resource:
    """The manifest file at ``canonical``, read and checked against ``PROMPT_MANIFEST``."""
    if canonical in stack:
        cycle = " -> ".join(str(item) for item in (*stack, canonical))
        raise PromptIncludeCycleError(f"prompt manifest cycle detected: {cycle}")
    if len(stack) >= max_include_depth:
        raise PromptValidationError(
            f"prompt manifest include depth exceeds maximum {max_include_depth}: {canonical}"
        )
    try:
        resource = resolver.resolve(canonical)
    except ResourceError as exc:
        raise PromptLoadError(f"could not load prompt manifest {canonical}: {exc}") from exc
    if not isinstance(resource.data, Mapping):
        raise PromptValidationError(f"prompt manifest {canonical} must contain an object")
    try:
        PROMPT_MANIFEST.check(resource.data)
    except ShapeError as exc:
        raise PromptValidationError(f"prompt manifest {canonical}: {exc}") from exc
    return resource


def _inherited_sections(
    base: PromptTemplate | None, included: Sequence[PromptTemplate], canonical: Path
) -> tuple[PromptTemplateSection, ...]:
    """The base manifest's sections, then each included one's; no name twice."""
    sections: list[PromptTemplateSection] = list(base.sections if base else ())
    names = {section.name for section in _walk_template_sections(sections)}
    for template in included:
        for section in template.sections:
            subtree = {item.name for item in _walk_template_sections((section,))}
            duplicated = (
                section.name if section.name in names else min(subtree & names, default=None)
            )
            if duplicated is not None:
                raise PromptValidationError(
                    f"included prompt section {duplicated!r} is duplicated in {canonical}"
                )
            sections.append(section)
            names.update(subtree)
    return tuple(sections)


def _template(
    data: Mapping[str, Any],
    base: PromptTemplate | None,
    sections: Sequence[PromptTemplateSection],
    variables: Mapping[str, PromptVariable],
    canonical: Path,
    fingerprint: str,
) -> PromptTemplate:
    """The manifest's template: its own fields, falling back to its base's."""
    try:
        layout = (
            _parse_layout(data["layout"])
            if "layout" in data
            else (base.layout if base is not None else None)
        )
    except PromptValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise PromptValidationError(f"invalid prompt layout: {exc}") from exc
    template = PromptTemplate(
        sections=tuple(sections),
        variables=tuple(variables.values()),
        name=data.get("name", base.name if base else canonical.stem.removesuffix(".prompt")),
        description=data.get("description", base.description if base else ""),
        separator=data.get("separator", base.separator if base else "\n\n"),
        layout=layout,
        metadata={
            **(dict(base.metadata) if base else {}),
            **data.get("metadata", {}),
            "manifest": str(canonical),
            "manifest_fingerprint": fingerprint,
        },
    )
    try:
        template.validate()
    except (TypeError, ValueError) as exc:
        raise PromptValidationError(f"invalid prompt manifest {canonical}: {exc}") from exc
    return template


def _parse_variables(value: Mapping[str, Any]) -> tuple[PromptVariable, ...]:
    variables: list[PromptVariable] = []
    for name, config in value.items():
        declared: Mapping[str, Any] = {"type": config} if isinstance(config, str) else config
        kwargs: dict[str, Any] = {
            "name": name,
            "value_type": declared.get("type", "any"),
            "required": declared.get("required", False),
            "description": declared.get("description", ""),
            "json_schema": declared.get("json_schema"),
        }
        if "default" in declared:
            kwargs["default"] = declared["default"]
        try:
            variables.append(PromptVariable(**kwargs))
        except (TypeError, ValueError) as exc:
            raise PromptValidationError(f"invalid prompt variable {name!r}: {exc}") from exc
    return tuple(variables)


def _parse_sections(
    value: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    resolver: ResourceResolver,
    knowledge: KnowledgeRegistry | None,
) -> tuple[_SectionOperation, ...]:
    operations: list[_SectionOperation] = []
    for config in value:
        name: str = config["name"]
        action = _section_action(config)
        if action == "remove":
            operations.append(_SectionOperation(name=name, action="remove"))
        elif action == "merge":
            nested = _parse_sections(
                config["sections"],
                manifest_path=manifest_path,
                resolver=resolver,
                knowledge=knowledge,
            )
            operations.append(_SectionOperation(name=name, action="merge", operations=nested))
        else:
            section = _parse_section_definition(
                config,
                name=name,
                manifest_path=manifest_path,
                resolver=resolver,
                knowledge=knowledge,
            )
            operations.append(_SectionOperation(name=name, action=action, section=section))
    return tuple(operations)


def _section_action(config: Mapping[str, Any]) -> str:
    """What a section entry does to the inherited sections: add, replace, remove or merge."""
    name = config["name"]
    remove, replace_flag, merge = (
        config.get(flag, False) for flag in ("remove", "replace", "merge")
    )
    if remove and replace_flag:
        raise PromptValidationError(f"section {name!r} cannot both remove and replace")
    if merge and (remove or replace_flag):
        raise PromptValidationError(
            f"section {name!r} cannot combine merge with remove or replace"
        )
    if remove:
        if {"content", "knowledge", "source", "template"} & set(config):
            raise PromptValidationError(f"removed section {name!r} cannot define content")
        if "sections" in config:
            raise PromptValidationError(f"removed section {name!r} cannot define sections")
        return "remove"
    if merge:
        extra = sorted(set(config) - {"merge", "name", "sections"})
        if extra:
            raise PromptValidationError(
                f"merge section {name!r} may only define sections; found: "
                + ", ".join(repr(field) for field in extra)
            )
        if not config.get("sections"):
            raise PromptValidationError(
                f"merge section {name!r} requires a non-empty sections list"
            )
        return "merge"
    return "replace" if replace_flag else "add"


def _parse_section_definition(
    config: Mapping[str, Any],
    *,
    name: str,
    manifest_path: Path,
    resolver: ResourceResolver,
    knowledge: KnowledgeRegistry | None,
) -> PromptTemplateSection:
    subsections = _parse_subsections(
        config.get("sections", ()),
        parent_name=name,
        manifest_path=manifest_path,
        resolver=resolver,
        knowledge=knowledge,
    )
    source_fields = {"content", "knowledge", "source", "template"} & set(config)
    if len(source_fields) > 1 or (not source_fields and not subsections):
        raise PromptValidationError(
            f"section {name!r} must define exactly one of content, knowledge, source, template"
        )
    try:
        engine: str | None = None
        source: LiteralSource | ResourceSource | KnowledgeSource
        if "content" in config:
            source = LiteralSource(config["content"])
        elif "source" in config:
            source = _parse_resource_source(
                config["source"], manifest_path=manifest_path, resolver=resolver
            )
        elif "template" in config:
            source, engine = _parse_template_source(
                config["template"], manifest_path=manifest_path, resolver=resolver
            )
        elif "knowledge" in config:
            if knowledge is None:
                raise PromptValidationError(
                    f"section {name!r} uses knowledge but load_prompt() received no registry"
                )
            source = _parse_knowledge_source(config["knowledge"], registry=knowledge)
        else:
            source = LiteralSource("")
        return PromptTemplateSection(
            name=name,
            source=source,
            order=config.get("order", 0),
            stability=config.get("stability", "static"),
            engine=engine,
            metadata=config.get("metadata", {}),
            sections=subsections,
        )
    except (PromptLoadError, PromptValidationError):
        raise
    except (TypeError, ValueError) as exc:
        raise PromptValidationError(f"invalid prompt section {name!r}: {exc}") from exc


def _parse_subsections(
    value: Sequence[Mapping[str, Any]],
    *,
    parent_name: str,
    manifest_path: Path,
    resolver: ResourceResolver,
    knowledge: KnowledgeRegistry | None,
) -> tuple[PromptTemplateSection, ...]:
    subsections: list[PromptTemplateSection] = []
    for config in value:
        name: str = config["name"]
        if any(config.get(flag, False) for flag in ("merge", "remove", "replace")):
            raise PromptValidationError(
                f"subsection {name!r} of section {parent_name!r} cannot use "
                "remove, replace, or merge flags"
            )
        subsections.append(
            _parse_section_definition(
                config,
                name=name,
                manifest_path=manifest_path,
                resolver=resolver,
                knowledge=knowledge,
            )
        )
    return tuple(subsections)


def _rewrite_section(
    sections: tuple[PromptTemplateSection, ...],
    name: str,
    rewrite: Callable[[PromptTemplateSection], tuple[PromptTemplateSection, ...]],
) -> tuple[PromptTemplateSection, ...] | None:
    """Rewrite the named node anywhere in the tree; None when the name is absent."""
    for index, section in enumerate(sections):
        if section.name == name:
            return (*sections[:index], *rewrite(section), *sections[index + 1 :])
        rewritten_children = _rewrite_section(section.sections, name, rewrite)
        if rewritten_children is not None:
            updated = replace(section, sections=rewritten_children)
            return (*sections[:index], updated, *sections[index + 1 :])
    return None


def _apply_section_operations(
    sections: tuple[PromptTemplateSection, ...],
    operations: Sequence[_SectionOperation],
    canonical: Path,
) -> tuple[PromptTemplateSection, ...]:
    for operation in operations:
        existing_names = {section.name for section in _walk_template_sections(sections)}
        if operation.action == "remove":
            if operation.name not in existing_names:
                raise PromptValidationError(
                    f"cannot remove unknown prompt section {operation.name!r} in {canonical}"
                )
            rewritten = _rewrite_section(sections, operation.name, lambda _target: ())
            assert rewritten is not None
            sections = rewritten
        elif operation.action == "replace":
            if operation.name not in existing_names:
                raise PromptValidationError(
                    f"cannot replace unknown prompt section {operation.name!r} in {canonical}"
                )
            assert operation.section is not None
            replacement = operation.section
            rewritten = _rewrite_section(
                sections,
                operation.name,
                lambda _target, _section=replacement: (_section,),
            )
            assert rewritten is not None
            sections = rewritten
        elif operation.action == "merge":
            if operation.name not in existing_names:
                raise PromptValidationError(
                    f"cannot merge into unknown prompt section {operation.name!r} in {canonical}"
                )
            nested_operations = operation.operations

            def merge_children(
                target: PromptTemplateSection,
                nested: tuple[_SectionOperation, ...] = nested_operations,
            ) -> tuple[PromptTemplateSection, ...]:
                merged = _apply_section_operations(target.sections, nested, canonical)
                return (replace(target, sections=merged),)

            rewritten = _rewrite_section(sections, operation.name, merge_children)
            assert rewritten is not None
            sections = rewritten
        else:
            if operation.name in existing_names:
                raise PromptValidationError(
                    f"prompt section {operation.name!r} is duplicated in {canonical}; "
                    "use replace: true when extending a base manifest"
                )
            assert operation.section is not None
            sections = (*sections, operation.section)
    return sections


def _parse_resource_source(
    value: str | Mapping[str, Any],
    *,
    manifest_path: Path,
    resolver: ResourceResolver,
) -> ResourceSource:
    config: Mapping[str, Any] = {"path": value} if isinstance(value, str) else value
    path: str = config["path"]
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
    value: str | Mapping[str, Any],
    *,
    manifest_path: Path,
    resolver: ResourceResolver,
) -> tuple[LiteralSource | ResourceSource, str]:
    config: Mapping[str, Any] = {"path": value} if isinstance(value, str) else value
    engine: str = config.get("engine", "string-template")
    if ("path" in config) == ("content" in config):
        raise PromptValidationError("prompt template must define exactly one of path or content")
    if "content" in config:
        read = [key for key in ("select", "serialize_as") if config.get(key) is not None]
        if read:
            raise PromptValidationError(
                f"inline prompt template content cannot use {', '.join(read)}: "
                "they read a template file (path)"
            )
        return LiteralSource(config["content"]), engine
    return _parse_resource_source(
        {key: config[key] for key in ("path", "select", "serialize_as") if key in config},
        manifest_path=manifest_path,
        resolver=resolver,
    ), engine


def _parse_knowledge_source(
    value: str | Sequence[str] | Mapping[str, Any], *, registry: KnowledgeRegistry
) -> KnowledgeSource:
    if isinstance(value, str):
        config: Mapping[str, Any] = {"keys": [value]}
    elif isinstance(value, Mapping):
        config = value
    else:
        config = {"keys": value}
    return KnowledgeSource(
        registry=registry,
        keys=tuple(config["keys"]),
        separator=config.get("separator", "\n\n---\n\n"),
        include_names=config.get("include_names", False),
    )


def _parse_selector(value: str | Mapping[str, Any] | None) -> Any:
    if value is None or isinstance(value, str):
        return value
    kind = value["type"]
    if kind == "json_pointer":
        return JsonPointer(value.get("value", ""))
    if kind == "heading":
        return MarkdownHeading(
            heading=value["heading"],
            occurrence=value.get("occurrence"),
            include_heading=value.get("include_heading", False),
        )
    if kind == "lines":
        return LineRange(start=value["start"], end=value.get("end"))
    return NamedBlock(
        start_marker=value["start_marker"],
        end_marker=value["end_marker"],
        include_markers=value.get("include_markers", False),
    )


def _parse_layout(value: str | Mapping[str, Any] | None) -> str | PromptLayout | None:
    if value is None or isinstance(value, str):
        return value
    kind = value["type"]
    if kind == "text":
        return TextLayout(separator=_parse_separator_policy(value))
    if kind == "markdown":
        return MarkdownLayout(
            heading_level=value.get("heading_level", 2),
            separator=_parse_separator_policy(value),
            include_headings=value.get("include_headings", True),
        )
    if kind == "xml":
        return XmlLayout(
            root_tag=value.get("root_tag", "prompt"),
            section_tag=value.get("section_tag", "section"),
            separator=value.get("separator", "\n"),
            include_stability=value.get("include_stability", False),
            metadata_attributes=tuple(value.get("metadata_attributes", ())),
        )
    return JsonLayout(
        indent=value.get("indent", 2),
        include_stability=value.get("include_stability", False),
        ensure_ascii=value.get("ensure_ascii", False),
        mode=value.get("mode", "array"),
    )


def _parse_separator_policy(config: Mapping[str, Any]) -> str | SeparatorPolicy:
    separator: str = config.get("separator", "\n\n")
    before: dict[str, str] = dict(config.get("before", {}))
    after: dict[str, str] = dict(config.get("after", {}))
    between = {
        (boundary["from"], boundary["to"]): boundary["separator"]
        for boundary in config.get("between", ())
    }
    if not between and not before and not after:
        return separator
    return SeparatorPolicy(default=separator, between=between, before=before, after=after)


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
    for section in _walk_template_sections(tuple(sections)):
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


__all__ = ["load_prompt"]
