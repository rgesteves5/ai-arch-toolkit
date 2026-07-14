"""Prompt template, variable, engine, and source tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import (
    CallableSource,
    JinjaTemplateEngine,
    KnowledgeSource,
    LiteralSource,
    MissingPromptVariableError,
    PromptTemplate,
    PromptTemplateError,
    PromptTemplateSection,
    PromptVariable,
    PromptVariableError,
    ResourceSource,
    SourceResolution,
    StringTemplateEngine,
)
from ai_arch_toolkit.toolkit.prompts._variables import resolve_variables
from ai_arch_toolkit.toolkit.resources import Resource


def test_literal_sections_never_substitute_variables() -> None:
    template = PromptTemplate(
        sections=(PromptTemplateSection.literal(name="literal", content='{"value": "${name}"}'),),
        variables=(PromptVariable(name="name", required=True),),
    )

    rendered = template.render(name="Ada")

    assert rendered.text == '{"value": "${name}"}'


def test_string_template_substitution_is_explicit_and_strict() -> None:
    template = PromptTemplate(
        sections=(
            PromptTemplateSection.literal(
                name="request",
                content="Hello ${name}; flags=${flags}; active=${active}",
                engine="string-template",
            ),
        ),
        variables=(
            PromptVariable(name="name", value_type="string", required=True),
            PromptVariable(name="flags", value_type="array", required=True),
            PromptVariable(name="active", value_type="boolean", default=True),
        ),
    )

    rendered = template.render(name="Ada", flags=["a", "b"])

    assert rendered.text == 'Hello Ada; flags=["a", "b"]; active=true'
    assert rendered.provenance["variables"] == ("flags", "name")
    assert "Ada" not in str(rendered.provenance)
    assert rendered.sections[0].metadata["template_engine"] == "string-template"


def test_missing_unknown_and_invalid_variables() -> None:
    template = PromptTemplate(
        sections=(PromptTemplateSection.literal(name="x", content="${name}", engine="string"),),
        variables=(PromptVariable(name="name", value_type="string", required=True),),
    )

    with pytest.raises(MissingPromptVariableError, match="'name'"):
        template.render()
    with pytest.raises(PromptVariableError, match="unknown prompt variables: 'extra'"):
        template.render(name="Ada", extra=True)
    with pytest.raises(PromptVariableError, match="must be string"):
        template.render(name=1)


def test_allow_extra_variables_supports_dynamic_python_templates() -> None:
    template = PromptTemplate(
        sections=(PromptTemplateSection.literal(name="x", content="${name}", engine="string"),),
        allow_extra_variables=True,
    )
    assert template.render(name="Ada").text == "Ada"


@pytest.mark.parametrize(
    ("value_type", "valid", "invalid"),
    [
        ("integer", 1, True),
        ("number", 1.5, False),
        ("boolean", True, 1),
        ("array", [1], {"a": 1}),
        ("object", {"a": 1}, [1]),
        ("string", "x", 1),
        ("any", object(), None),
    ],
)
def test_prompt_variable_types(value_type: str, valid: object, invalid: object) -> None:
    variable = PromptVariable(name="value", value_type=value_type)  # type: ignore[arg-type]
    variable.validate(valid)
    if value_type != "any":
        with pytest.raises(PromptVariableError):
            variable.validate(invalid)


def test_prompt_variable_declaration_validation() -> None:
    with pytest.raises(ValueError, match="name is required"):
        PromptVariable(name="")
    with pytest.raises(ValueError, match="invalid prompt variable type"):
        PromptVariable(name="x", value_type="date")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="required"):
        PromptVariable(name="x", required=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="description"):
        PromptVariable(name="x", description=1)  # type: ignore[arg-type]
    with pytest.raises(PromptVariableError, match="must be integer"):
        PromptVariable(name="x", value_type="integer", default="bad")
    with pytest.raises(TypeError, match="json_schema"):
        PromptVariable(name="x", json_schema=[])  # type: ignore[arg-type]


def test_json_schema_variable_validation_and_dependency_error() -> None:
    variable = PromptVariable(
        name="rating",
        value_type="integer",
        json_schema={"type": "integer", "minimum": 1, "maximum": 5},
    )
    variable.validate(3)
    with pytest.raises(PromptVariableError, match="does not match its JSON Schema"):
        variable.validate(7)
    with (
        patch.dict("sys.modules", {"jsonschema": None}),
        pytest.raises(ImportError, match=r"\[prompts\]"),
    ):
        variable.validate(3)


def test_variable_resolution_duplicate_optional_and_default_paths() -> None:
    duplicate = (PromptVariable(name="x"), PromptVariable(name="x"))
    with pytest.raises(PromptVariableError, match="duplicate prompt variable"):
        resolve_variables(duplicate, {})
    declarations = (
        PromptVariable(name="optional"),
        PromptVariable(name="defaulted", default=False),
    )
    assert resolve_variables(declarations, {}) == {"defaulted": False}


def test_duplicate_declarations_and_sections_are_rejected() -> None:
    duplicate_variables = PromptTemplate(
        variables=(PromptVariable(name="x"), PromptVariable(name="x")),
    )
    with pytest.raises(ValueError, match="variable names must be unique"):
        duplicate_variables.validate()

    duplicate_sections = PromptTemplate(
        sections=(
            PromptTemplateSection.literal(name="x", content="a"),
            PromptTemplateSection.literal(name="x", content="b"),
        )
    )
    with pytest.raises(ValueError, match="section names must be unique"):
        duplicate_sections.validate()
    with pytest.raises(ValueError, match="duplicates"):
        duplicate_sections.compile()


def test_string_engine_discovers_variables_and_validates_syntax() -> None:
    engine = StringTemplateEngine()

    assert engine.variables("${name} $other $$ literal") == frozenset({"name", "other"})
    with pytest.raises(PromptTemplateError, match="invalid string template"):
        engine.render("bad $", {})
    with pytest.raises(PromptTemplateError, match="missing variable"):
        engine.render("${missing}", {})
    with pytest.raises(PromptTemplateError, match="invalid string template"):
        engine.variables("bad $")
    assert engine.render(
        "${false}|${none}|${count}", {"false": False, "none": None, "count": 2}
    ) == ("false||2")


def test_prompt_template_from_file_infers_required_variables(tmp_path: Path) -> None:
    path = tmp_path / "request.template.md"
    path.write_text("Write a ${genre} story for ${audience}.")

    template = PromptTemplate.from_file(path, name="request")

    assert template.variable_names == ("audience", "genre")
    assert template.render(genre="mystery", audience="adults").text == (
        "Write a mystery story for adults."
    )
    assert template.sources[0]["source"] == str(path)


def test_resource_source_preserves_raw_whole_file_and_dynamic_selector(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    raw = '{"genres":{"mystery":["clues"],"fantasy":["magic"]}}'
    path.write_text(raw)
    whole = ResourceSource.from_file(path)
    selected = ResourceSource.from_file(
        path,
        selector="/genres/${genre}",
        serialize_as="markdown",
    )

    assert whole.resolve({}).content == raw
    assert selected.resolve({"genre": "fantasy"}).content == "- magic"
    assert selected.describe()["selector"] == "/genres/${genre}"


def test_knowledge_source_and_callable_source() -> None:
    registry = KnowledgeRegistry()
    registry.register("style", "Be concise.", source="style.md")
    registry.register("domain", "Use architecture terms.")
    knowledge = KnowledgeSource(
        registry=registry,
        keys=("style", "domain"),
        separator="\n",
        include_names=True,
    )
    callable_source = CallableSource(
        name="runtime", function=lambda variables: f"Task: {variables['task']}"
    )

    assert knowledge.resolve({}).content == (
        "[style]\nBe concise.\n[domain]\nUse architecture terms."
    )
    assert knowledge.resolve({}).provenance["sources"] == ("style.md", "")
    assert callable_source.resolve({"task": "review"}).content == "Task: review"
    with pytest.raises(TypeError, match="must return a string"):
        CallableSource(function=lambda variables: 1).resolve({})  # type: ignore[arg-type,return-value]


def test_source_contract_validation() -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        SourceResolution(content=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="provenance must be a mapping"):
        SourceResolution(content="x", provenance=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resource must be a Resource"):
        ResourceSource(resource=object())  # type: ignore[arg-type]
    resource = Resource.from_text("x")
    with pytest.raises(TypeError, match="selector"):
        ResourceSource(resource=resource, selector=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="serialize_as"):
        ResourceSource(resource=resource, serialize_as=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="template_selector"):
        ResourceSource(resource=resource, template_selector=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be callable"):
        CallableSource(function=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name is required"):
        CallableSource(function=lambda variables: "x", name="")


def test_knowledge_source_description_covers_render_configuration() -> None:
    registry = KnowledgeRegistry()
    source = KnowledgeSource(
        registry=registry,
        keys=("a",),
        separator="|",
        include_names=True,
    )
    assert source.describe() == {
        "kind": "knowledge",
        "keys": ("a",),
        "separator": "|",
        "include_names": True,
    }


def test_source_and_template_section_validation() -> None:
    with pytest.raises(TypeError, match=r"LiteralSource\.content"):
        LiteralSource(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name is required"):
        PromptTemplateSection(name="", source=LiteralSource("x"))
    with pytest.raises(TypeError, match="order"):
        PromptTemplateSection(name="x", source=LiteralSource("x"), order="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata"):
        PromptTemplateSection(name="x", source=LiteralSource("x"), metadata=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stability"):
        PromptTemplateSection(
            name="x",
            source=LiteralSource("x"),
            stability="daily",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="PromptSource"):
        PromptTemplateSection(name="x", source=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TemplateEngine"):
        PromptTemplateSection(name="x", source=LiteralSource("x"), engine=object())  # type: ignore[arg-type]


def test_prompt_template_configuration_validation() -> None:
    with pytest.raises(TypeError, match="sections must contain"):
        PromptTemplate(sections=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="variables must contain"):
        PromptTemplate(variables=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="separator"):
        PromptTemplate(separator=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name and description"):
        PromptTemplate(name=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="layout"):
        PromptTemplate(layout=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="allow_extra"):
        PromptTemplate(allow_extra_variables=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata"):
        PromptTemplate(metadata=[])  # type: ignore[arg-type]


def test_template_definition_fingerprint_ignores_runtime_values() -> None:
    template = PromptTemplate(
        name="greeting",
        sections=(
            PromptTemplateSection.literal(name="x", content="Hello ${name}", engine="string"),
        ),
        variables=(PromptVariable(name="name", required=True),),
    )

    first = template.render(name="Ada")
    second = template.render(name="Grace")

    assert (
        first.provenance["definition_fingerprint"] == second.provenance["definition_fingerprint"]
    )
    assert first.fingerprint != second.fingerprint


def test_template_inspect_is_serializable_and_secret_free() -> None:
    template = PromptTemplate(
        name="request",
        sections=(
            PromptTemplateSection.literal(
                name="request", content="Hello ${name}", engine="string-template"
            ),
        ),
        variables=(PromptVariable(name="name", required=True),),
    )
    inspected = template.inspect()
    assert inspected["name"] == "request"
    assert inspected["variables"][0]["name"] == "name"
    assert "Ada" not in str(inspected)


def test_definition_fingerprint_covers_semantic_template_configuration() -> None:
    baseline = PromptTemplate(
        name="greeting",
        sections=(PromptTemplateSection.literal(name="x", content="Hello"),),
        variables=(PromptVariable(name="tone", default="warm"),),
    )
    changes = (
        PromptTemplate(
            name="greeting",
            sections=(PromptTemplateSection.literal(name="x", content="Goodbye"),),
            variables=(PromptVariable(name="tone", default="warm"),),
        ),
        PromptTemplate(
            name="greeting",
            sections=(PromptTemplateSection.literal(name="x", content="Hello"),),
            variables=(PromptVariable(name="tone", default="formal"),),
        ),
        PromptTemplate(
            name="greeting",
            sections=(PromptTemplateSection.literal(name="x", content="Hello"),),
            variables=(PromptVariable(name="tone", default="warm"),),
            layout="xml",
        ),
        PromptTemplate(
            name="greeting",
            sections=(
                PromptTemplateSection.literal(
                    name="x", content="Hello", metadata={"title": "Greeting"}
                ),
            ),
            variables=(PromptVariable(name="tone", default="warm"),),
        ),
    )
    assert all(candidate.fingerprint != baseline.fingerprint for candidate in changes)


def test_custom_template_engine_object() -> None:
    class UpperEngine:
        name = "upper"

        def render(self, template, variables):
            return template.upper()

        def variables(self, template):
            return frozenset()

    template = PromptTemplate(
        sections=(PromptTemplateSection.literal(name="x", content="hello", engine=UpperEngine()),)
    )
    assert template.render().text == "HELLO"


def test_unknown_template_engine() -> None:
    template = PromptTemplate(
        sections=(PromptTemplateSection.literal(name="x", content="x", engine="unknown"),)
    )
    with pytest.raises(ValueError, match="unknown template engine"):
        template.render()


def test_jinja_engine_or_missing_dependency() -> None:
    try:
        import jinja2  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\[templates\]"):
            JinjaTemplateEngine().render("{{ value }}", {"value": "x"})
    else:
        engine = JinjaTemplateEngine()
        assert engine.variables("Hello {{ name }}") == frozenset({"name"})
        assert engine.render("Hello {{ name }}", {"name": "Ada"}) == "Hello Ada"
        with pytest.raises(PromptTemplateError, match="rendering failed"):
            engine.render("{{ missing }}", {})
        with pytest.raises(PromptTemplateError, match="invalid Jinja template"):
            engine.variables("{% if %}")


def test_jinja_missing_dependency_error_is_actionable() -> None:
    with (
        patch.dict("sys.modules", {"jinja2": None, "jinja2.sandbox": None}),
        pytest.raises(ImportError, match=r"\[templates\]"),
    ):
        JinjaTemplateEngine().render("{{ value }}", {"value": "x"})
