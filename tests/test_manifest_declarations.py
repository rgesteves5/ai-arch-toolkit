"""Each manifest has one declared shape: the loader enforces it, its JSON Schema is it (D36)."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from ai_arch_toolkit.toolkit._shape import Fields, ShapeError
from ai_arch_toolkit.toolkit.agents import AgentManifestError, load_agent_manifest
from ai_arch_toolkit.toolkit.agents import _manifest_shape as agent_shape
from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry
from ai_arch_toolkit.toolkit.prompts import _manifest_shape as prompt_shape
from ai_arch_toolkit.toolkit.prompts import load_prompt
from ai_arch_toolkit.toolkit.prompts._errors import PromptError
from tests.manifest_corpus import documents

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[1]
TOOLKIT = ROOT / "src/ai_arch_toolkit/toolkit"

_MANIFESTS: dict[str, tuple[Fields, Callable[[], dict[str, object]], Path]] = {
    "agent": (
        agent_shape.AGENT_MANIFEST,
        agent_shape.schema,
        TOOLKIT / "agents/schemas/agent-manifest-v1.schema.json",
    ),
    "prompt": (
        prompt_shape.PROMPT_MANIFEST,
        prompt_shape.schema,
        TOOLKIT / "prompts/schemas/prompt-manifest-v1.schema.json",
    ),
}


def _checked(shape: Fields, document: object) -> str | None:
    """The declaration's complaint about ``document``, or ``None`` when it has the shape."""
    try:
        shape.check(document)
    except ShapeError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("name", sorted(_MANIFESTS))
def test_the_packaged_schema_is_the_declarations(name: str) -> None:
    _, schema, packaged = _MANIFESTS[name]

    assert json.loads(packaged.read_text()) == schema(), (
        f"{packaged.name} is stale; rewrite it as the docstring of {name} schema() says"
    )
    jsonschema.Draft202012Validator.check_schema(schema())


@pytest.mark.parametrize("name", sorted(_MANIFESTS))
def test_the_schema_accepts_exactly_what_the_declaration_accepts(name: str) -> None:
    shape, schema, _ = _MANIFESTS[name]
    validator = jsonschema.Draft202012Validator(schema())
    corpus = documents(shape)
    disagreements = [
        (document, complaint)
        for document in corpus
        if validator.is_valid(document) is not ((complaint := _checked(shape, document)) is None)
    ]

    assert len(corpus) > 500
    assert disagreements == []


# What a loader may still refuse in a document of the declared shape: rules that span fields,
# the file system, registries, and the runtime types' own cross-field checks.
_AGENT_RULES = (
    "must use an .agent.yaml",  # an inherited manifest that is not an agent manifest
    "referenced file does not exist",
    "outside allowed roots",
    "must declare system or system_file, not both",
    "must reference verbatim prompt text",
    "are both set; declare the phase prompt in one place",
)
_PROMPT_RULES = (
    "could not load",  # a path that names no file
    "exactly one of",
    "cannot both remove and replace",
    "cannot combine merge",
    "cannot define content",
    "cannot define sections",
    "may only define sections",
    "requires a non-empty sections list",
    "cannot use remove, replace, or merge flags",
    "cannot remove unknown",
    "cannot replace unknown",
    "cannot merge into unknown",
    "inline prompt template content cannot use",
    "unknown template engine",
    "unknown serializer",
    "invalid prompt section",  # a selector's own cross-field rule (block markers, line range)
    "invalid prompt variable",  # a default that is not of the variable's type
    "invalid prompt manifest",  # the template's own validation
)


def _load_agent(document: object, path: Path) -> str | None:
    path.write_text(json.dumps(document))
    try:
        load_agent_manifest(path)
    except AgentManifestError as exc:
        return str(exc)
    return None


def _load_prompt(document: object, path: Path) -> str | None:
    path.write_text(json.dumps(document))
    try:
        load_prompt(path, knowledge=KnowledgeRegistry())
    except PromptError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize(
    ("name", "load", "suffix", "rules"),
    [
        ("agent", _load_agent, ".agent.json", _AGENT_RULES),
        ("prompt", _load_prompt, ".prompt.json", _PROMPT_RULES),
    ],
)
def test_the_loader_enforces_the_declaration_and_adds_only_rules_between_fields(
    tmp_path: Path,
    name: str,
    load: Callable[[object, Path], str | None],
    suffix: str,
    rules: tuple[str, ...],
) -> None:
    shape = _MANIFESTS[name][0]
    unenforced, unexplained = [], []
    for index, document in enumerate(documents(shape, depth=3)):
        if not isinstance(document, dict):
            continue  # a file that is not an object never reaches the declaration
        complaint = _checked(shape, document)
        refusal = load(document, tmp_path / f"m{index}{suffix}")
        if complaint is not None and (refusal is None or not refusal.endswith(complaint)):
            unenforced.append((document, complaint, refusal))
        elif complaint is None and refusal is not None and not any(r in refusal for r in rules):
            unexplained.append((document, refusal))

    assert unenforced == []
    assert unexplained == []


_SHAPE_WORDING = re.compile(r"must be an? |unknown fields|did you mean")


def shape_wording(source: str) -> list[int]:
    """Lines whose strings read like a shape check, which only ``toolkit/_shape.py`` words."""
    return sorted(
        {
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _SHAPE_WORDING.search(node.value)
        }
    )


@pytest.mark.parametrize("loader", ["agents/_manifest.py", "prompts/_manifest.py"])
def test_the_loaders_check_no_shape_by_hand(loader: str) -> None:
    assert shape_wording((TOOLKIT / loader).read_text()) == []


def test_the_shape_wording_detector_sees_a_hand_written_check() -> None:
    assert shape_wording('raise E(f"{where}.name must be a string")\n') == [1]
    assert shape_wording('raise E("unknown fields: " + names)\n') == [1]
    assert shape_wording('raise E("depth must be at least 1")\n') == []
