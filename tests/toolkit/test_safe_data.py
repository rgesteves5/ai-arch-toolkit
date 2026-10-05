"""Data files the toolkit reads stay bounded: YAML aliases cannot multiply a small file, and no
file nests deeper than the toolkit's walkers follow (G-34)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ai_arch_toolkit.toolkit._safe_data import (
    MAX_ALIAS_CHARS,
    MAX_ALIAS_NODES,
    MAX_DEPTH,
    UnsafeDataError,
    check_depth,
    load_json,
    load_toml,
    load_yaml,
)
from ai_arch_toolkit.toolkit.agents import (
    AgentManifestError,
    AgentOverrideError,
    load_agent_manifest,
)
from ai_arch_toolkit.toolkit.agents import _manifest as agent_manifest
from ai_arch_toolkit.toolkit.prompts import PromptLoadError, load_prompt
from ai_arch_toolkit.toolkit.prompts import _manifest as prompt_manifest
from ai_arch_toolkit.toolkit.resources import ResourceDecodeError, ResourceResolver

pytest.importorskip("yaml")


def bomb(levels: int = 6) -> str:
    """Ten aliases per level: 412 bytes that expand to a million strings at six levels."""
    lines = ['a0: &a0 ["x","x","x","x","x","x","x","x","x","x"]']
    lines += [f"a{i}: &a{i} [{','.join([f'*a{i - 1}'] * 10)}]" for i in range(1, levels + 1)]
    return "\n".join(lines) + "\n"


def nested_yaml(levels: int) -> str:
    return "[" * levels + "]" * levels


# ---------------------------------------------------------------- the loader


def test_an_alias_bomb_is_refused_before_it_expands() -> None:
    started = time.monotonic()

    with pytest.raises(UnsafeDataError, match="aliases"):
        load_yaml(bomb())

    assert time.monotonic() - started < 1.0


def test_anchors_and_merge_keys_still_load() -> None:
    text = """
    defaults: &defaults {timeout: 12, retries: 2}
    fast:
      <<: *defaults
      timeout: 3
    same: *defaults
    """

    data = load_yaml(text)

    assert data["fast"] == {"timeout": 3, "retries": 2}
    assert data["same"] == {"timeout": 12, "retries": 2}


def test_aliases_may_add_up_to_the_node_limit() -> None:
    # The anchored list is 100 nodes (itself and 99 scalars), and each alias to it adds 100; the
    # alias to the one-node scalar adds one more.
    copies = MAX_ALIAS_NODES // 100
    base = "one: &one y\nbase: &b [" + ",".join(["x"] * 99) + "]\n"
    base += "\n".join(f"c{i}: *b" for i in range(copies)) + "\n"

    assert load_yaml(base)[f"c{copies - 1}"][0] == "x"
    with pytest.raises(UnsafeDataError, match="aliases"):
        load_yaml(base + "extra: *one\n")


def scalar_bomb(chars: int, copies: int) -> str:
    """One long scalar, aliased ``copies`` times: few nodes, many characters."""
    return f's: &s "{"x" * chars}"\nc: [{",".join(["*s"] * copies)}]\n'


def test_an_alias_counts_the_characters_it_copies() -> None:
    # 69 KB that a serialization would turn into 300 MB.
    started = time.monotonic()

    with pytest.raises(UnsafeDataError, match="characters"):
        load_yaml(scalar_bomb(30_000, 9_990))

    assert time.monotonic() - started < 1.0


def test_aliases_may_add_up_to_the_character_limit() -> None:
    copies = MAX_ALIAS_CHARS // 1_000

    assert len(load_yaml(scalar_bomb(1_000, copies))["c"]) == copies
    with pytest.raises(UnsafeDataError, match="characters"):
        load_yaml(scalar_bomb(1_000, copies + 1))


def test_a_large_document_may_alias_as_much_as_it_holds() -> None:
    own = 3 * MAX_ALIAS_NODES
    text = "big: &b [" + ",".join(["x"] * own) + "]\ncopy: *b\n"

    assert len(load_yaml(text)["copy"]) == own


def test_a_recursive_alias_is_refused() -> None:
    with pytest.raises(UnsafeDataError, match="refers to itself"):
        load_yaml("loop: &a [*a]\n")


def merge_chain(links: int, *, listed: bool = False) -> str:
    """Each mapping merges the one before it: flat data, ``links`` merge keys deep."""
    lines = ["a0: &a0 {k0: v}"]
    for i in range(1, links + 1):
        merged = f"[*a{i - 1}]" if listed else f"*a{i - 1}"
        lines.append(f"a{i}: &a{i} {{<<: {merged}, k{i}: v}}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("listed", [False, True], ids=["mapping", "list"])
def test_merge_keys_do_not_count_as_nesting(listed: bool) -> None:
    # A merged mapping's keys land beside the mapping's own: the data stays two levels deep.
    data = load_yaml(merge_chain(60, listed=listed))

    assert data["a60"] == {f"k{i}": "v" for i in range(61)}


def test_a_chain_of_merge_keys_is_capped() -> None:
    # Enough other data that the merged copies fit the alias budget; only the chain is too long.
    filler = "filler: [" + ",".join(["x"] * 3 * MAX_ALIAS_NODES) + "]\n"
    load_yaml(merge_chain(MAX_DEPTH) + filler)

    with pytest.raises(UnsafeDataError, match="merge keys"):
        load_yaml(merge_chain(MAX_DEPTH + 1) + filler)


@pytest.mark.parametrize(
    ("load", "deep"),
    [
        (load_yaml, lambda n: "k: " + nested_yaml(n)),
        (load_json, lambda n: '{"k":' + "[" * n + "]" * n + "}"),
        (load_toml, lambda n: "k = " + "[" * n + "]" * n + "\n"),
    ],
    ids=["yaml", "json", "toml"],
)
def test_nesting_is_capped_in_every_format(load, deep) -> None:  # type: ignore[no-untyped-def]
    load(deep(MAX_DEPTH - 1))  # the key's mapping is one level more
    with pytest.raises(UnsafeDataError, match="nests deeper"):
        load(deep(MAX_DEPTH))
    with pytest.raises(UnsafeDataError, match="nests deeper"):
        load(deep(2000))


def test_a_nested_yaml_document_counts_its_levels_through_aliases() -> None:
    half = MAX_DEPTH // 2 + 1
    text = f"inner: &i {nested_yaml(half)}\nouter: {'[' * half}*i{']' * half}\n"

    with pytest.raises(UnsafeDataError, match="nests deeper"):
        load_yaml(text)


def test_check_depth_counts_the_levels_above_and_shared_values_once() -> None:
    check_depth([[]], above=MAX_DEPTH - 2)
    with pytest.raises(UnsafeDataError, match="nests deeper"):
        check_depth([[]], above=MAX_DEPTH - 1)
    # A value shared at every level is checked once per object, not once per path.
    shared: list[object] = []
    for _ in range(60):
        shared = [shared] * 10
    with pytest.raises(UnsafeDataError, match="nests deeper"):
        check_depth({"k": [shared]}, above=40)


def test_check_depth_stops_at_a_cycle() -> None:
    loop: list[object] = []
    loop.append(loop)

    check_depth({"loop": loop})  # the cycle is the caller's to refuse


def test_syntax_errors_are_left_to_the_caller() -> None:
    import yaml

    with pytest.raises(yaml.YAMLError):
        load_yaml("a: [")
    assert load_yaml("") is None


# ---------------------------------------------------------------- where files come in


def test_an_agent_manifest_refuses_the_bomb(tmp_path: Path) -> None:
    path = tmp_path / "bomb.agent.yaml"
    path.write_text("version: 1\nmetadata:\n" + _indented(bomb()), encoding="utf-8")
    started = time.monotonic()

    with pytest.raises(AgentManifestError, match="aliases"):
        load_agent_manifest(path)

    assert time.monotonic() - started < 1.0


def test_an_agent_manifest_takes_an_anchor(tmp_path: Path) -> None:
    path = tmp_path / "anchored.agent.yaml"
    path.write_text(
        "version: 1\n"
        "metadata:\n"
        "  team: &team {owner: ops, pager: true}\n"
        "  copy:\n"
        "    <<: *team\n"
        "    owner: dev\n",
        encoding="utf-8",
    )

    data = load_agent_manifest(path).as_dict()

    assert data["metadata"]["copy"] == {"owner": "dev", "pager": True}


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("deep.agent.yaml", "version: 1\nmetadata: " + nested_yaml(2000)),
        ("deep.agent.json", '{"version": 1, "metadata": ' + "[" * 2000 + "]" * 2000 + "}"),
        ("deep.agent.toml", "version = 1\nmetadata = { k = " + "[" * 2000 + "]" * 2000 + " }"),
    ],
    ids=["yaml", "json", "toml"],
)
def test_an_agent_manifest_that_nests_too_deep_is_refused(
    tmp_path: Path, name: str, text: str
) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")

    with pytest.raises(AgentManifestError, match="nests deeper"):
        load_agent_manifest(path)


def test_an_agent_manifest_refuses_aliases_to_a_long_scalar(tmp_path: Path) -> None:
    path = tmp_path / "scalar.agent.yaml"
    path.write_text("version: 1\nmetadata:\n" + _indented(scalar_bomb(30_000, 9_990)))
    started = time.monotonic()

    with pytest.raises(AgentManifestError, match="characters"):
        load_agent_manifest(path)

    assert time.monotonic() - started < 1.0


def test_an_override_cannot_nest_past_the_limit(tmp_path: Path) -> None:
    path = tmp_path / "open.agent.yaml"
    path.write_text("version: 1\noverride_policy:\n  allow: [metadata]\n", encoding="utf-8")
    manifest = load_agent_manifest(path)

    deep_path = "metadata." + ".".join(f"k{i}" for i in range(MAX_DEPTH))
    deep_value: object = []
    for _ in range(2000):
        deep_value = [deep_value]
    for overrides in ({deep_path: 1}, {"metadata.k": deep_value}):
        with pytest.raises(AgentOverrideError, match="nests deeper"):
            load_agent_manifest(path, overrides=overrides)
        with pytest.raises(AgentOverrideError, match="nests deeper"):
            manifest.with_overrides(overrides)
    fits = "metadata." + ".".join(f"k{i}" for i in range(MAX_DEPTH - 2))
    assert load_agent_manifest(path, overrides={fits: 1}).as_dict()["metadata"]


def test_each_inherited_manifest_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Eight files, each extending the next four times: 16,384 reads before, eight now.
    levels = 8
    for i in range(levels):
        parents = (
            "" if i == levels - 1 else f"extends: [{', '.join([f'm{i + 1}.agent.yaml'] * 4)}]\n"
        )
        (tmp_path / f"m{i}.agent.yaml").write_text(f"version: 1\n{parents}", encoding="utf-8")
    reads: list[Path] = []
    real = agent_manifest._read_manifest
    monkeypatch.setattr(
        agent_manifest, "_read_manifest", lambda path: (reads.append(path), real(path))[1]
    )

    load_agent_manifest(tmp_path / "m0.agent.yaml")

    assert len(reads) == levels


def test_a_shared_parent_still_counts_toward_the_inheritance_depth(tmp_path: Path) -> None:
    # entry -> base -> root, and entry -> x -> y -> base: the second path to base is too deep.
    files = {
        "entry": "extends: [base.agent.yaml, x.agent.yaml]",
        "x": "extends: [y.agent.yaml]",
        "y": "extends: [base.agent.yaml]",
        "base": "extends: [root.agent.yaml]",
        "root": "",
    }
    for name, extends in files.items():
        (tmp_path / f"{name}.agent.yaml").write_text(f"version: 1\n{extends}\n")

    load_agent_manifest(tmp_path / "entry.agent.yaml", max_inheritance_depth=5)
    with pytest.raises(AgentManifestError, match="inheritance exceeds maximum 4"):
        load_agent_manifest(tmp_path / "entry.agent.yaml", max_inheritance_depth=4)


def test_each_included_prompt_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    levels = 8
    for i in range(levels):
        include = (
            "" if i == levels - 1 else f"include: [{', '.join([f'p{i + 1}.prompt.yaml'] * 4)}]\n"
        )
        (tmp_path / f"p{i}.prompt.yaml").write_text(f"version: 1\n{include}", encoding="utf-8")
    reads: list[Path] = []
    real = prompt_manifest._manifest_resource
    monkeypatch.setattr(
        prompt_manifest,
        "_manifest_resource",
        lambda path, *rest: (reads.append(path), real(path, *rest))[1],
    )

    load_prompt(tmp_path / "p0.prompt.yaml")

    assert len(reads) == levels


def test_a_prompt_manifest_refuses_the_bomb(tmp_path: Path) -> None:
    path = tmp_path / "bomb.prompt.yaml"
    path.write_text("template: hi\nmetadata:\n" + _indented(bomb()), encoding="utf-8")

    with pytest.raises(PromptLoadError, match="aliases"):
        load_prompt(path)


def test_a_prompt_manifest_that_nests_too_deep_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "deep.prompt.yaml"
    path.write_text("template: hi\nmetadata: " + nested_yaml(2000), encoding="utf-8")

    with pytest.raises(PromptLoadError, match="nests deeper"):
        load_prompt(path)


@pytest.mark.parametrize(
    ("name", "text", "match"),
    [
        ("bomb.yaml", bomb(), "aliases"),
        ("scalar.yaml", scalar_bomb(30_000, 9_990), "characters"),
        ("loop.yaml", "loop: &a [*a]\n", "refers to itself"),
        ("deep.yaml", nested_yaml(2000), "nests deeper"),
        ("deep.json", "[" * 2000 + "]" * 2000, "nests deeper"),
    ],
    ids=["bomb", "scalar-bomb", "self-alias", "deep-yaml", "deep-json"],
)
def test_a_resource_refuses_what_would_blow_it_up(
    tmp_path: Path, name: str, text: str, match: str
) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ResourceDecodeError, match=match):
        ResourceResolver().resolve(path)


def _indented(text: str) -> str:
    return "".join(f"  {line}\n" for line in text.splitlines())
