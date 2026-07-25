"""Tests for the public file-backed configurable-agent manifest loader."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from ai_arch_toolkit.toolkit.agents import (
    AgentManifestCycleError,
    AgentManifestError,
    AgentOverrideError,
    load_agent_manifest,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _base_tree(root: Path) -> Path:
    _write(root / "prompts/system.md", "Be exact.")
    _write(root / "prompts/user.md", "Question: {{ question }}")
    _write(
        root / "profiles/base.agent.yaml",
        """
        version: 1
        description: shared profile
        strategy:
          name: completion
          max_iterations: 2
        model:
          model: base-model
          temperature: 0.7
        override_policy:
          allow: [model.model, model.temperature, limits.timeout_seconds]
          deny: [strategy]
        profiles:
          fast:
            model:
              model: fast-model
              temperature: 0.2
        """,
    )
    return _write(
        root / "agents/example.agent.yaml",
        """
        version: 1
        id: example.agent
        phase: test
        extends: [../profiles/base.agent.yaml]
        prompts:
          system_manifest: ../prompts/system.md
          request_template: ../prompts/user.md
        output:
          schema: example.output
        limits:
          timeout_seconds: 30
          max_llm_calls: 2
          max_total_tokens: 5000
          max_cost: 0.25
          reserve: strict
        """,
    )


def test_inheritance_profile_overrides_paths_and_reasoning_spec(tmp_path: Path) -> None:
    path = _base_tree(tmp_path)

    manifest = load_agent_manifest(
        path,
        profile="fast",
        overrides={"model.temperature": 0.1, "limits.timeout_seconds": 5},
        allowed_roots=(tmp_path,),
    )

    data = manifest.as_dict()
    assert manifest.id == "example.agent"
    assert manifest.version == 1
    assert data["model"] == {
        "model": "fast-model",
        "temperature": 0.1,
    }
    assert data["prompts"]["system_manifest"] == str((tmp_path / "prompts/system.md").resolve())
    assert manifest.sources == (
        (tmp_path / "profiles/base.agent.yaml").resolve(),
        path.resolve(),
    )
    assert set(manifest.referenced_fingerprints) == {
        "prompts.request_template:prompts/user.md",
        "prompts.system_manifest:prompts/system.md",
    }

    spec = manifest.reasoning_spec(system="rendered system", output_schema=dict)
    assert spec.strategy == "completion"
    assert spec.system == "rendered system"
    assert spec.max_iterations == 2
    assert spec.timeout == 5
    assert spec.output_schema is dict
    budget = manifest.budget_policy()
    assert budget is not None
    assert budget.max_llm_calls == 2
    assert budget.max_total_tokens == 5000
    assert budget.max_cost == 0.25
    assert budget.reserve == "strict"


def test_strategy_compatibility_fields_become_knobs(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "react.agent.json",
        json.dumps(
            {
                "version": 1,
                "id": "react",
                "strategy": {
                    "name": "react",
                    "max_iterations": 4,
                    "parallel_tool_calls": False,
                    "final_answer_hint": False,
                    "show_turn_counter": True,
                    "strip_tools_on_final": True,
                    "knobs": {"custom": "value"},
                    "llm_kwargs": {"temperature": 0.3},
                },
            }
        ),
    )

    spec = load_agent_manifest(path).reasoning_spec()

    assert spec.max_iterations == 4
    assert spec.knobs == {
        "custom": "value",
        "parallel_tool_calls": False,
        "final_answer_hint": False,
        "show_turn_counter": True,
        "strip_tools_on_final": True,
    }
    assert spec.llm_kwargs == {"temperature": 0.3}


def test_reasoning_spec_defaults_when_strategy_is_absent(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "defaults.agent.yaml",
        """
        version: 1
        limits:
          timeout_seconds: 12
        """,
    )

    spec = load_agent_manifest(path).reasoning_spec()

    assert spec.strategy == "react"
    assert spec.max_iterations == 10
    assert spec.timeout == 12
    assert spec.knobs == {}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reserve", "strcit", "limits.reserve must be 'none' or 'strict'"),
        (
            "unpriced",
            "fail-close",
            "limits.unpriced must be 'fail_closed' or 'allow'",
        ),
    ],
)
def test_budget_modes_are_validated_by_manifest(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    path = _write(
        tmp_path / "budget.agent.yaml",
        f"""
        version: 1
        limits:
          {field}: {value}
        """,
    )

    with pytest.raises(AgentManifestError, match=message):
        load_agent_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_cost", ".nan"),
        ("max_cost", ".inf"),
        ("max_wall_s", ".inf"),
    ],
)
def test_non_finite_manifest_limits_are_rejected(tmp_path: Path, field: str, value: str) -> None:
    path = _write(
        tmp_path / "non-finite.agent.yaml",
        f"""
        version: 1
        limits:
          {field}: {value}
        """,
    )

    with pytest.raises(AgentManifestError, match="finite"):
        load_agent_manifest(path)


def test_deny_wins_and_unlisted_override_is_rejected(tmp_path: Path) -> None:
    path = _base_tree(tmp_path)

    with pytest.raises(AgentOverrideError, match="denied"):
        load_agent_manifest(
            path,
            overrides={"strategy.name": "react"},
            allowed_roots=(tmp_path,),
        )
    with pytest.raises(AgentOverrideError, match="not allowed"):
        load_agent_manifest(
            path,
            overrides={"prompts.input_adapter": "unsafe"},
            allowed_roots=(tmp_path,),
        )


def test_parent_override_cannot_bypass_descendant_deny(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "policy.agent.json",
        json.dumps(
            {
                "version": 1,
                "metadata": {"protected": "original", "public": "original"},
                "override_policy": {
                    "allow": ["metadata"],
                    "deny": ["metadata.protected"],
                },
            }
        ),
    )

    with pytest.raises(AgentOverrideError, match="denied"):
        load_agent_manifest(
            path,
            overrides={"metadata": {"protected": "bypassed", "public": "changed"}},
        )


def test_overlapping_override_paths_are_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "overlap.agent.json",
        json.dumps(
            {
                "version": 1,
                "override_policy": {"allow": ["metadata"]},
            }
        ),
    )

    with pytest.raises(AgentOverrideError, match="overlapping"):
        load_agent_manifest(
            path,
            overrides={"metadata": {"value": 1}, "metadata.value": 2},
        )


def test_resolved_manifest_can_apply_later_overrides(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_base_tree(tmp_path), allowed_roots=(tmp_path,))

    overridden = manifest.with_overrides({"model.temperature": 0.4})

    assert overridden.as_dict()["model"]["temperature"] == 0.4
    assert overridden.fingerprint != manifest.fingerprint
    assert manifest.as_dict()["model"]["temperature"] == 0.7


def test_override_values_are_revalidated(tmp_path: Path) -> None:
    path = _base_tree(tmp_path)

    with pytest.raises(AgentManifestError, match="between 0 and 2"):
        load_agent_manifest(
            path,
            overrides={"model.temperature": 99},
            allowed_roots=(tmp_path,),
        )


def test_unknown_fields_and_profiles_fail_strictly(tmp_path: Path) -> None:
    invalid = _write(
        tmp_path / "invalid.agent.yaml",
        """
        version: 1
        stratgey:
          name: react
        """,
    )
    with pytest.raises(AgentManifestError, match="unknown fields: stratgey"):
        load_agent_manifest(invalid)

    valid = _base_tree(tmp_path)
    with pytest.raises(AgentManifestError, match="unknown agent profile"):
        load_agent_manifest(valid, profile="missing", allowed_roots=(tmp_path,))


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_version_must_be_the_exact_integer_one(tmp_path: Path, version: object) -> None:
    path = _write(
        tmp_path / "version.agent.json",
        json.dumps({"version": version}),
    )

    with pytest.raises(AgentManifestError, match="integer version"):
        load_agent_manifest(path)


def test_inheritance_cycles_are_reported(tmp_path: Path) -> None:
    first = _write(
        tmp_path / "first.agent.yaml",
        """
        version: 1
        extends: second.agent.yaml
        """,
    )
    _write(
        tmp_path / "second.agent.yaml",
        """
        version: 1
        extends: first.agent.yaml
        """,
    )

    with pytest.raises(AgentManifestCycleError, match="cycle detected"):
        load_agent_manifest(first)


def test_inherited_manifest_must_use_agent_suffix(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        """
        version: 1
        strategy:
          name: completion
        """,
    )
    child = _write(
        tmp_path / "child.agent.yaml",
        """
        version: 1
        extends: base.yaml
        """,
    )

    with pytest.raises(AgentManifestError, match=r"inherited manifest.*must use"):
        load_agent_manifest(child)


def test_relative_paths_cannot_escape_allowed_roots(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    outside = _write(tmp_path / "outside.md", "secret")
    path = _write(
        root / "escape.agent.yaml",
        f"""
        version: 1
        prompts:
          system_manifest: ../../{outside.name}
        """,
    )

    with pytest.raises(AgentManifestError, match="outside allowed roots"):
        load_agent_manifest(path, allowed_roots=(root,))


def test_embedded_profile_paths_resolve_against_declaring_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    system = _write(root / "prompts/profile.md", "Profile prompt")
    _write(
        root / "profiles/nested/base.agent.yaml",
        """
        version: 1
        profiles:
          production:
            prompts:
              system_manifest: ../../prompts/profile.md
        """,
    )
    entry = _write(
        root / "agents/entry.agent.yaml",
        """
        version: 1
        extends: ../profiles/nested/base.agent.yaml
        """,
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    manifest = load_agent_manifest(
        entry,
        profile="production",
        allowed_roots=(root,),
    )

    assert manifest.as_dict()["prompts"]["system_manifest"] == str(system.resolve())
    assert set(manifest.referenced_fingerprints) == {"prompts.system_manifest:prompts/profile.md"}


def test_path_override_resolves_against_entry_manifest(tmp_path: Path) -> None:
    root = tmp_path / "project"
    original = _write(root / "prompts/original.md", "Original")
    replacement = _write(root / "prompts/replacement.md", "Replacement")
    entry = _write(
        root / "agents/entry.agent.yaml",
        """
        version: 1
        prompts:
          system_manifest: ../prompts/original.md
        override_policy:
          allow: [prompts.system_manifest]
        """,
    )
    manifest = load_agent_manifest(entry, allowed_roots=(root,))
    assert manifest.as_dict()["prompts"]["system_manifest"] == str(original.resolve())

    overridden = manifest.with_overrides({"prompts.system_manifest": "../prompts/replacement.md"})

    assert overridden.as_dict()["prompts"]["system_manifest"] == str(replacement.resolve())
    assert set(overridden.referenced_fingerprints) == {
        "prompts.system_manifest:prompts/replacement.md"
    }
    assert overridden.fingerprint != manifest.fingerprint


def test_path_override_cannot_escape_allowed_roots(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    _write(root / "prompts/original.md", "Original")
    outside = _write(tmp_path / "outside.md", "Outside")
    entry = _write(
        root / "agents/entry.agent.yaml",
        """
        version: 1
        prompts:
          system_manifest: ../prompts/original.md
        override_policy:
          allow: [prompts.system_manifest]
        """,
    )
    manifest = load_agent_manifest(entry, allowed_roots=(root,))

    with pytest.raises(AgentManifestError, match="outside allowed roots"):
        manifest.with_overrides({"prompts.system_manifest": f"../../{outside.name}"})


def test_fingerprint_includes_referenced_content_but_not_machine_root(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first_manifest = _base_tree(first)
    second_manifest = _base_tree(second)

    first_loaded = load_agent_manifest(first_manifest, allowed_roots=(first,))
    second_loaded = load_agent_manifest(second_manifest, allowed_roots=(second,))
    assert first_loaded.fingerprint == second_loaded.fingerprint

    (second / "prompts/system.md").write_text("Be creative.\n", encoding="utf-8")
    changed = load_agent_manifest(second_manifest, allowed_roots=(second,))
    assert changed.fingerprint != first_loaded.fingerprint


def test_source_fingerprints_keep_same_relative_path_from_distinct_roots(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    parent = _write(
        second_root / "same.agent.json",
        json.dumps({"version": 1, "metadata": {"source": "parent"}}),
    )
    child = _write(
        first_root / "same.agent.json",
        json.dumps(
            {
                "version": 1,
                "extends": str(parent),
                "metadata": {"source": "child"},
            }
        ),
    )

    manifest = load_agent_manifest(
        child,
        allowed_roots=(first_root, second_root),
    )

    assert len(manifest.sources) == 2
    assert set(manifest.source_fingerprints) == {
        "root[0]:same.agent.json",
        "root[1]:same.agent.json",
    }


def test_secret_like_fields_never_enter_fingerprint(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "secret.agent.yaml",
        """
        version: 1
        metadata:
          api_key: do-not-store-this
        """,
    )

    with pytest.raises(AgentManifestError, match="secret-like field"):
        load_agent_manifest(path)


def test_secret_like_fields_in_unselected_profiles_are_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "profile-secret.agent.json",
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "unused": {
                        "metadata": {"api_key": "do-not-store-this"},
                    }
                },
            }
        ),
    )

    with pytest.raises(AgentManifestError, match="secret-like field"):
        load_agent_manifest(path)


def test_override_values_must_be_canonical_json_like_data(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "canonical.agent.json",
        json.dumps(
            {
                "version": 1,
                "override_policy": {"allow": ["metadata.runtime"]},
            }
        ),
    )

    with pytest.raises(AgentManifestError, match="unsupported value type object"):
        load_agent_manifest(path, overrides={"metadata.runtime": object()})


def test_equivalent_canonical_overrides_have_the_same_fingerprint(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "deterministic.agent.json",
        json.dumps(
            {
                "version": 1,
                "override_policy": {"allow": ["metadata"]},
            }
        ),
    )

    first = load_agent_manifest(path, overrides={"metadata": {"alpha": 1, "beta": [2, 3]}})
    second = load_agent_manifest(path, overrides={"metadata": {"beta": [2, 3], "alpha": 1}})

    assert first.fingerprint == second.fingerprint


def test_invalid_override_keys_raise_domain_error_before_sorting(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "override-keys.agent.json",
        json.dumps(
            {
                "version": 1,
                "override_policy": {"allow": ["metadata"]},
            }
        ),
    )
    invalid = {1: "invalid", "metadata.value": "valid"}

    with pytest.raises(AgentOverrideError, match="non-empty strings"):
        load_agent_manifest(path, overrides=invalid)  # type: ignore[arg-type]
