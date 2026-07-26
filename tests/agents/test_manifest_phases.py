"""Tests for the ``strategy.phases`` section of agent manifests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ai_arch_toolkit.toolkit.agents import (
    AgentManifestError,
    AgentOverrideError,
    load_agent_manifest,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _phases_tree(root: Path) -> Path:
    _write(root / "prompts/planner.md", "Plan carefully.")
    return _write(
        root / "agents/phased.agent.yaml",
        """
        version: 1
        id: phased.agent
        strategy:
          name: plan_execute
          system: base
          phases:
            planner:
              system_file: ../prompts/planner.md
              model:
                model: claude-haiku-4-5
                temperature: 0
            solver:
              system: Solve tersely.
        """,
    )


def test_phases_fold_into_reasoning_spec_knobs(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_phases_tree(tmp_path), allowed_roots=(tmp_path,))

    spec = manifest.reasoning_spec()

    assert spec.knobs["planner_system"].strip() == "Plan carefully."
    assert spec.knobs["solver_system"] == "Solve tersely."


def test_phase_models_accessor(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_phases_tree(tmp_path), allowed_roots=(tmp_path,))

    assert manifest.phase_models() == {"planner": {"model": "claude-haiku-4-5", "temperature": 0}}


def test_phase_prompt_file_changes_fingerprint(tmp_path: Path) -> None:
    path = _phases_tree(tmp_path)
    first = load_agent_manifest(path, allowed_roots=(tmp_path,))

    _write(tmp_path / "prompts/planner.md", "Plan differently.")
    second = load_agent_manifest(path, allowed_roots=(tmp_path,))

    assert first.fingerprint != second.fingerprint
    key = next(k for k in first.referenced_fingerprints if "strategy.phases.planner" in k)
    assert first.referenced_fingerprints[key] != second.referenced_fingerprints[key]


def test_phase_system_and_file_conflict_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "prompts/p.md", "x")
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          phases:
            planner:
              system: inline
              system_file: prompts/p.md
        """,
    )
    with pytest.raises(AgentManifestError, match="not both"):
        load_agent_manifest(path, allowed_roots=(tmp_path,))


def test_phase_knob_ambiguity_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          knobs:
            planner_system: from knobs
          phases:
            planner:
              system: from phases
        """,
    )
    with pytest.raises(AgentManifestError, match="both set"):
        load_agent_manifest(path, allowed_roots=(tmp_path,))


def test_phase_unknown_field_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          phases:
            planner:
              sytem: typo
        """,
    )
    with pytest.raises(AgentManifestError, match="unknown fields"):
        load_agent_manifest(path, allowed_roots=(tmp_path,))


def test_phase_file_outside_roots_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "outside.md", "secret prompt")
    path = _write(
        tmp_path / "agents/a.agent.yaml",
        """
        version: 1
        strategy:
          phases:
            planner:
              system_file: ../outside.md
        """,
    )
    with pytest.raises(AgentManifestError, match="outside allowed roots"):
        load_agent_manifest(path, allowed_roots=(tmp_path / "agents",))


def test_phase_model_temperature_validated(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          phases:
            planner:
              model:
                temperature: 5
        """,
    )
    with pytest.raises(AgentManifestError, match="between 0 and 2"):
        load_agent_manifest(path, allowed_roots=(tmp_path,))


def test_secret_in_phase_model_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          phases:
            planner:
              model:
                api_key: sk-nope
        """,
    )
    with pytest.raises(AgentManifestError, match="secret-like"):
        load_agent_manifest(path, allowed_roots=(tmp_path,))


def test_phase_override_paths_governed(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          name: plan_execute
          phases:
            planner:
              model:
                model: base-model
        override_policy:
          allow: [strategy.phases.planner.model.model]
        """,
    )
    manifest = load_agent_manifest(
        path,
        overrides={"strategy.phases.planner.model.model": "claude-haiku-4-5"},
        allowed_roots=(tmp_path,),
    )
    assert manifest.phase_models()["planner"]["model"] == "claude-haiku-4-5"

    with pytest.raises(AgentOverrideError, match="not allowed"):
        load_agent_manifest(
            path,
            overrides={"strategy.phases.solver.system": "x"},
            allowed_roots=(tmp_path,),
        )


def test_system_file_rejects_prompt_manifests(tmp_path: Path) -> None:
    _write(tmp_path / "p.prompt.yaml", "version: 1")
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          phases:
            planner:
              system_file: p.prompt.yaml
        """,
    )
    with pytest.raises(AgentManifestError, match="verbatim"):
        load_agent_manifest(path, allowed_roots=(tmp_path,))


def test_system_file_drift_detected_at_bridge(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_phases_tree(tmp_path), allowed_roots=(tmp_path,))

    _write(tmp_path / "prompts/planner.md", "Plan differently.")

    with pytest.raises(AgentManifestError, match="changed since the manifest was loaded"):
        manifest.reasoning_spec()


def test_profile_can_add_phases(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          name: rewoo
        profiles:
          tuned:
            strategy:
              phases:
                planner:
                  system: profile planner
        """,
    )
    manifest = load_agent_manifest(path, profile="tuned", allowed_roots=(tmp_path,))

    assert manifest.reasoning_spec().knobs["planner_system"] == "profile planner"
