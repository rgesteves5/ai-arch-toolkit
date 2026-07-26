"""Tests for the ``ai-arch agent`` CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from ai_arch_toolkit._cli import main


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _valid_manifest(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        id: cli.agent
        strategy:
          name: plan_execute
          phases:
            planner:
              system: PLAN
        """,
    )


def test_agent_validate_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["agent", "validate", str(_valid_manifest(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "valid agent manifest" in out
    assert "plan_execute" in out
    assert "sha256:" in out


def test_agent_validate_unknown_phase(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          name: plan_execute
          phases:
            plannr:
              system: typo
        """,
    )
    assert main(["agent", "validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "has no phases: plannr" in err


def test_agent_validate_unknown_strategy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          name: nope
        """,
    )
    assert main(["agent", "validate", str(path)]) == 1
    assert "unknown strategy" in capsys.readouterr().err


def test_agent_validate_invalid_knob(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          name: plan_execute
          knobs:
            max_replans: -1
        """,
    )
    assert main(["agent", "validate", str(path)]) == 1
    assert "max_replans" in capsys.readouterr().err


def test_agent_inspect_outputs_config_and_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["agent", "inspect", str(_valid_manifest(tmp_path))]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "cli.agent"
    assert payload["fingerprint"].startswith("sha256:")
    assert payload["config"]["strategy"]["phases"]["planner"]["system"] == "PLAN"


def test_agent_validate_load_error_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["agent", "validate", str(tmp_path / "missing.agent.yaml")]) == 2
    assert "error:" in capsys.readouterr().err


def test_agent_validate_allowed_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path / "prompts/p.md", "PLAN {tools}")
    path = _write(
        tmp_path / "agents/a.agent.yaml",
        """
        version: 1
        strategy:
          name: rewoo
          phases:
            planner:
              system_file: ../prompts/p.md
        """,
    )
    assert main(["agent", "validate", str(path)]) == 2  # outside the default root
    capsys.readouterr()
    assert main(["agent", "validate", str(path), "--allowed-root", str(tmp_path)]) == 0


def test_agent_validate_unbindable_phase_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from ai_arch_toolkit.toolkit.agents import FlowStrategy, register_strategy

    register_strategy(
        "cli_unbindable_test",
        FlowStrategy(
            lambda ctx: (_ for _ in ()).throw(AssertionError("never built")),
            lambda task: {},
            phases=frozenset({"planner"}),
            allowed_deps=frozenset(),
        ),
    )
    path = _write(
        tmp_path / "a.agent.yaml",
        """
        version: 1
        strategy:
          name: cli_unbindable_test
          phases:
            planner:
              model:
                model: claude-haiku-4-5
        """,
    )
    assert main(["agent", "validate", str(path)]) == 1
    assert "does not accept an LLM binding" in capsys.readouterr().err
