"""Tests for agent_from_manifest — phase model configs resolved into deps."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents import agent_from_manifest, load_agent_manifest


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _make_response(text: str = "") -> Response:
    return Response(text=text, usage=Usage(input_tokens=10, output_tokens=5), cost=0.001)


class _RecordingProvider:
    def __init__(self, *texts: str) -> None:
        self._responses = [_make_response(text) for text in (texts or ("",))]
        self.calls = 0

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _llm(*texts: str) -> LLM:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    llm._provider = _RecordingProvider(*texts)  # type: ignore[assignment]
    return llm


def _calls(llm: LLM) -> int:
    return llm._provider.calls  # type: ignore[union-attr]


def _manifest_path(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "phased.agent.yaml",
        """
        version: 1
        id: phased.agent
        strategy:
          name: plan_execute
          phases:
            planner:
              system: PLAN IT
              model:
                model: claude-haiku-4-5
            solver:
              system: SOLVE IT
        """,
    )


async def test_llm_factory_resolves_phase_models(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_manifest_path(tmp_path), allowed_roots=(tmp_path,))
    default = _llm("default answer")
    planner = _llm("1. Do the thing")
    seen: list[tuple[str, dict[str, Any]]] = []

    def factory(phase: str, model_config: Any) -> LLM:
        seen.append((phase, dict(model_config)))
        return planner

    agent = agent_from_manifest(manifest, default, ToolGroup(), llm_factory=factory)
    result = await agent.run("the task")

    assert seen == [("planner", {"model": "claude-haiku-4-5"})]
    assert _calls(planner) == 1
    # Executor and solver have no declared model, so they use the default LLM.
    assert _calls(default) == 2
    assert result.text == "default answer"


def test_missing_factory_is_an_error(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_manifest_path(tmp_path), allowed_roots=(tmp_path,))
    with pytest.raises(ValueError, match="llm_factory"):
        agent_from_manifest(manifest, _llm())


async def test_generator_phase_model_resolves(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "gr.agent.yaml",
        """
        version: 1
        strategy:
          name: generate_review
          phases:
            generator:
              model:
                model: claude-haiku-4-5
        """,
    )
    manifest = load_agent_manifest(path, allowed_roots=(tmp_path,))
    generator = _llm("draft answer")
    seen: list[str] = []

    def factory(phase: str, model_config: Any) -> LLM:
        seen.append(phase)
        return generator

    agent = agent_from_manifest(manifest, _llm("ACCEPT"), llm_factory=factory)
    result = await agent.run("task")

    assert seen == ["generator"]
    assert result.text == "draft answer"
    assert _calls(generator) == 1


def test_invalid_spec_fails_before_factory(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.agent.yaml",
        """
        version: 1
        strategy:
          name: plan_execute
          knobs:
            max_replans: -1
          phases:
            planner:
              model:
                model: claude-haiku-4-5
        """,
    )
    manifest = load_agent_manifest(path, allowed_roots=(tmp_path,))
    seen: list[str] = []

    def factory(phase: str, model_config: Any) -> LLM:
        seen.append(phase)
        return _llm()

    with pytest.raises(ValueError, match="max_replans"):
        agent_from_manifest(manifest, _llm(), llm_factory=factory)
    assert seen == []


def test_explicit_dep_wins_over_factory(tmp_path: Path) -> None:
    manifest = load_agent_manifest(_manifest_path(tmp_path), allowed_roots=(tmp_path,))

    def factory(phase: str, model_config: Any) -> LLM:  # pragma: no cover - must not run
        raise AssertionError("factory should not be called for explicitly bound phases")

    agent = agent_from_manifest(
        manifest, _llm(), llm_factory=factory, deps={"planner_llm": _llm("1. Do")}
    )
    assert agent is not None
