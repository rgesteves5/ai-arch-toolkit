"""Hermetic system tests for the public configurable-agent path."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai_arch_toolkit.core import LLM, Response, ToolCall, ToolGroup, Usage, tool
from ai_arch_toolkit.toolkit.agents import Agent, ResolvedAgentManifest, load_agent_manifest
from ai_arch_toolkit.toolkit.prompts import load_prompt

pytestmark = pytest.mark.integration


class _ScriptedProvider:
    """Deterministic provider boundary while the real toolkit stack stays active."""

    def __init__(self, *responses: Response) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response:
        self.requests.append(
            {
                "messages": deepcopy(messages),
                "system": system,
                "tools": deepcopy(tools),
                "kwargs": deepcopy(kwargs),
            }
        )
        index = len(self.requests) - 1
        if index >= len(self._responses):
            raise AssertionError("scripted provider received an unexpected call")
        return self._responses[index]

    async def close(self) -> None:
        self.closed = True


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _configured_project(root: Path) -> tuple[Path, Path]:
    prompt = _write_json(
        root / "prompts/calculator.prompt.json",
        {
            "version": 1,
            "name": "calculator-system",
            "variables": {"persona": {"type": "string", "required": True}},
            "sections": [
                {
                    "name": "role",
                    "template": {"content": "You are a ${persona}."},
                    "order": 100,
                },
                {
                    "name": "tools",
                    "content": "Use the add tool for arithmetic.",
                    "order": 200,
                },
            ],
        },
    )
    _write_json(
        root / "profiles/default.agent.json",
        {
            "version": 1,
            "strategy": {
                "name": "react",
                "max_iterations": 3,
                "parallel_tool_calls": False,
            },
            "model": {
                "model": "gpt-4.1-nano",
                "temperature": 0.4,
                "max_tokens": 64,
            },
            "limits": {
                "max_llm_calls": 3,
                "max_tool_calls": 1,
                "max_total_tokens": 100,
            },
            "override_policy": {
                "allow": ["model.temperature", "limits.max_llm_calls"],
                "deny": ["strategy"],
            },
            "profiles": {
                "production": {
                    "model": {"temperature": 0.2},
                    "prompts": {"system_manifest": "../prompts/calculator.prompt.json"},
                }
            },
        },
    )
    entry = _write_json(
        root / "agents/calculator.agent.json",
        {
            "version": 1,
            "id": "calculator",
            "extends": "../profiles/default.agent.json",
        },
    )
    return entry, prompt


def _load_configured_manifest(
    root: Path,
    *,
    max_llm_calls: int = 3,
) -> ResolvedAgentManifest:
    entry, _prompt = _configured_project(root)
    return load_agent_manifest(
        entry,
        profile="production",
        overrides={
            "model.temperature": 0.1,
            "limits.max_llm_calls": max_llm_calls,
        },
        allowed_roots=(root,),
    )


def _llm_from_manifest(
    manifest: ResolvedAgentManifest,
    provider: _ScriptedProvider,
) -> LLM:
    model = manifest.as_dict()["model"]
    with patch("ai_arch_toolkit.core._llm.create_provider", return_value=provider):
        return LLM(
            model["model"],
            temperature=model["temperature"],
            max_tokens=model["max_tokens"],
            api_key="test",
            retry=False,
        )


def _system_from_manifest(manifest: ResolvedAgentManifest) -> str:
    prompt_path = Path(manifest.as_dict()["prompts"]["system_manifest"])
    return load_prompt(prompt_path).render(persona="precise calculator").text


async def test_manifest_prompt_agent_tool_and_metering_work_as_one_system(
    tmp_path: Path,
) -> None:
    manifest = _load_configured_manifest(tmp_path)
    system = _system_from_manifest(manifest)
    provider = _ScriptedProvider(
        Response(
            tool_calls=(ToolCall(id="add-1", name="add", input={"a": 3, "b": 4}),),
            usage=Usage(input_tokens=11, output_tokens=3),
        ),
        Response(text="The answer is 7.", usage=Usage(input_tokens=9, output_tokens=2)),
    )
    executions: list[tuple[int, int]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        executions.append((a, b))
        return a + b

    llm = _llm_from_manifest(manifest, provider)
    async with llm:
        result = await Agent(
            manifest.reasoning_spec(system=system),
            llm,
            ToolGroup(add),
        ).run(
            "What is 3 + 4?",
            budget_policy=manifest.budget_policy(),
        )

    assert result.text == "The answer is 7."
    assert result.errors == ()
    assert executions == [(3, 4)]
    assert len(provider.requests) == 2
    assert provider.closed

    first, second = provider.requests
    assert first["system"] == system
    assert "precise calculator" in first["system"]
    assert first["kwargs"]["temperature"] == 0.1
    assert first["kwargs"]["max_tokens"] == 64
    assert [definition["name"] for definition in first["tools"]] == ["add"]
    assert any(
        message.get("role") == "tool" and message.get("content") == "7"
        for message in second["messages"]
    )

    assert result.report is not None
    assert result.report.llm_calls == 2
    assert result.report.tool_calls == 1
    assert result.usage == Usage(input_tokens=20, output_tokens=5)
    assert result.cost == result.report.cost > 0


async def test_manifest_budget_stops_the_configured_agent_at_the_charge_site(
    tmp_path: Path,
) -> None:
    manifest = _load_configured_manifest(tmp_path, max_llm_calls=1)
    provider = _ScriptedProvider(
        Response(
            tool_calls=(ToolCall(id="add-1", name="add", input={"a": 5, "b": 6}),),
            usage=Usage(input_tokens=8, output_tokens=2),
        )
    )
    executions: list[tuple[int, int]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        executions.append((a, b))
        return a + b

    llm = _llm_from_manifest(manifest, provider)
    async with llm:
        result = await Agent(
            manifest.reasoning_spec(system=_system_from_manifest(manifest)),
            llm,
            ToolGroup(add),
        ).run(
            "What is 5 + 6?",
            budget_policy=manifest.budget_policy(),
        )

    assert len(provider.requests) == 1
    assert executions == [(5, 6)]
    assert "budget_exceeded" in result.flow_result.results
    assert result.report is not None
    assert result.report.llm_calls == 1
    assert result.report.tool_calls == 1
    assert result.report.breached == ("llm_calls", "tool_calls")
    assert provider.closed


def test_manifest_fingerprint_tracks_the_configured_prompt_content(tmp_path: Path) -> None:
    entry, prompt = _configured_project(tmp_path)
    original = load_agent_manifest(
        entry,
        profile="production",
        allowed_roots=(tmp_path,),
    )

    prompt.write_text(prompt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = load_agent_manifest(
        entry,
        profile="production",
        allowed_roots=(tmp_path,),
    )

    assert changed.fingerprint != original.fingerprint
    assert changed.referenced_fingerprints != original.referenced_fingerprints
