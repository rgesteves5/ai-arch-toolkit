"""Nanope integration with toolkit prompt manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import (
    ConfigurableAgent,
    agent_config_from_mapping,
    load_agent_config,
    render_system_prompt,
)
from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry


def write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "agent.prompt.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "layout": "markdown",
                "variables": {"domain": {"required": True}},
                "sections": [
                    {
                        "name": "domain",
                        "template": {"content": "Domain: ${domain}"},
                        "order": 250,
                    }
                ],
            }
        )
    )
    return path


def base_config(tmp_path: Path, *, mode: str = "append"):
    return agent_config_from_mapping(
        {
            "identity": {"name": "writer", "description": "Writes stories"},
            "model": {"name": "gpt-5-mini"},
            "context": {"role": "Story Writer"},
            "prompt": {
                "manifest": str(write_manifest(tmp_path)),
                "variables": {"domain": "fiction"},
                "mode": mode,
            },
        }
    )


def test_nanope_appends_manifest_sections_and_uses_manifest_layout(tmp_path: Path) -> None:
    rendered = render_system_prompt(base_config(tmp_path))

    assert rendered.layout == "markdown"
    assert rendered.section_names == ("identity", "role", "domain")
    assert "## identity" in rendered.text
    assert "Domain: fiction" in rendered.text


def test_nanope_can_replace_builtin_prompt_with_manifest(tmp_path: Path) -> None:
    rendered = render_system_prompt(base_config(tmp_path, mode="replace"))

    assert rendered.section_names == ("domain",)
    assert "Agent name" not in rendered.text


def test_agent_config_file_resolves_relative_prompt_manifest(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    config_path = tmp_path / "agent.toml"
    config_path.write_text(
        f"""
[identity]
name = "writer"
description = "Writes stories"

[model]
name = "gpt-5-mini"

[prompt]
manifest = "{manifest.name}"

[prompt.variables]
domain = "fiction"
""".lstrip()
    )

    config = load_agent_config(config_path)

    assert config.prompt.manifest == str(manifest.resolve())
    assert render_system_prompt(config).section_names == ("identity", "domain")


def test_agent_prompt_config_round_trips_and_affects_fingerprint(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    data = config.to_dict()
    rehydrated = agent_config_from_mapping(data)

    assert rehydrated.prompt == config.prompt
    assert rehydrated.fingerprint == config.fingerprint


def test_nanope_injects_knowledge_into_manifest_sections(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.prompt.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sections": [{"name": "knowledge", "knowledge": "story.rules"}],
            }
        )
    )
    config = agent_config_from_mapping(
        {
            "identity": {"name": "writer", "description": "Writes stories"},
            "model": {"name": "gpt-5-mini"},
            "prompt": {"manifest": str(path), "mode": "replace"},
        }
    )
    knowledge = KnowledgeRegistry()
    knowledge.register("story.rules", "Keep continuity.")
    agent = ConfigurableAgent(config, knowledge=knowledge)
    assert agent.render_prompt().text == "Keep continuity."


def test_agent_prompt_config_validates_direct_construction() -> None:
    from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import AgentPromptConfig

    with pytest.raises(TypeError, match="manifest"):
        AgentPromptConfig(manifest=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="variables"):
        AgentPromptConfig(variables=[])  # type: ignore[arg-type]
