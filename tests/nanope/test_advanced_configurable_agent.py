"""Tests for the advanced configurable agent nano project."""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import AsyncMock

import pytest

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, ToolCall, Usage
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import (
    CapabilityProfile,
    ChatSession,
    ConfigurableAgent,
    ToolGovernance,
    ToolRegistry,
    agent_config_from_mapping,
    build_chat_agent,
    create_private_memory,
    load_agent_config,
    private_memory_tools,
    profile_details,
    render_system_prompt,
    resolve_agent_config,
    resolve_tools,
    run_terminal_chat,
    web_search_query,
)
from ai_arch_toolkit.toolkit.memory._types import Node


def _response(
    text: str = "",
    tool_calls: tuple[ToolCall, ...] = (),
    cost: float = 0.001,
) -> Response:
    return Response(
        text=text,
        tool_calls=tool_calls,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class _CapturingProvider:
    """Real-LLM provider stand-in: runs the metering charge site and captures call kwargs."""

    def __init__(self, *responses: Response) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.last_kwargs: dict = {}

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        self.calls += 1
        self.last_kwargs = {"system": system, "tools": tools, **kwargs}
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _metered_llm(*responses: Response) -> tuple[LLM, _CapturingProvider]:
    llm = LLM("claude-sonnet-4-6", api_key="test")
    provider = _CapturingProvider(*responses)
    llm._provider = provider  # type: ignore[assignment]
    return llm, provider


def _base_config() -> dict:
    return {
        "identity": {
            "name": "researcher",
            "description": "Researches topics and produces grounded notes.",
        },
        "model": {"name": "test-model"},
    }


def test_config_validation_and_normalization() -> None:
    config = agent_config_from_mapping(
        {
            "name": "researcher",
            "description": "Researches topics.",
            "model": "test-model",
            "goals": ["collect facts", "summarize"],
            "tasks": ["search"],
        }
    )

    assert config.identity.name == "researcher"
    assert config.model.name == "test-model"
    assert config.context.goals == ("collect facts", "summarize")
    assert config.context.tasks == ("search",)


def test_config_requires_identity_and_model() -> None:
    with pytest.raises(ValueError, match=r"identity\.name"):
        agent_config_from_mapping({"description": "missing name", "model": "test-model"})

    with pytest.raises(ValueError, match=r"model\.name"):
        agent_config_from_mapping({"name": "a", "description": "b"})


def test_unknown_reasoning_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="supported strategies"):
        agent_config_from_mapping(
            {
                **_base_config(),
                "reasoning": {"strategy": "bogus"},
            }
        )


def test_prompt_includes_mandatory_identity_and_present_context_only() -> None:
    config = agent_config_from_mapping(
        {
            **_base_config(),
            "context": {
                "role": "Research Agent",
                "goals": ["find reliable sources"],
                "style": "precise",
            },
        }
    )

    prompt = render_system_prompt(config)

    assert "Agent name: researcher" in prompt.system
    assert "Agent description: Researches topics" in prompt.system
    assert "Role: Research Agent" in prompt.system
    assert "- find reliable sources" in prompt.system
    assert "Style: precise" in prompt.system
    assert "Tasks:" not in prompt.system
    assert prompt.section_names == ("identity", "role", "goals", "style")


def test_extra_sections_default_to_after_built_ins() -> None:
    config = agent_config_from_mapping(
        {
            **_base_config(),
            "context": {
                "role": "Story analyst",
                "extra_sections": [
                    {"name": "menus", "content": "Genres: MYSTERY, ROMANCE, ..."},
                    {"name": "output_contract", "content": "Pick by stable KEY."},
                ],
            },
        }
    )

    prompt = render_system_prompt(config)

    # extras come after every built-in by default
    assert prompt.section_names == ("identity", "role", "menus", "output_contract")
    assert prompt.system.index("Role:") < prompt.system.index("Genres:")
    assert prompt.system.index("Genres:") < prompt.system.index("Pick by stable KEY")


def test_extra_section_position_can_interleave_with_built_ins() -> None:
    # position=350 lands between goals (300) and tasks (400)
    config = agent_config_from_mapping(
        {
            **_base_config(),
            "context": {
                "role": "Story analyst",
                "goals": ["produce a categorization"],
                "tasks": ["pick by stable key"],
                "extra_sections": [
                    {"name": "menus", "content": "MENUS BLOCK", "position": 350},
                ],
            },
        }
    )

    prompt = render_system_prompt(config)

    assert prompt.section_names == ("identity", "role", "goals", "menus", "tasks")


def test_extra_section_negative_position_goes_before_identity() -> None:
    config = agent_config_from_mapping(
        {
            **_base_config(),
            "context": {
                "extra_sections": [
                    {"name": "header", "content": "HEADER", "position": -1},
                ],
            },
        }
    )

    prompt = render_system_prompt(config)

    assert prompt.section_names == ("header", "identity")


def test_extra_sections_round_trip_through_to_dict() -> None:
    config = agent_config_from_mapping(
        {
            **_base_config(),
            "context": {
                "extra_sections": [
                    {"name": "menus", "content": "MENUS", "position": 350},
                ],
            },
        }
    )

    rehydrated = agent_config_from_mapping(config.to_dict())

    assert rehydrated.context.extra_sections == config.context.extra_sections


def test_section_providers_add_runtime_sections() -> None:
    from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import (
        PromptSection,
    )

    config = agent_config_from_mapping({**_base_config(), "context": {"role": "Analyst"}})

    def menus_provider(_config) -> PromptSection:
        return PromptSection(name="menus", content="MENUS", position=350)

    def skip_provider(_config) -> PromptSection | None:
        return None

    prompt = render_system_prompt(config, providers=[menus_provider, skip_provider])

    assert prompt.section_names == ("identity", "role", "menus")
    assert "MENUS" in prompt.system


def test_promptsection_requires_name() -> None:
    from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import (
        PromptSection,
    )

    with pytest.raises(ValueError, match=r"PromptSection\.name"):
        PromptSection(name="", content="x")


def test_extra_sections_append_across_profile_and_config_layers() -> None:
    profiles = {
        "story_pack": CapabilityProfile(
            "story_pack",
            {"context": {"extra_sections": [{"name": "menus", "content": "MENUS"}]}},
        ),
    }

    resolved = resolve_agent_config(
        {
            **_base_config(),
            "capability_profiles": ["story_pack"],
            "context": {
                "extra_sections": [{"name": "output_contract", "content": "PICK BY KEY"}],
            },
        },
        profiles=profiles,
    )

    names = tuple(section.name for section in resolved.config.context.extra_sections)
    assert names == ("menus", "output_contract")


def test_configurable_agent_passes_providers_to_render() -> None:
    from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import (
        PromptSection,
    )

    def runtime_provider(_config) -> PromptSection:
        return PromptSection(name="runtime", content="RUNTIME-INJECTED", position=999)

    agent = ConfigurableAgent(
        {**_base_config(), "context": {"role": "Analyst"}},
        prompt_section_providers=[runtime_provider],
    )

    prompt = agent.render_prompt()

    assert "runtime" in prompt.section_names
    assert "RUNTIME-INJECTED" in prompt.system


def test_profile_precedence_and_override_policy() -> None:
    profiles = {
        "web_researcher": CapabilityProfile(
            "web_researcher",
            {
                "tools": {"enabled": ["wikipedia"]},
                "limits": {"max_cost": 0.25},
            },
        )
    }
    config = {
        **_base_config(),
        "capability_profiles": ["web_researcher"],
        "tools": {"enabled": ["dictionary"]},
        "override_policy": {
            "allow": ["model.temperature", "limits.max_cost", "tools.enabled"],
            "deny": ["identity.name", "tools.permissions"],
        },
    }

    resolved = resolve_agent_config(
        config,
        profiles=profiles,
        run_overrides={
            "model.temperature": 0.8,
            "limits": {"max_cost": 0.75},
            "tools.enabled": ["web.search"],
            "identity.name": "other",
            "tools.permissions": {"shell": {"enabled": True}},
        },
    )

    assert resolved.config.identity.name == "researcher"
    assert resolved.config.model.temperature == 0.8
    assert resolved.config.limits.max_cost == 0.75
    assert resolved.config.tools.enabled == ("wikipedia", "dictionary", "web.search")
    assert "identity.name" in resolved.override_report.rejected
    assert "tools.permissions.shell.enabled" in resolved.override_report.rejected


def test_tool_resolution_enables_and_disables_tools() -> None:
    def wiki() -> str:
        """Search wiki."""
        return "wiki"

    def web() -> str:
        """Search web."""
        return "web"

    registry = ToolRegistry.from_mapping({"wikipedia": wiki, "web.search": web})
    config = agent_config_from_mapping(
        {
            **_base_config(),
            "tools": {"enabled": ["wikipedia", "web.search"], "disabled": ["web.search"]},
        }
    )

    resolved = resolve_tools(config.tools, registry)

    assert resolved.names == ("wikipedia",)
    assert len(resolved.group) == 1


def test_unknown_enabled_tool_raises() -> None:
    config = agent_config_from_mapping(
        {
            **_base_config(),
            "tools": {"enabled": ["missing"], "disabled": ["also_missing"]},
        }
    )

    with pytest.raises(KeyError, match="Unknown enabled tool"):
        resolve_tools(config.tools, ToolRegistry())


def test_dangerous_tools_are_blocked_by_default() -> None:
    def run_command(command: str) -> str:
        """Run a command."""
        return command

    config = agent_config_from_mapping(
        {
            **_base_config(),
            "tools": {"enabled": ["run_command"]},
        }
    )

    resolved = resolve_tools(config.tools, ToolRegistry.from_mapping({"run_command": run_command}))
    result = resolved.group.execute(
        ToolCall(id="tc1", name="run_command", input={"command": "echo hi"})
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "dangerous_tool_blocked"
    assert "--allow-dangerous-tools" in result.to_model_text()


def test_tool_governance_dry_run_does_not_execute() -> None:
    calls: list[str] = []

    def echo(text: str) -> str:
        """Echo text."""
        calls.append(text)
        return text

    config = agent_config_from_mapping(
        {
            **_base_config(),
            "tools": {"enabled": ["echo"], "permissions": {"dry_run": True}},
        }
    )

    resolved = resolve_tools(config.tools, ToolRegistry.from_mapping({"echo": echo}))
    first = resolved.group.execute(ToolCall(id="tc1", name="echo", input={"text": "hi"}))
    second = resolved.group.execute(ToolCall(id="tc2", name="echo", input={"text": "again"}))

    assert isinstance(ToolGovernance(), ToolGovernance)
    # Dry-run never executes and never consumes the (absent) budget.
    assert first.metadata["governance"]["outcome"] == "dry_run"
    assert second.metadata["governance"]["outcome"] == "dry_run"
    assert calls == []


def test_tool_governance_max_calls_blocks_after_limit() -> None:
    def echo(text: str) -> str:
        """Echo text."""
        return text

    config = agent_config_from_mapping(
        {
            **_base_config(),
            "tools": {"enabled": ["echo"], "permissions": {"max_calls": 1}},
        }
    )

    resolved = resolve_tools(config.tools, ToolRegistry.from_mapping({"echo": echo}))
    first = resolved.group.execute(ToolCall(id="tc1", name="echo", input={"text": "hi"}))
    second = resolved.group.execute(ToolCall(id="tc2", name="echo", input={"text": "again"}))

    assert first.ok is True
    assert first.value == "hi"
    assert second.ok is False
    assert second.error is not None
    assert second.error.type == "max_calls_exceeded"


async def test_configurable_agent_run_simple_response() -> None:
    # Real LLM + fake provider so cost/usage flow through the meter (the single source of truth).
    llm, provider = _metered_llm(_response(text="done"))
    agent = ConfigurableAgent(_base_config(), llm_factory=lambda _: llm)

    result = await agent.run("hello")

    assert result.status == "completed"
    assert result.agent_name == "researcher"
    assert result.final_text == "done"
    assert result.final_response is not None
    assert result.cost > 0  # metered from a priced model, not the response's manual cost field
    assert result.usage.input_tokens == 10
    assert result.enabled_tools == ()
    assert "Agent name: researcher" in provider.last_kwargs["system"]


async def test_configurable_agent_passes_output_schema_to_react_llm() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_response(text='{"answer": "done"}'))
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "output": {
                "name": "structured_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
        llm_factory=lambda _: llm,
    )

    await agent.run("hello")
    output_schema = llm.complete.call_args.kwargs["output_schema"]

    assert output_schema.name == "structured_answer"
    assert output_schema.schema["type"] == "object"
    json_payload = {
        "schema": output_schema.schema,
    }
    json.dumps(json_payload)


async def test_output_schema_rejects_unsupported_strategy() -> None:
    llm = AsyncMock()
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "reasoning": {"strategy": "plan_execute"},
            "output": {"schema": {"type": "object"}},
        },
        llm_factory=lambda _: llm,
    )

    with pytest.raises(ValueError, match=r"output\.schema"):
        await agent.run("hello")


async def test_output_schema_rejects_generate_review_strategy() -> None:
    llm = AsyncMock()
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "reasoning": {"strategy": "generate_review"},
            "output": {"schema": {"type": "object"}},
        },
        llm_factory=lambda _: llm,
    )

    with pytest.raises(ValueError, match="generate_review"):
        await agent.run("hello")


async def test_configurable_agent_react_tool_execution() -> None:
    def echo(text: str) -> str:
        """Echo text."""
        return f"echo:{text}"

    tool_call = ToolCall(id="tc1", name="echo", input={"text": "hi"})
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _response(tool_calls=(tool_call,)),
            _response(text="echo:hi"),
        ]
    )
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "tools": {"enabled": ["echo"]},
            "reasoning": {"max_iterations": 5},
        },
        tool_registry={"echo": echo},
        llm_factory=lambda _: llm,
    )

    result = await agent.run("use echo")

    assert result.status == "completed"
    assert result.final_text == "echo:hi"
    assert result.enabled_tools == ("echo",)
    assert llm.complete.call_count == 2


async def test_limits_max_tool_calls_blocks_extra_tool_calls() -> None:
    def echo(text: str) -> str:
        """Echo text."""
        return f"echo:{text}"

    tool_calls = (
        ToolCall(id="tc1", name="echo", input={"text": "one"}),
        ToolCall(id="tc2", name="echo", input={"text": "two"}),
    )
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _response(tool_calls=tool_calls),
            _response(text="done"),
        ]
    )
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "tools": {"enabled": ["echo"]},
            "limits": {"max_tool_calls": 1},
            "reasoning": {"parallel_tool_calls": False, "max_iterations": 4},
        },
        tool_registry={"echo": echo},
        llm_factory=lambda _: llm,
    )

    await agent.run("call echo twice")
    second_messages = llm.complete.call_args_list[1].args[0]

    assert "echo:one" in second_messages[-2]["content"]
    assert "max tool calls exceeded" in second_messages[-1]["content"]


def test_configurable_agent_run_sync() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_response(text="sync done"))
    agent = ConfigurableAgent(_base_config(), llm_factory=lambda _: llm)

    result = agent.run_sync("hello")

    assert result.status == "completed"
    assert result.final_text == "sync done"


def test_config_accepts_all_executable_strategies() -> None:
    strategies = (
        "react",
        "plan_execute",
        "reflexion",
        "self_discovery",
        "generate_review",
        "rewoo",
        "llm_compiler",
        "tot",
        "lats",
    )

    for strategy in strategies:
        config = agent_config_from_mapping(
            {
                **_base_config(),
                "reasoning": {"strategy": strategy},
            }
        )
        assert config.reasoning.strategy == strategy


async def test_private_memory_tools_are_enabled_and_store_memory() -> None:
    tool_call = ToolCall(
        id="tc1",
        name="remember",
        input={"text": "The user likes concise answers.", "node_type": "preference"},
    )
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _response(tool_calls=(tool_call,)),
            _response(text="I remembered that."),
        ]
    )
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "memory": {"private_enabled": True, "read": True, "write": True},
            "reasoning": {"max_iterations": 4},
        },
        llm_factory=lambda _: llm,
    )

    result = await agent.run("Remember that I like concise answers.")

    assert result.status == "completed"
    assert "remember" in result.enabled_tools
    assert "recall" in result.enabled_tools
    assert result.memory_report["implemented"] is True
    assert result.memory_report["node_count"] == 1


def test_private_memory_tools_respect_read_write_policy() -> None:
    write_only_agent = ConfigurableAgent(
        {
            **_base_config(),
            "memory": {"private_enabled": True, "read": False, "write": True},
        }
    )
    read_only_agent = ConfigurableAgent(
        {
            **_base_config(),
            "memory": {"private_enabled": True, "read": True, "write": False},
        }
    )

    write_only_tools = set(write_only_agent.resolve_tool_names())
    read_only_tools = set(read_only_agent.resolve_tool_names())

    assert {"remember", "forget_memory", "consolidate_memories"} <= write_only_tools
    assert "recall" not in write_only_tools
    assert "list_memories" not in write_only_tools
    assert {
        "recall",
        "explore_memory",
        "list_memories",
        "find_duplicate_memories",
    } <= read_only_tools
    assert "remember" not in read_only_tools
    assert "forget_memory" not in read_only_tools


def test_capability_inspection_does_not_persist_memory_store() -> None:
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "memory": {"private_enabled": True, "read": True, "write": True},
        }
    )

    capabilities = agent.describe_capabilities()

    assert "remember" in capabilities["tools_enabled"]
    assert agent._memory is None


async def test_private_memory_remember_deduplicates_equivalent_memory() -> None:
    first_call = ToolCall(
        id="tc1",
        name="remember",
        input={
            "text": "User's name is Rafael.",
            "node_type": "fact",
            "source": "user_stated",
            "subject": "Rafael",
        },
    )
    second_call = ToolCall(
        id="tc2",
        name="remember",
        input={
            "text": "User's name is Rafael",
            "node_type": "fact",
            "source": "user_stated",
            "subject": "Rafael",
        },
    )
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            _response(tool_calls=(first_call,)),
            _response(text="remembered once"),
            _response(tool_calls=(second_call,)),
            _response(text="already remembered"),
        ]
    )
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "memory": {"private_enabled": True, "read": True, "write": True},
            "reasoning": {"max_iterations": 4},
        },
        llm_factory=lambda _: llm,
    )

    first = await agent.run("My name is Rafael.")
    second = await agent.run("Remember that my name is Rafael.")

    assert first.memory_report["node_count"] == 1
    assert second.memory_report["node_count"] == 1


async def test_private_memory_inspection_tools_find_and_consolidate_duplicates() -> None:
    store = create_private_memory()
    first = await store.add(
        Node(
            type="fact",
            content={"text": "User's name is Rafael.", "subject": "Rafael"},
            source="user_stated",
        )
    )
    second = await store.add(
        Node(
            type="fact",
            content={"text": "User's name is Rafael", "subject": "Rafael"},
            source="user_stated",
        )
    )
    tools = private_memory_tools(store)

    listed = (
        await tools.async_execute(
            ToolCall(id="tc1", name="list_memories", input={"node_type": "fact"})
        )
    ).to_model_text()
    duplicates = (
        await tools.async_execute(ToolCall(id="tc2", name="find_duplicate_memories", input={}))
    ).to_model_text()
    consolidated = (
        await tools.async_execute(ToolCall(id="tc3", name="consolidate_memories", input={}))
    ).to_model_text()

    assert first.id in listed
    assert second.id in listed
    assert "Found 1 duplicate memory group" in duplicates
    assert first.id in duplicates
    assert second.id in duplicates
    assert "Removed 1 duplicate memory node" in consolidated
    assert await store.count() == 1


async def test_private_memory_is_injected_into_task_context() -> None:
    store = create_private_memory()
    await store.add(
        Node(
            type="fact",
            content={"text": "User's name is Rafael.", "subject": "Rafael"},
            source="user_stated",
        )
    )
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_response(text="Your name is Rafael."))
    agent = ConfigurableAgent(
        {
            **_base_config(),
            "memory": {
                "private_enabled": True,
                "read": True,
                "write": True,
                "inject": True,
                "inject_k": 3,
            },
        },
        memory=store,
        llm_factory=lambda _: llm,
    )

    result = await agent.run("What is my name?")
    messages = llm.complete.call_args.args[0]

    assert result.final_text == "Your name is Rafael."
    assert "Relevant private memories:" in messages[0]["content"]
    assert "User's name is Rafael" in messages[0]["content"]


def test_load_agent_config_from_toml(tmp_path) -> None:
    config_path = tmp_path / "agent.toml"
    config_path.write_text(
        """
        capability_profiles = ["math_helper"]

        [identity]
        name = "toml_agent"
        description = "Loaded from TOML."

        [model]
        name = "test-model"

        [output]
        name = "answer"
        strict = true

        [output.schema]
        type = "object"
        required = ["answer"]
        """,
    )

    config = load_agent_config(config_path)

    assert config.identity.name == "toml_agent"
    assert config.capability_profiles == ("math_helper",)
    assert config.output.schema["type"] == "object"


def test_profile_details_describe_profiles() -> None:
    details = profile_details("all_tools")

    assert "dangerous" in details["all_tools"]["description"].lower()
    assert "tools" in details["all_tools"]["config"]


def test_capability_profile_override_applies_profile_fragment() -> None:
    config = {
        **_base_config(),
        "override_policy": {"allow": ["capability_profiles"]},
    }

    resolved = resolve_agent_config(
        config,
        profiles={
            "memory_profile": CapabilityProfile(
                "memory_profile",
                {
                    "memory": {"private_enabled": True, "read": True, "write": False},
                    "tools": {"enabled": ["recall"]},
                },
            )
        },
        run_overrides={"capability_profiles": ["memory_profile"]},
    )

    assert resolved.config.capability_profiles == ("memory_profile",)
    assert resolved.config.memory.private_enabled is True
    assert resolved.config.memory.read is True
    assert resolved.config.tools.enabled == ("recall",)


def test_build_chat_agent_defaults() -> None:
    agent = build_chat_agent(model="fake-model", tools=("datetime_now",), max_iterations=3)

    capabilities = agent.describe_capabilities()

    assert capabilities["agent"] == "manual_chat_agent"
    assert capabilities["tools_enabled"] == ["datetime_now"]
    assert capabilities["reasoning_strategy"] == "react"


def test_describe_capabilities_includes_runtime_memory_tools() -> None:
    agent = build_chat_agent(
        model="fake-model",
        tools=("datetime_now",),
        profiles=("private_memory_user",),
        memory_enabled=True,
    )

    capabilities = agent.describe_capabilities()

    assert "datetime_now" in capabilities["tools_enabled"]
    assert "remember" in capabilities["tools_enabled"]
    assert "list_memories" in capabilities["tools_enabled"]


def test_terminal_chat_applies_config_cli_flags_and_handles_unknown_profile(tmp_path) -> None:
    config_path = tmp_path / "agent.toml"
    config_path.write_text(
        """
        [identity]
        name = "config_agent"
        description = "Loaded from config."

        [model]
        name = "old-model"

        [reasoning]
        strategy = "react"
        max_iterations = 2

        [tools]
        enabled = ["datetime_now"]
        """,
    )
    inputs = iter(("/profile missing", "/exit"))
    printed: list[str] = []

    run_terminal_chat(
        config_path=config_path,
        model="new-model",
        temperature=0.7,
        max_iterations=4,
        tools=("math_eval",),
        profiles=("private_memory_user",),
        strategy="generate_review",
        load_env=False,
        input_fn=lambda _prompt: next(inputs),
        print_fn=lambda *values: printed.append(" ".join(str(value) for value in values)),
    )
    output = "\n".join(printed)

    assert "model=new-model strategy=generate_review profiles=private_memory_user" in output
    assert "math_eval" in output
    assert "remember" in output
    assert "unknown profile: missing" in output


def test_chat_session_builds_history_task() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_response(text="first"))
    agent = ConfigurableAgent(_base_config(), llm_factory=lambda _: llm)
    session = ChatSession(agent=agent)

    first = session.ask("hello")
    task = session._build_task("what did I say?")

    assert first.final_text == "first"
    assert "Conversation so far:" in task
    assert "User: hello" in task
    assert "Assistant: first" in task
    assert "New user message: what did I say?" in task


def test_chat_session_persists_history(tmp_path) -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_response(text="first"))
    agent = ConfigurableAgent(_base_config(), llm_factory=lambda _: llm)
    path = tmp_path / "session.json"
    session = ChatSession(agent=agent, max_history_chars=80)

    session.ask("hello")
    session.save(path)
    loaded = ChatSession(agent=agent, max_history_chars=80)
    loaded.load(path)
    task = loaded._build_task("what now?")

    assert loaded.history == [("hello", "first")]
    assert len(task) <= 80


def test_web_search_query_parses_duckduckgo_results(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com">'
                b"Example result</a>"
            )

    def fake_urlopen(_request: urllib.request.Request, timeout: int) -> FakeResponse:
        assert timeout == 10
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = web_search_query("example")

    assert "Example result" in result
    assert "https://example.com" in result
