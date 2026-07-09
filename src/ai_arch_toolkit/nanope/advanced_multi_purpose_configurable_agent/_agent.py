"""Configurable agent runtime backed by existing toolkit flows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ai_arch_toolkit.core._content import Content
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import OutputSchema, Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._sync import _run_sync
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._config import (
    AgentConfig,
    CapabilityProfile,
    ModelConfig,
    agent_config_from_mapping,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._memory import (
    create_private_memory,
    memory_count,
    private_memory_tools,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._profiles import (
    MEMORY_TOOLS,
    built_in_profiles,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._prompt import (
    RenderedPrompt,
    SectionProvider,
    render_system_prompt,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._resolver import (
    OverrideReport,
    ResolvedAgentConfig,
    resolve_agent_config,
)
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent._tools import (
    ResolvedTools,
    ToolRegistry,
    built_in_tool_registry,
    resolve_tools_with_limits,
)
from ai_arch_toolkit.toolkit.agents._compile import build_flow, extract_text, initial_state
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.flow._flow import FlowResult

type AgentStatus = Literal["completed", "failed", "cancelled", "stopped"]
type LLMFactory = Callable[[ModelConfig], Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRunResult:
    """Structured result returned by ConfigurableAgent."""

    agent_name: str
    task: Content
    final_text: str
    final_response: Response | None
    status: AgentStatus
    resolved_config_fingerprint: str
    prompt_fingerprint: str
    flow_result: FlowResult | None
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    events: tuple[Any, ...] = ()
    override_report: OverrideReport = field(default_factory=OverrideReport)
    enabled_tools: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    memory_report: Mapping[str, Any] = field(default_factory=dict)


class ConfigurableAgent:
    """Advanced configurable single-agent runtime.

    The agent resolves configuration into a prompt, tool group, and ReAct flow.
    """

    def __init__(
        self,
        config: AgentConfig | Mapping[str, Any],
        *,
        profiles: Mapping[str, CapabilityProfile | Mapping[str, Any]] | None = None,
        tool_registry: ToolRegistry | Mapping[str, Callable[..., Any]] | None = None,
        llm_factory: LLMFactory | None = None,
        memory: Any = None,
        prompt_section_providers: Sequence[SectionProvider] = (),
    ) -> None:
        self._config = (
            config if isinstance(config, AgentConfig) else agent_config_from_mapping(config)
        )
        self._profiles = {**built_in_profiles(), **dict(profiles or {})}
        if isinstance(tool_registry, ToolRegistry) or tool_registry is None:
            self._tool_registry = tool_registry or built_in_tool_registry()
        else:
            self._tool_registry = ToolRegistry.from_mapping(tool_registry)
        self._llm_factory = llm_factory or _default_llm_factory
        self._memory = memory
        self._prompt_section_providers = tuple(prompt_section_providers)

    def resolve_config(
        self,
        *,
        environment_config: Mapping[str, Any] | None = None,
        session_overrides: Mapping[str, Any] | None = None,
        run_overrides: Mapping[str, Any] | None = None,
        step_overrides: Mapping[str, Any] | None = None,
    ) -> ResolvedAgentConfig:
        """Resolve base config and optional override layers."""
        return resolve_agent_config(
            self._config,
            profiles=self._profiles,
            environment_config=environment_config,
            session_overrides=session_overrides,
            run_overrides=run_overrides,
            step_overrides=step_overrides,
        )

    def render_prompt(self, config: AgentConfig | None = None) -> RenderedPrompt:
        """Render a prompt from a config or the resolved base config."""
        resolved = config or self.resolve_config().config
        return render_system_prompt(resolved, providers=self._prompt_section_providers)

    def describe_capabilities(
        self,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a compact description of configured capabilities."""
        resolved = self.resolve_config(run_overrides=overrides).config
        return {
            "agent": resolved.identity.name,
            "profiles": list(resolved.capability_profiles),
            "tools_enabled": list(self.resolve_tool_names(overrides=overrides)),
            "tools_disabled": list(resolved.tools.disabled),
            "memory": resolved.memory.private_enabled,
            "reasoning_strategy": resolved.reasoning.strategy,
        }

    def resolve_tool_names(
        self,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...]:
        """Return the actual tool names that would be available for a run."""
        config = self.resolve_config(run_overrides=overrides).config
        tools, _memory_store = self._resolve_tools_for_config(config, create_memory=False)
        return tools.names

    async def run(
        self,
        task: Content,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> AgentRunResult:
        """Run the configured agent on one task."""
        resolved = self.resolve_config(run_overrides=overrides)
        config = resolved.config
        prompt = render_system_prompt(config, providers=self._prompt_section_providers)
        tools, memory_store = self._resolve_tools_for_config(config, create_memory=True)
        llm = self._llm_factory(config.model)

        _validate_output_schema(config)
        spec = _reasoning_spec(config, system=prompt.system)
        flow = build_flow(spec, llm, tools.group)
        run_task = await _inject_memory_context(task, config=config, memory_store=memory_store)

        state = State(
            operational=initial_state(spec, run_task),
            persistent={"memory": memory_store} if memory_store is not None else None,
        )
        flow_result = await flow.run(state)
        response = state.get("response") or state.get("last_response")
        if not isinstance(response, Response):
            response = None
        final_text = extract_text(state, flow_result)

        errors = _flow_errors(flow_result)
        status: AgentStatus = "failed" if errors else "completed"

        return AgentRunResult(
            agent_name=config.identity.name,
            task=task,
            final_text=final_text,
            final_response=response,
            status=status,
            resolved_config_fingerprint=config.fingerprint,
            prompt_fingerprint=prompt.fingerprint,
            flow_result=flow_result,
            usage=flow_result.usage,  # meter-derived (single source of truth)
            cost=flow_result.total_cost,
            override_report=resolved.override_report,
            enabled_tools=tools.names,
            errors=errors,
            memory_report={
                "enabled": config.memory.private_enabled,
                "implemented": True,
                "node_count": await memory_count(memory_store),
            },
        )

    def run_sync(
        self,
        task: Content,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> AgentRunResult:
        """Synchronous wrapper for run()."""
        return _run_sync(self.run(task, overrides=overrides))

    def _resolve_tools_for_config(
        self,
        config: AgentConfig,
        *,
        create_memory: bool,
    ) -> tuple[ResolvedTools, Any]:
        memory_store = self._memory
        if config.memory.private_enabled and memory_store is None:
            memory_store = create_private_memory()
            if create_memory:
                self._memory = memory_store
        tool_registry = self._tool_registry.copy()
        tool_config = config.tools
        if config.memory.private_enabled:
            if memory_store is not None:
                tool_registry.extend(
                    private_memory_tools(
                        memory_store,
                        read=config.memory.read,
                        write=config.memory.write,
                    )
                )
            enabled = [
                name
                for name in tool_config.enabled
                if name not in MEMORY_TOOLS or _memory_tool_allowed(name, config)
            ]
            for name in MEMORY_TOOLS:
                if _memory_tool_allowed(name, config) and name not in enabled:
                    enabled.append(name)
            tool_config = replace(tool_config, enabled=tuple(enabled))
        return (
            resolve_tools_with_limits(
                tool_config,
                tool_registry,
                max_tool_calls=config.limits.max_tool_calls,
            ),
            memory_store,
        )


def _default_llm_factory(config: ModelConfig) -> LLM:
    fallback: str | list[str] | None
    if len(config.fallback) == 0:
        fallback = None
    elif len(config.fallback) == 1:
        fallback = config.fallback[0]
    else:
        fallback = list(config.fallback)
    return LLM(
        config.name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        fallback=fallback,
        api_key=config.api_key,
        base_url=config.base_url,
    )


def _flow_policy(config: AgentConfig) -> Policy | None:
    if config.limits.max_cost is None:
        return None
    return Policy(
        timeout=config.reasoning.timeout or config.limits.max_runtime_seconds,
        max_cost=config.limits.max_cost,
    )


def _validate_output_schema(config: AgentConfig) -> None:
    if not config.output.schema:
        return
    strategy = config.reasoning.strategy
    if strategy not in {"react", "generate_review"}:
        raise ValueError("output.schema is currently supported only for react strategy")
    if strategy == "generate_review":
        raise ValueError(
            "output.schema is currently not supported for generate_review because the reviewer "
            "uses ACCEPT/RETRY control text"
        )


def _reasoning_spec(config: AgentConfig, *, system: str) -> ReasoningSpec:
    reasoning = config.reasoning
    knobs: dict[str, Any] = {
        **dict(reasoning.strategy_kwargs),
        "parallel_tool_calls": reasoning.parallel_tool_calls,
        "final_answer_hint": reasoning.final_answer_hint,
        "strip_tools_on_final": reasoning.strip_tools_on_final,
        "show_turn_counter": reasoning.show_turn_counter,
    }
    output_schema = None
    if config.output.schema:
        output_schema = OutputSchema(
            name=config.output.name,
            schema=_plain_data(config.output.schema),
            strict=config.output.strict,
        )
    return ReasoningSpec(
        strategy=reasoning.strategy,
        system=system,
        max_iterations=reasoning.max_iterations,
        knobs=knobs,
        policy=_flow_policy(config),
        timeout=reasoning.timeout or config.limits.max_runtime_seconds,
        llm_kwargs=dict(reasoning.llm_kwargs),
        output_schema=output_schema,
    )


async def _inject_memory_context(
    task: Content,
    *,
    config: AgentConfig,
    memory_store: Any,
) -> Content:
    if not (
        isinstance(task, str)
        and config.memory.private_enabled
        and config.memory.read
        and config.memory.inject
        and memory_store is not None
    ):
        return task

    results = await memory_store.search(task, k=config.memory.inject_k)
    if not results:
        return task

    lines = ["Relevant private memories:"]
    for result in results:
        text = " ".join(str(v) for v in result.node.content.values() if isinstance(v, str))
        lines.append(f"- [{result.node.type}] {text}")
    lines.append("")
    lines.append(f"User task: {task}")
    return "\n".join(lines)


def _flow_errors(flow_result: FlowResult) -> tuple[str, ...]:
    errors: list[str] = []
    for result in flow_result.results.values():
        if result.error:
            errors.append(result.error)
    return tuple(errors)


def _memory_tool_allowed(name: str, config: AgentConfig) -> bool:
    read_tools = {"recall", "explore_memory", "list_memories", "find_duplicate_memories"}
    write_tools = {"remember", "forget_memory", "consolidate_memories"}
    return (name in read_tools and config.memory.read) or (
        name in write_tools and config.memory.write
    )


def _plain_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_data(item) for item in value]
    return value
