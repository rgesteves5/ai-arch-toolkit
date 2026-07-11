"""Configuration types for the advanced configurable agent nano project."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from ai_arch_toolkit.toolkit.prompts import (
    PromptSection as ToolkitPromptSection,
)
from ai_arch_toolkit.toolkit.prompts import (
    PromptStability,
)

type ReasoningStrategy = Literal[
    "react",
    "plan_execute",
    "reflexion",
    "self_discovery",
    "generate_review",
    "rewoo",
    "llm_compiler",
    "tot",
    "lats",
]

_SUPPORTED_REASONING: tuple[str, ...] = (
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


def _tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def _fingerprint(data: Mapping[str, Any]) -> str:
    payload = json.dumps(_plain(data), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_SECTION_POSITION = 1000
"""Default position for prompt sections without an explicit one.

Built-in sections occupy 100-700; this default places custom sections after
all built-ins. Use values between (e.g. 350 to land between goals and tasks)
or negatives to land before identity.
"""


class PromptSection(ToolkitPromptSection):
    """Nanope-compatible prompt section defaulting after built-in sections."""

    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        content: str,
        position: int | None = None,
        order: int | None = None,
        stability: PromptStability = "static",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if position is None and order is None:
            position = DEFAULT_SECTION_POSITION
        super().__init__(
            name=name,
            content=content,
            position=position,
            order=order,
            stability=stability,
            metadata=metadata,
        )


def _coerce_prompt_section(value: Any) -> ToolkitPromptSection:
    if isinstance(value, ToolkitPromptSection):
        return value
    if isinstance(value, Mapping):
        raw_order = value.get("order", value.get("position", DEFAULT_SECTION_POSITION))
        return PromptSection(
            name=str(value.get("name", "")),
            content=str(value.get("content", "")),
            order=int(raw_order),
            stability=value.get("stability", "static"),
            metadata=value.get("metadata") or {},
        )
    raise TypeError(
        f"extra_sections entries must be PromptSection or mapping, got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """Mandatory agent identity included in every rendered system prompt."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("identity.name is required")
        if not self.description:
            raise ValueError("identity.description is required")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentContext:
    """Optional role/task context included in the prompt only when present.

    ``extra_sections`` carries domain-specific prompt sections that don't fit
    the built-in role/goals/tasks/style/constraints shape. Each section has an
    optional ``position`` that controls render order (see
    ``_prompt.PromptSection``); by default they land after the built-ins.
    """

    role: str = ""
    goals: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    style: str = ""
    constraints: tuple[str, ...] = ()
    extra_sections: tuple[ToolkitPromptSection, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", tuple(str(v) for v in self.goals))
        object.__setattr__(self, "tasks", tuple(str(v) for v in self.tasks))
        object.__setattr__(self, "constraints", tuple(str(v) for v in self.constraints))
        object.__setattr__(
            self,
            "extra_sections",
            tuple(_coerce_prompt_section(item) for item in self.extra_sections),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelConfig:
    """LLM construction settings."""

    name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float | None = None
    fallback: tuple[str, ...] = ()
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model.name is required")
        object.__setattr__(self, "fallback", tuple(str(v) for v in self.fallback))


@dataclass(frozen=True, slots=True, kw_only=True)
class ReasoningConfig:
    """Reasoning strategy settings."""

    strategy: ReasoningStrategy = "react"
    max_iterations: int = 10
    parallel_tool_calls: bool = True
    timeout: float | None = None
    llm_kwargs: Mapping[str, Any] = field(default_factory=dict)
    strategy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    final_answer_hint: bool = True
    strip_tools_on_final: bool = False
    show_turn_counter: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in _SUPPORTED_REASONING:
            raise ValueError(
                f"Unsupported reasoning.strategy {self.strategy!r}; "
                f"supported strategies: {', '.join(_SUPPORTED_REASONING)}"
            )
        object.__setattr__(self, "llm_kwargs", _freeze(self.llm_kwargs))
        object.__setattr__(self, "strategy_kwargs", _freeze(self.strategy_kwargs))


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolsConfig:
    """Tool selection and permissions placeholder."""

    enabled: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()
    permissions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", tuple(str(v) for v in self.enabled))
        object.__setattr__(self, "disabled", tuple(str(v) for v in self.disabled))
        object.__setattr__(self, "permissions", _freeze(self.permissions))


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryConfig:
    """Private memory placeholder for later implementation."""

    private_enabled: bool = False
    read: bool = False
    write: bool = False
    inject: bool = True
    inject_k: int = 5


@dataclass(frozen=True, slots=True, kw_only=True)
class LimitsConfig:
    """Run limits handled by the agent or delegated flow."""

    max_cost: float | None = None
    max_runtime_seconds: float | None = None
    max_tool_calls: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservabilityConfig:
    """Observability toggles for run metadata."""

    trace: bool = True
    debug_prompts: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputConfig:
    """Structured output settings."""

    schema: Mapping[str, Any] = field(default_factory=dict)
    name: str = "agent_output"
    strict: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _freeze(self.schema))


@dataclass(frozen=True, slots=True, kw_only=True)
class OverridePolicy:
    """Runtime override allow/deny policy."""

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow", tuple(str(v) for v in self.allow))
        object.__setattr__(self, "deny", tuple(str(v) for v in self.deny))


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentConfig:
    """Fully parsed agent configuration."""

    identity: AgentIdentity
    model: ModelConfig
    context: AgentContext = field(default_factory=AgentContext)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    capability_profiles: tuple[str, ...] = ()
    override_policy: OverridePolicy = field(default_factory=OverridePolicy)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_profiles",
            tuple(str(v) for v in self.capability_profiles),
        )

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint of the resolved config."""
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain serializable data."""
        return {
            "identity": {
                "name": self.identity.name,
                "description": self.identity.description,
            },
            "context": {
                "role": self.context.role,
                "goals": list(self.context.goals),
                "tasks": list(self.context.tasks),
                "style": self.context.style,
                "constraints": list(self.context.constraints),
                "extra_sections": [
                    {
                        "name": section.name,
                        "content": section.content,
                        "position": section.position,
                        "stability": section.stability,
                        "metadata": _plain(section.metadata),
                    }
                    for section in self.context.extra_sections
                ],
            },
            "model": {
                "name": self.model.name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "timeout": self.model.timeout,
                "fallback": list(self.model.fallback),
                "api_key": self.model.api_key,
                "base_url": self.model.base_url,
            },
            "reasoning": {
                "strategy": self.reasoning.strategy,
                "max_iterations": self.reasoning.max_iterations,
                "parallel_tool_calls": self.reasoning.parallel_tool_calls,
                "timeout": self.reasoning.timeout,
                "llm_kwargs": _plain(self.reasoning.llm_kwargs),
                "strategy_kwargs": _plain(self.reasoning.strategy_kwargs),
                "final_answer_hint": self.reasoning.final_answer_hint,
                "strip_tools_on_final": self.reasoning.strip_tools_on_final,
                "show_turn_counter": self.reasoning.show_turn_counter,
            },
            "tools": {
                "enabled": list(self.tools.enabled),
                "disabled": list(self.tools.disabled),
                "permissions": _plain(self.tools.permissions),
            },
            "memory": {
                "private_enabled": self.memory.private_enabled,
                "read": self.memory.read,
                "write": self.memory.write,
                "inject": self.memory.inject,
                "inject_k": self.memory.inject_k,
            },
            "limits": {
                "max_cost": self.limits.max_cost,
                "max_runtime_seconds": self.limits.max_runtime_seconds,
                "max_tool_calls": self.limits.max_tool_calls,
            },
            "observability": {
                "trace": self.observability.trace,
                "debug_prompts": self.observability.debug_prompts,
            },
            "output": {
                "schema": _plain(self.output.schema),
                "name": self.output.name,
                "strict": self.output.strict,
            },
            "capability_profiles": list(self.capability_profiles),
            "override_policy": {
                "allow": list(self.override_policy.allow),
                "deny": list(self.override_policy.deny),
            },
        }


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """Named reusable config fragment."""

    name: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CapabilityProfile.name is required")
        object.__setattr__(self, "config", _freeze(self.config))

    def to_dict(self) -> dict[str, Any]:
        """Convert profile config to plain data."""
        return _plain(self.config)


def agent_config_from_mapping(data: Mapping[str, Any]) -> AgentConfig:
    """Parse an agent config from a mapping."""
    raw = _normalize_raw_config(data)

    identity_data = raw.get("identity") or {}
    model_data = raw.get("model") or {}
    context_data = raw.get("context") or {}
    reasoning_data = raw.get("reasoning") or {}
    tools_data = raw.get("tools") or {}
    memory_data = raw.get("memory") or {}
    limits_data = raw.get("limits") or {}
    observability_data = raw.get("observability") or {}
    output_data = raw.get("output") or {}
    override_policy_data = raw.get("override_policy") or {}

    return AgentConfig(
        identity=AgentIdentity(
            name=str(identity_data.get("name", "")),
            description=str(identity_data.get("description", "")),
        ),
        context=AgentContext(
            role=str(context_data.get("role", "")),
            goals=tuple(str(v) for v in _tuple(context_data.get("goals"))),
            tasks=tuple(str(v) for v in _tuple(context_data.get("tasks"))),
            style=str(context_data.get("style", "")),
            constraints=tuple(str(v) for v in _tuple(context_data.get("constraints"))),
            extra_sections=tuple(
                _coerce_prompt_section(item) for item in _tuple(context_data.get("extra_sections"))
            ),
        ),
        model=ModelConfig(
            name=str(model_data.get("name", "")),
            temperature=float(model_data.get("temperature", 0.0)),
            max_tokens=int(model_data.get("max_tokens", 4096)),
            timeout=model_data.get("timeout"),
            fallback=tuple(str(v) for v in _tuple(model_data.get("fallback"))),
            api_key=model_data.get("api_key"),
            base_url=model_data.get("base_url"),
        ),
        reasoning=ReasoningConfig(
            strategy=reasoning_data.get("strategy", "react"),
            max_iterations=int(reasoning_data.get("max_iterations", 10)),
            parallel_tool_calls=bool(reasoning_data.get("parallel_tool_calls", True)),
            timeout=reasoning_data.get("timeout"),
            llm_kwargs=reasoning_data.get("llm_kwargs", {}),
            strategy_kwargs=reasoning_data.get("strategy_kwargs", {}),
            final_answer_hint=bool(reasoning_data.get("final_answer_hint", True)),
            strip_tools_on_final=bool(reasoning_data.get("strip_tools_on_final", False)),
            show_turn_counter=bool(reasoning_data.get("show_turn_counter", False)),
        ),
        tools=ToolsConfig(
            enabled=tuple(str(v) for v in _tuple(tools_data.get("enabled"))),
            disabled=tuple(str(v) for v in _tuple(tools_data.get("disabled"))),
            permissions=tools_data.get("permissions", {}),
        ),
        memory=MemoryConfig(
            private_enabled=bool(
                memory_data.get("private_enabled", memory_data.get("private", False))
            ),
            read=bool(memory_data.get("read", False)),
            write=bool(memory_data.get("write", False)),
            inject=bool(memory_data.get("inject", True)),
            inject_k=int(memory_data.get("inject_k", 5)),
        ),
        limits=LimitsConfig(
            max_cost=limits_data.get("max_cost", limits_data.get("budget")),
            max_runtime_seconds=limits_data.get("max_runtime_seconds"),
            max_tool_calls=limits_data.get("max_tool_calls"),
        ),
        observability=ObservabilityConfig(
            trace=bool(observability_data.get("trace", True)),
            debug_prompts=bool(observability_data.get("debug_prompts", False)),
        ),
        output=OutputConfig(
            schema=output_data.get("schema", {}),
            name=str(output_data.get("name", "agent_output")),
            strict=bool(output_data.get("strict", True)),
        ),
        capability_profiles=tuple(str(v) for v in _tuple(raw.get("capability_profiles"))),
        override_policy=OverridePolicy(
            allow=tuple(str(v) for v in _tuple(override_policy_data.get("allow"))),
            deny=tuple(str(v) for v in _tuple(override_policy_data.get("deny"))),
        ),
    )


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load an agent config from TOML, or YAML when PyYAML is installed."""
    path = Path(path)
    if path.suffix == ".toml":
        return agent_config_from_mapping(tomllib.loads(path.read_text()))
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config loading requires the optional pyyaml dependency"
            ) from exc
        return agent_config_from_mapping(yaml.safe_load(path.read_text()) or {})
    raise ValueError(f"Unsupported config file extension: {path.suffix!r}")


def _normalize_raw_config(data: Mapping[str, Any]) -> dict[str, Any]:
    raw = _plain(data)

    if "identity" not in raw:
        raw["identity"] = {}
    if "name" in raw:
        raw["identity"].setdefault("name", raw["name"])
    if "description" in raw:
        raw["identity"].setdefault("description", raw["description"])

    if isinstance(raw.get("model"), str):
        raw["model"] = {"name": raw["model"]}
    elif "model" not in raw:
        raw["model"] = {}

    if "context" not in raw:
        raw["context"] = {}
    for key in ("role", "goals", "tasks", "style", "constraints", "extra_sections"):
        if key in raw:
            raw["context"].setdefault(key, raw[key])

    if "reasoning" not in raw:
        raw["reasoning"] = {}
    if "flow" in raw:
        raw["reasoning"].setdefault("strategy", raw["flow"])

    if isinstance(raw.get("tools"), list):
        raw["tools"] = {"enabled": raw["tools"]}
    elif "tools" not in raw:
        raw["tools"] = {}

    return raw
