"""Agents — Flow-based agent architectures built on core/ primitives."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.agents._agent import Agent, AgentResult
from ai_arch_toolkit.toolkit.agents._builders import (
    BuildContext,
    FlowStrategy,
    StrategyBuilder,
    get_strategy,
    register_strategy,
    strategy_names,
)
from ai_arch_toolkit.toolkit.agents._compile import build_flow, extract_text, initial_state
from ai_arch_toolkit.toolkit.agents._manifest import (
    AgentManifestCycleError,
    AgentManifestError,
    AgentOverrideError,
    ResolvedAgentManifest,
    load_agent_manifest,
)
from ai_arch_toolkit.toolkit.agents._spec import ReasoningSpec
from ai_arch_toolkit.toolkit.agents.flows import (
    generate_review_flow,
    generate_review_initial_state,
    lats_flow,
    lats_initial_state,
    llm_compiler_flow,
    llm_compiler_initial_state,
    plan_execute_flow,
    plan_execute_initial_state,
    react_flow,
    react_initial_state,
    reflexion_flow,
    reflexion_initial_state,
    rewoo_flow,
    rewoo_initial_state,
    self_discovery_flow,
    self_discovery_initial_state,
    tot_flow,
    tot_initial_state,
)

__all__ = [
    "Agent",
    "AgentManifestCycleError",
    "AgentManifestError",
    "AgentOverrideError",
    "AgentResult",
    "BuildContext",
    "FlowStrategy",
    "ReasoningSpec",
    "ResolvedAgentManifest",
    "StrategyBuilder",
    "build_flow",
    "extract_text",
    "generate_review_flow",
    "generate_review_initial_state",
    "get_strategy",
    "initial_state",
    "lats_flow",
    "lats_initial_state",
    "llm_compiler_flow",
    "llm_compiler_initial_state",
    "load_agent_manifest",
    "plan_execute_flow",
    "plan_execute_initial_state",
    "react_flow",
    "react_initial_state",
    "reflexion_flow",
    "reflexion_initial_state",
    "register_strategy",
    "rewoo_flow",
    "rewoo_initial_state",
    "self_discovery_flow",
    "self_discovery_initial_state",
    "strategy_names",
    "tot_flow",
    "tot_initial_state",
]
