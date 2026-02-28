"""Agents — architecture implementations built on core/ primitives."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.agents._base import (
    AgentConfig,
    AgentEvent,
    AgentResult,
    AgentStep,
    BaseAgent,
    PhaseConfig,
    StopReason,
)
from ai_arch_toolkit.toolkit.agents._lats import LATSAgent, LATSConfig
from ai_arch_toolkit.toolkit.agents._llm_compiler import LLMCompilerAgent, LLMCompilerConfig
from ai_arch_toolkit.toolkit.agents._plan_execute import PlanExecuteAgent, PlanExecuteConfig
from ai_arch_toolkit.toolkit.agents._react import ReActAgent
from ai_arch_toolkit.toolkit.agents._reflexion import ReflexionAgent, ReflexionConfig
from ai_arch_toolkit.toolkit.agents._rewoo import ReWOOAgent, ReWOOConfig
from ai_arch_toolkit.toolkit.agents._self_discovery import SelfDiscoveryAgent, SelfDiscoveryConfig
from ai_arch_toolkit.toolkit.agents._tot import ToTAgent, ToTConfig

__all__ = [
    "AgentConfig",
    "AgentEvent",
    "AgentResult",
    "AgentStep",
    "BaseAgent",
    "LATSAgent",
    "LATSConfig",
    "LLMCompilerAgent",
    "LLMCompilerConfig",
    "PhaseConfig",
    "PlanExecuteAgent",
    "PlanExecuteConfig",
    "ReActAgent",
    "ReWOOAgent",
    "ReWOOConfig",
    "ReflexionAgent",
    "ReflexionConfig",
    "SelfDiscoveryAgent",
    "SelfDiscoveryConfig",
    "StopReason",
    "ToTAgent",
    "ToTConfig",
]
