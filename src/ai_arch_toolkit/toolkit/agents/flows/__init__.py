"""Agent architectures as Flow factories."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.agents.flows._lats import lats_flow, lats_initial_state
from ai_arch_toolkit.toolkit.agents.flows._llm_compiler import (
    llm_compiler_flow,
    llm_compiler_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._plan_execute import (
    plan_execute_flow,
    plan_execute_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.agents.flows._reflexion import (
    reflexion_flow,
    reflexion_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._rewoo import rewoo_flow, rewoo_initial_state
from ai_arch_toolkit.toolkit.agents.flows._self_discovery import (
    self_discovery_flow,
    self_discovery_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._tot import tot_flow, tot_initial_state

__all__ = [
    "lats_flow",
    "lats_initial_state",
    "llm_compiler_flow",
    "llm_compiler_initial_state",
    "plan_execute_flow",
    "plan_execute_initial_state",
    "react_flow",
    "react_initial_state",
    "reflexion_flow",
    "reflexion_initial_state",
    "rewoo_flow",
    "rewoo_initial_state",
    "self_discovery_flow",
    "self_discovery_initial_state",
    "tot_flow",
    "tot_initial_state",
]
