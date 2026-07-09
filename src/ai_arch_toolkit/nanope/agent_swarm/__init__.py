"""Agent swarm nano project.

This module is intentionally isolated under ``nanope``. It composes existing
``toolkit.agents.Agent`` instances without changing core or toolkit APIs.
"""

from __future__ import annotations

from ai_arch_toolkit.nanope.agent_swarm._bus import (
    SharedNotes,
    SwarmBus,
    format_messages,
    format_notes,
    swarm_tool_group,
)
from ai_arch_toolkit.nanope.agent_swarm._models import (
    AgentID,
    AgentNode,
    Grid,
    GridID,
    Message,
    MessageKind,
    SharedNote,
    SwarmError,
    SwarmEvent,
    SwarmEventType,
    SwarmMode,
    SwarmPermissions,
    SwarmPolicy,
    SwarmRunResult,
)
from ai_arch_toolkit.nanope.agent_swarm._swarm import Swarm

__all__ = [
    "AgentID",
    "AgentNode",
    "Grid",
    "GridID",
    "Message",
    "MessageKind",
    "SharedNote",
    "SharedNotes",
    "Swarm",
    "SwarmBus",
    "SwarmError",
    "SwarmEvent",
    "SwarmEventType",
    "SwarmMode",
    "SwarmPermissions",
    "SwarmPolicy",
    "SwarmRunResult",
    "format_messages",
    "format_notes",
    "swarm_tool_group",
]
