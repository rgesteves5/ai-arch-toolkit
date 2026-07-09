"""Data model for the agent swarm nano project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ai_arch_toolkit.core import Content, Usage
from ai_arch_toolkit.toolkit.agents import Agent, AgentResult

type AgentID = str
type GridID = str
type MessageKind = Literal["direct", "broadcast", "request", "reply", "system"]
type SwarmMode = Literal["parallel", "sequential"]
type SwarmEventType = Literal[
    "swarm_start",
    "swarm_end",
    "agent_start",
    "agent_end",
    "agent_error",
    "message",
    "note",
]


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class SwarmPermissions:
    """Access rules for one agent inside a swarm run."""

    read_inbox: bool = True
    send_direct_messages: bool = True
    broadcast_messages: bool = True
    read_shared_notes: bool = True
    write_shared_notes: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class Grid:
    """Named collaboration space for agents, notes, and messages."""

    id: GridID
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentNode:
    """A live swarm participant backed by a toolkit Agent."""

    id: AgentID
    agent: Agent
    name: str = ""
    role: str = ""
    grid_id: GridID = "default"
    permissions: SwarmPermissions = field(default_factory=SwarmPermissions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    """Message exchanged through the swarm runtime."""

    id: str
    sender: AgentID
    recipients: tuple[AgentID, ...]
    content: str
    kind: MessageKind = "direct"
    reply_to: str | None = None
    grid_id: GridID = "default"
    created_at: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedNote:
    """Shared note written by an agent into a grid."""

    id: str
    author: AgentID
    content: str
    grid_id: GridID = "default"
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SwarmPolicy:
    """Execution and context policy for a swarm run."""

    mode: SwarmMode = "parallel"
    finalizer_id: AgentID | None = None
    max_concurrency: int | None = None
    stop_on_error: bool = False
    reset_bus_on_run: bool = True
    include_inbox: bool = True
    include_shared_notes: bool = True
    include_previous_outputs: bool = True
    record_outputs_as_notes: bool = True
    context_message_limit: int = 20
    context_note_limit: int = 20
    context_output_chars: int = 4000


@dataclass(frozen=True, slots=True, kw_only=True)
class SwarmError:
    """Non-fatal error captured during a swarm run."""

    agent_id: AgentID
    message: str
    exception_type: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class SwarmRunResult:
    """Outcome of one swarm run."""

    run_id: str
    task: Content
    policy: SwarmPolicy
    outputs: dict[AgentID, AgentResult] = field(default_factory=dict)
    messages: tuple[Message, ...] = ()
    shared_notes: tuple[SharedNote, ...] = ()
    errors: tuple[SwarmError, ...] = ()
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    started_at: datetime = field(default_factory=_now)
    ended_at: datetime = field(default_factory=_now)

    @property
    def final_text(self) -> str:
        """Best textual answer for the whole swarm."""
        if self.policy.finalizer_id and self.policy.finalizer_id in self.outputs:
            return self.outputs[self.policy.finalizer_id].text
        return "\n\n".join(
            f"[{agent_id}]\n{result.text}" for agent_id, result in self.outputs.items()
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SwarmEvent:
    """Observable event from swarm execution."""

    type: SwarmEventType
    run_id: str
    agent_id: AgentID | None = None
    message: Message | None = None
    note: SharedNote | None = None
    result: AgentResult | SwarmRunResult | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
