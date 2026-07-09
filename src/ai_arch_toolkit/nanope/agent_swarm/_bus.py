"""In-memory communication bus for the agent swarm nano project."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from typing import Any
from uuid import uuid4

from ai_arch_toolkit.core import ToolGroup, tool
from ai_arch_toolkit.nanope.agent_swarm._models import (
    AgentID,
    GridID,
    Message,
    MessageKind,
    SharedNote,
    SwarmPermissions,
)

_BROADCAST_RECIPIENT = "*"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class SwarmBus:
    """Thread-safe in-memory message and shared-note store."""

    __slots__ = ("_agent_ids", "_lock", "_messages", "_notes")

    def __init__(self) -> None:
        self._agent_ids: set[AgentID] = set()
        self._messages: list[Message] = []
        self._notes: list[SharedNote] = []
        self._lock = RLock()

    def register_agents(self, agent_ids: Iterable[AgentID]) -> None:
        """Register known agent ids for direct-message validation."""
        with self._lock:
            self._agent_ids.update(agent_ids)

    def reset(self) -> None:
        """Clear run-local messages and notes while keeping registered agents."""
        with self._lock:
            self._messages.clear()
            self._notes.clear()

    def send(
        self,
        *,
        sender: AgentID,
        recipient: AgentID,
        content: str,
        kind: MessageKind = "direct",
        reply_to: str | None = None,
        grid_id: GridID = "default",
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Send a direct message to one agent."""
        if kind == "broadcast":
            raise ValueError("use broadcast() for broadcast messages")
        with self._lock:
            self._ensure_recipient(recipient)
            message = Message(
                id=_new_id("msg"),
                sender=sender,
                recipients=(recipient,),
                content=content,
                kind=kind,
                reply_to=reply_to,
                grid_id=grid_id,
                metadata=dict(metadata or {}),
            )
            self._messages.append(message)
            return message

    def broadcast(
        self,
        *,
        sender: AgentID,
        content: str,
        grid_id: GridID = "default",
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Broadcast a message to all agents in the grid."""
        with self._lock:
            message = Message(
                id=_new_id("msg"),
                sender=sender,
                recipients=(_BROADCAST_RECIPIENT,),
                content=content,
                kind="broadcast",
                grid_id=grid_id,
                metadata=dict(metadata or {}),
            )
            self._messages.append(message)
            return message

    def reply(
        self,
        *,
        sender: AgentID,
        recipient: AgentID,
        content: str,
        reply_to: str,
        grid_id: GridID = "default",
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Reply to a previous message."""
        return self.send(
            sender=sender,
            recipient=recipient,
            content=content,
            kind="reply",
            reply_to=reply_to,
            grid_id=grid_id,
            metadata=metadata,
        )

    def inbox(
        self,
        agent_id: AgentID,
        *,
        grid_id: GridID | None = None,
        include_own_broadcasts: bool = False,
        limit: int | None = None,
    ) -> tuple[Message, ...]:
        """Return messages visible to one agent."""
        with self._lock:
            messages = [
                message
                for message in self._messages
                if self._is_visible(
                    message,
                    agent_id,
                    grid_id=grid_id,
                    include_own_broadcasts=include_own_broadcasts,
                )
            ]
            if limit is not None:
                messages = messages[-max(0, limit) :]
            return tuple(messages)

    def messages(self, *, grid_id: GridID | None = None) -> tuple[Message, ...]:
        """Return all stored messages, optionally filtered by grid."""
        with self._lock:
            if grid_id is None:
                return tuple(self._messages)
            return tuple(message for message in self._messages if message.grid_id == grid_id)

    def add_note(
        self,
        *,
        author: AgentID,
        content: str,
        grid_id: GridID = "default",
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> SharedNote:
        """Add a shared note to one grid."""
        with self._lock:
            note = SharedNote(
                id=_new_id("note"),
                author=author,
                content=content,
                grid_id=grid_id,
                tags=tuple(tags),
                metadata=dict(metadata or {}),
            )
            self._notes.append(note)
            return note

    def notes(
        self, *, grid_id: GridID | None = None, limit: int | None = None
    ) -> tuple[SharedNote, ...]:
        """Return shared notes, optionally filtered by grid."""
        with self._lock:
            notes = self._notes
            if grid_id is not None:
                notes = [note for note in notes if note.grid_id == grid_id]
            if limit is not None:
                notes = notes[-max(0, limit) :]
            return tuple(notes)

    def search_notes(
        self,
        query: str,
        *,
        grid_id: GridID | None = None,
        limit: int = 10,
    ) -> tuple[SharedNote, ...]:
        """Search shared notes with a simple case-insensitive substring match."""
        query_text = query.strip().lower()
        notes = self.notes(grid_id=grid_id)
        if not query_text:
            return notes[-max(0, limit) :]
        matches = [
            note
            for note in notes
            if query_text in note.content.lower()
            or query_text in note.author.lower()
            or any(query_text in tag.lower() for tag in note.tags)
        ]
        return tuple(matches[-max(0, limit) :])

    def _ensure_recipient(self, recipient: AgentID) -> None:
        if self._agent_ids and recipient not in self._agent_ids:
            known = ", ".join(sorted(self._agent_ids))
            raise ValueError(f"unknown swarm recipient {recipient!r}; known agents: {known}")

    def _is_visible(
        self,
        message: Message,
        agent_id: AgentID,
        *,
        grid_id: GridID | None,
        include_own_broadcasts: bool,
    ) -> bool:
        if grid_id is not None and message.grid_id != grid_id:
            return False
        if agent_id in message.recipients:
            return True
        if _BROADCAST_RECIPIENT not in message.recipients:
            return False
        return include_own_broadcasts or message.sender != agent_id


class SharedNotes:
    """Small grid-scoped facade over SwarmBus notes."""

    __slots__ = ("_bus", "_grid_id")

    def __init__(self, bus: SwarmBus, *, grid_id: GridID = "default") -> None:
        self._bus = bus
        self._grid_id = grid_id

    def add(
        self,
        *,
        author: AgentID,
        content: str,
        tags: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> SharedNote:
        """Add a note to this facade's grid."""
        return self._bus.add_note(
            author=author,
            content=content,
            grid_id=self._grid_id,
            tags=tags,
            metadata=metadata,
        )

    def list(self, *, limit: int | None = None) -> tuple[SharedNote, ...]:
        """List notes in this facade's grid."""
        return self._bus.notes(grid_id=self._grid_id, limit=limit)

    def search(self, query: str, *, limit: int = 10) -> tuple[SharedNote, ...]:
        """Search notes in this facade's grid."""
        return self._bus.search_notes(query, grid_id=self._grid_id, limit=limit)


def format_messages(messages: Iterable[Message]) -> str:
    """Format messages for an agent prompt or tool return."""
    lines = []
    for message in messages:
        recipients = ", ".join(message.recipients)
        lines.append(
            f"- {message.id} [{message.kind}] {message.sender} -> {recipients}: {message.content}"
        )
    return "\n".join(lines) if lines else "(none)"


def format_notes(notes: Iterable[SharedNote]) -> str:
    """Format shared notes for an agent prompt or tool return."""
    lines = []
    for note in notes:
        tags = f" tags={','.join(note.tags)}" if note.tags else ""
        lines.append(f"- {note.id} [{note.author}{tags}]: {note.content}")
    return "\n".join(lines) if lines else "(none)"


def swarm_tool_group(
    bus: SwarmBus,
    agent_id: AgentID,
    *,
    permissions: SwarmPermissions | None = None,
    grid_id: GridID = "default",
) -> ToolGroup:
    """Build communication tools for one swarm agent."""
    perms = permissions or SwarmPermissions()

    @tool
    def send_message(to: str, content: str) -> str:
        """Send a direct message to another agent in the swarm."""
        if not perms.send_direct_messages:
            return "error: this agent is not allowed to send direct messages"
        try:
            message = bus.send(
                sender=agent_id,
                recipient=to,
                content=content,
                grid_id=grid_id,
            )
        except Exception as exc:
            return f"error: {exc}"
        return f"sent {message.id} to {to}"

    @tool
    def broadcast_message(content: str) -> str:
        """Broadcast a message to the swarm."""
        if not perms.broadcast_messages:
            return "error: this agent is not allowed to broadcast messages"
        message = bus.broadcast(sender=agent_id, content=content, grid_id=grid_id)
        return f"broadcast {message.id}"

    @tool
    def read_inbox(limit: int = 10) -> str:
        """Read recent messages delivered to this agent."""
        if not perms.read_inbox:
            return "error: this agent is not allowed to read its inbox"
        messages = bus.inbox(agent_id, grid_id=grid_id, limit=limit)
        return format_messages(messages)

    @tool
    def write_shared_note(content: str, tags: str = "") -> str:
        """Write a note to the swarm's shared notes."""
        if not perms.write_shared_notes:
            return "error: this agent is not allowed to write shared notes"
        parsed_tags = tuple(tag.strip() for tag in tags.split(",") if tag.strip())
        note = bus.add_note(author=agent_id, content=content, grid_id=grid_id, tags=parsed_tags)
        return f"wrote {note.id}"

    @tool
    def search_shared_notes(query: str = "", limit: int = 10) -> str:
        """Search notes shared by this swarm."""
        if not perms.read_shared_notes:
            return "error: this agent is not allowed to read shared notes"
        notes = bus.search_notes(query, grid_id=grid_id, limit=limit)
        return format_notes(notes)

    return ToolGroup(
        send_message,
        broadcast_message,
        read_inbox,
        write_shared_note,
        search_shared_notes,
    )
