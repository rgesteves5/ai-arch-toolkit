"""Runtime for the agent swarm nano project."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime
from uuid import uuid4

from ai_arch_toolkit.core import Content, Usage
from ai_arch_toolkit.core._sync import _run_sync, _stream_sync
from ai_arch_toolkit.nanope.agent_swarm._bus import (
    SharedNotes,
    SwarmBus,
    format_messages,
    format_notes,
)
from ai_arch_toolkit.nanope.agent_swarm._models import (
    AgentID,
    AgentNode,
    Grid,
    SwarmError,
    SwarmEvent,
    SwarmPolicy,
    SwarmRunResult,
)
from ai_arch_toolkit.toolkit.agents import AgentResult
from ai_arch_toolkit.toolkit.budget import BudgetPolicy

type EmitFn = Callable[[SwarmEvent], Awaitable[None]]


async def _noop_emit(_event: SwarmEvent) -> None:
    return None


class Swarm:
    """Coordinate multiple toolkit Agents as one swarm run."""

    __slots__ = ("_bus", "_grids", "_nodes", "_policy")

    def __init__(
        self,
        nodes: Sequence[AgentNode],
        *,
        policy: SwarmPolicy | None = None,
        bus: SwarmBus | None = None,
        grids: Sequence[Grid] = (),
    ) -> None:
        if not nodes:
            raise ValueError("Swarm requires at least one AgentNode")
        ids = [node.id for node in nodes]
        duplicate_ids = {agent_id for agent_id in ids if ids.count(agent_id) > 1}
        if duplicate_ids:
            raise ValueError(f"duplicate AgentNode ids: {sorted(duplicate_ids)}")

        self._nodes = tuple(nodes)
        self._policy = policy or SwarmPolicy()
        self._bus = bus or SwarmBus()
        self._bus.register_agents(ids)
        self._grids = tuple(grids) or (Grid(id="default", name="default"),)

        if self._policy.mode not in {"parallel", "sequential"}:
            raise ValueError("SwarmPolicy.mode must be 'parallel' or 'sequential'")
        if self._policy.max_concurrency is not None and self._policy.max_concurrency < 1:
            raise ValueError("SwarmPolicy.max_concurrency must be >= 1")
        if self._policy.finalizer_id is not None and self._policy.finalizer_id not in ids:
            raise ValueError(f"unknown finalizer_id {self._policy.finalizer_id!r}")

    @property
    def nodes(self) -> tuple[AgentNode, ...]:
        """Swarm participants."""
        return self._nodes

    @property
    def policy(self) -> SwarmPolicy:
        """Execution policy."""
        return self._policy

    @property
    def bus(self) -> SwarmBus:
        """Message and note bus used by this swarm."""
        return self._bus

    @property
    def grids(self) -> tuple[Grid, ...]:
        """Configured collaboration grids."""
        return self._grids

    def shared_notes(self, *, grid_id: str = "default") -> SharedNotes:
        """Return a grid-scoped shared-notes facade."""
        return SharedNotes(self._bus, grid_id=grid_id)

    async def run(
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None = None,
    ) -> SwarmRunResult:
        """Run the swarm and return an aggregate result."""
        return await self._execute(task, budget_policy=budget_policy, emit=_noop_emit)

    def run_sync(
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None = None,
    ) -> SwarmRunResult:
        """Synchronous wrapper for run()."""
        return _run_sync(self.run(task, budget_policy=budget_policy))

    async def iter(
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None = None,
    ) -> AsyncIterator[SwarmEvent]:
        """Stream events while the swarm runs."""
        queue: asyncio.Queue[SwarmEvent | BaseException | object] = asyncio.Queue()
        sentinel = object()

        async def emit(event: SwarmEvent) -> None:
            await queue.put(event)

        async def runner() -> None:
            try:
                await self._execute(task, budget_policy=budget_policy, emit=emit)
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(sentinel)

        task_handle = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not task_handle.done():
                task_handle.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task_handle
            else:
                await task_handle

    def iter_sync(
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None = None,
    ) -> Iterator[SwarmEvent]:
        """Synchronous wrapper for iter()."""
        return _stream_sync(lambda: self.iter(task, budget_policy=budget_policy))

    async def _execute(
        self,
        task: Content,
        *,
        budget_policy: BudgetPolicy | None,
        emit: EmitFn,
    ) -> SwarmRunResult:
        run_id = f"swarm_{uuid4().hex}"
        started_at = datetime.now(UTC)
        if self._policy.reset_bus_on_run:
            self._bus.reset()

        await emit(
            SwarmEvent(
                type="swarm_start",
                run_id=run_id,
                data={
                    "mode": self._policy.mode,
                    "agents": [node.id for node in self._nodes],
                    "finalizer_id": self._policy.finalizer_id,
                },
            )
        )

        outputs: dict[AgentID, AgentResult] = {}
        errors: list[SwarmError] = []
        seen_message_ids: set[str] = set()
        seen_note_ids: set[str] = set()
        work_nodes = self._work_nodes()

        if self._policy.mode == "sequential":
            await self._run_sequential(
                run_id,
                work_nodes,
                task,
                outputs,
                errors,
                budget_policy,
                emit,
            )
            await self._emit_bus_delta(run_id, seen_message_ids, seen_note_ids, emit)
        else:
            await self._run_parallel(
                run_id,
                work_nodes,
                task,
                outputs,
                errors,
                budget_policy,
                emit,
            )
            await self._emit_bus_delta(run_id, seen_message_ids, seen_note_ids, emit)

        finalizer = self._finalizer_node()
        if finalizer is not None:
            await self._run_one(
                run_id,
                finalizer,
                task,
                outputs,
                errors,
                budget_policy,
                emit,
                finalizer=True,
            )
            await self._emit_bus_delta(run_id, seen_message_ids, seen_note_ids, emit)

        result = SwarmRunResult(
            run_id=run_id,
            task=task,
            policy=self._policy,
            outputs=outputs,
            messages=self._bus.messages(),
            shared_notes=self._bus.notes(),
            errors=tuple(errors),
            usage=_sum_usage(outputs.values()),
            cost=sum(item.cost for item in outputs.values()),
            started_at=started_at,
            ended_at=datetime.now(UTC),
        )
        await emit(SwarmEvent(type="swarm_end", run_id=run_id, result=result))
        return result

    async def _run_sequential(
        self,
        run_id: str,
        nodes: Sequence[AgentNode],
        task: Content,
        outputs: dict[AgentID, AgentResult],
        errors: list[SwarmError],
        budget_policy: BudgetPolicy | None,
        emit: EmitFn,
    ) -> None:
        for node in nodes:
            await self._run_one(
                run_id,
                node,
                task,
                outputs,
                errors,
                budget_policy,
                emit,
            )

    async def _run_parallel(
        self,
        run_id: str,
        nodes: Sequence[AgentNode],
        task: Content,
        outputs: dict[AgentID, AgentResult],
        errors: list[SwarmError],
        budget_policy: BudgetPolicy | None,
        emit: EmitFn,
    ) -> None:
        max_concurrency = self._policy.max_concurrency or len(nodes) or 1
        semaphore = asyncio.Semaphore(max_concurrency)

        async def guarded(node: AgentNode) -> None:
            async with semaphore:
                await self._run_one(
                    run_id,
                    node,
                    task,
                    outputs,
                    errors,
                    budget_policy,
                    emit,
                )

        await asyncio.gather(*(guarded(node) for node in nodes))

    async def _run_one(
        self,
        run_id: str,
        node: AgentNode,
        task: Content,
        outputs: dict[AgentID, AgentResult],
        errors: list[SwarmError],
        budget_policy: BudgetPolicy | None,
        emit: EmitFn,
        *,
        finalizer: bool = False,
    ) -> None:
        await emit(SwarmEvent(type="agent_start", run_id=run_id, agent_id=node.id))
        try:
            result = await node.agent.run(
                self._agent_task(task, node, outputs, finalizer=finalizer),
                budget_policy=budget_policy,
            )
        except Exception as exc:
            error = SwarmError(
                agent_id=node.id,
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            errors.append(error)
            await emit(
                SwarmEvent(
                    type="agent_error",
                    run_id=run_id,
                    agent_id=node.id,
                    error=error.message,
                    data={"exception_type": error.exception_type},
                )
            )
            if self._policy.stop_on_error:
                raise
            return

        outputs[node.id] = result
        if (
            self._policy.record_outputs_as_notes
            and node.permissions.write_shared_notes
            and result.text.strip()
        ):
            self._bus.add_note(
                author=node.id,
                content=result.text,
                grid_id=node.grid_id,
                tags=("agent_output",),
                metadata={"run_id": run_id, "finalizer": finalizer},
            )
        await emit(
            SwarmEvent(
                type="agent_end",
                run_id=run_id,
                agent_id=node.id,
                result=result,
            )
        )

    async def _emit_bus_delta(
        self,
        run_id: str,
        seen_message_ids: set[str],
        seen_note_ids: set[str],
        emit: EmitFn,
    ) -> None:
        for message in self._bus.messages():
            if message.id in seen_message_ids:
                continue
            seen_message_ids.add(message.id)
            await emit(SwarmEvent(type="message", run_id=run_id, message=message))
        for note in self._bus.notes():
            if note.id in seen_note_ids:
                continue
            seen_note_ids.add(note.id)
            await emit(SwarmEvent(type="note", run_id=run_id, note=note))

    def _agent_task(
        self,
        task: Content,
        node: AgentNode,
        outputs: dict[AgentID, AgentResult],
        *,
        finalizer: bool,
    ) -> Content:
        if not isinstance(task, str):
            return task

        sections = [f"Task:\n{task}"]
        if node.role:
            sections.append(f"Your swarm role:\n{node.role}")
        if finalizer:
            sections.append("You are the finalizer. Synthesize the swarm result.")

        if self._policy.include_inbox and node.permissions.read_inbox:
            inbox = self._bus.inbox(
                node.id,
                grid_id=node.grid_id,
                limit=self._policy.context_message_limit,
            )
            sections.append(f"Inbox:\n{format_messages(inbox)}")

        if self._policy.include_shared_notes and node.permissions.read_shared_notes:
            notes = self._bus.notes(
                grid_id=node.grid_id,
                limit=self._policy.context_note_limit,
            )
            sections.append(f"Shared notes:\n{format_notes(notes)}")

        if self._policy.include_previous_outputs and outputs:
            sections.append(f"Previous agent outputs:\n{self._format_outputs(outputs)}")

        return "\n\n".join(sections)

    def _format_outputs(self, outputs: dict[AgentID, AgentResult]) -> str:
        chunks = []
        for agent_id, result in outputs.items():
            text = result.text
            if len(text) > self._policy.context_output_chars:
                text = text[: self._policy.context_output_chars] + "\n[truncated]"
            chunks.append(f"[{agent_id}]\n{text}")
        return "\n\n".join(chunks)

    def _work_nodes(self) -> tuple[AgentNode, ...]:
        finalizer_id = self._policy.finalizer_id
        if finalizer_id is None:
            return self._nodes
        return tuple(node for node in self._nodes if node.id != finalizer_id)

    def _finalizer_node(self) -> AgentNode | None:
        finalizer_id = self._policy.finalizer_id
        if finalizer_id is None:
            return None
        return next(node for node in self._nodes if node.id == finalizer_id)


def _sum_usage(results: Iterable[AgentResult]) -> Usage:
    input_tokens = 0
    output_tokens = 0
    cache_write_tokens = 0
    cache_read_tokens = 0
    for result in results:
        input_tokens += result.usage.input_tokens
        output_tokens += result.usage.output_tokens
        cache_write_tokens += result.usage.cache_write_tokens
        cache_read_tokens += result.usage.cache_read_tokens
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )
