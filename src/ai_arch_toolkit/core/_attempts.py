"""One admission, dispatch, recovery and terminal transition per physical LLM attempt."""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from ai_arch_toolkit.core._concurrency import inference_slot
from ai_arch_toolkit.core._exceptions import APIError, Delivery, ProviderError, RequestError
from ai_arch_toolkit.core._metering._admission import AdmissionDenied, RequestSizing
from ai_arch_toolkit.core._metering._cost import Cost
from ai_arch_toolkit.core._metering._money import Money
from ai_arch_toolkit.core._metering._operation import MeterOperation, OperationRequest
from ai_arch_toolkit.core._metering._scope import MeterScope, current_meter, current_span_id
from ai_arch_toolkit.core._middleware import Request, _run_aafter, _run_abefore
from ai_arch_toolkit.core._pricing import _estimate_response_cost, pricing
from ai_arch_toolkit.core._providers._base import BaseProvider, StreamState
from ai_arch_toolkit.core._response import Attempt, Response, StreamEvent, Usage
from ai_arch_toolkit.core._retry import _wait_before_retry
from ai_arch_toolkit.core._stream_lifecycle import close_async

if TYPE_CHECKING:
    from ai_arch_toolkit.core._llm import LLM

logger = logging.getLogger(__name__)
type Path = Literal["complete", "stream", "stream_events"]
type Item = Response | str | StreamEvent
type Phase = Literal["pending", "running", "completed", "settled", "failed", "aborted"]

_OPTIONS: dict[str, Any] = {
    "thinking": False,
    "thinking_effort": None,
    "thinking_budget": None,
    "output_schema": None,
    "tool_choice": None,
    "json_mode": False,
    "logprobs": False,
}


def _same_value(left: object, right: object) -> bool:
    try:
        return left is right or bool(left == right)
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class Arguments:
    """Explicit call arguments; fallback defaults belong to the candidate LLM."""

    options: dict[str, Any]
    extra: dict[str, Any]

    def rewritten(self, before: Request, after: Request) -> Arguments:
        options, extra = dict(self.options), dict(self.extra)
        for key in before.kwargs.keys() - after.kwargs.keys():
            if key in options:
                options[key] = _OPTIONS[key]
            else:
                extra.pop(key, None)
        for key, value in after.kwargs.items():
            if key in before.kwargs and _same_value(before.kwargs[key], value):
                continue
            if key in options:
                options[key] = value
            else:
                extra[key] = value
        return Arguments(options, extra)


def _request_size(request: Request) -> tuple[int, int]:
    """Conservative serialized character count and non-text part count."""
    chars = len(request.system or "") + len(str(request.messages))
    if request.tools:
        chars += len(str(request.tools))
    schema = request.kwargs.get("output_schema")
    if schema is not None:
        chars += len(str(schema))
    non_text = 0
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, list):
            non_text += sum(
                isinstance(part, dict) and part.get("type") not in (None, "text")
                for part in content
            )
    return chars, non_text


def request_facts(
    owner: LLM,
    request: Request,
    mode: Literal["complete", "stream"],
    scope: MeterScope,
    *,
    sized: bool = False,
    parent_span_id: str | None = None,
) -> OperationRequest:
    """Build neutral facts, stringifying content only for strict admission or a soft failure."""
    controller = scope.controller
    wants_size = sized
    if controller is not None and not wants_size:
        wants_size = (
            controller.wants_request_size()
            if isinstance(controller, RequestSizing) and callable(controller.wants_request_size)
            else True
        )
    chars, non_text = _request_size(request) if wants_size else (None, 0)
    return OperationRequest(
        kind="llm",
        parent_span_id=parent_span_id or current_span_id() or scope.run_span_id,
        mode=mode,
        model=owner._model,
        provider=owner._provider_name,
        declared_max_output_tokens=request.kwargs.get("max_tokens"),
        content_size_hint=chars,
        non_text_parts=non_text,
        has_server_tools=bool(request.tools) and any(t.get("_server_tool") for t in request.tools),
    )


def _settlement_cost(scope: MeterScope, request: OperationRequest, response: Response) -> Cost:
    pricer = scope.pricer or pricing
    if pricer is pricing and response.provider_cost is not None:
        return Cost.known(Money.from_usd(response.provider_cost))
    try:
        cost = pricer.price(request, response.usage)
    except Exception:
        logger.exception("pricer raised while settling an LLM attempt")
        return Cost.unknown("pricer raised")
    if cost.kind == "estimated":
        return Cost.unknown("pricer returned an estimate at settle")
    return cost


def dispatch(
    provider: BaseProvider, request: Request, path: Path
) -> tuple[AsyncIterator[Item], StreamState]:
    """The sole provider boundary; R02 replaces this call with pure prepare plus send."""
    if path == "complete":

        async def result() -> AsyncIterator[Item]:
            yield await provider.complete(
                request.messages, system=request.system, tools=request.tools, **request.kwargs
            )

        return result(), StreamState()
    if path == "stream_events":
        iterator, state = provider.stream_events(
            request.messages, system=request.system, tools=request.tools, **request.kwargs
        )
    else:
        iterator, state = provider.stream(
            request.messages, system=request.system, tools=request.tools, **request.kwargs
        )
    return cast("AsyncIterator[Item]", iterator), state


def _response(state: StreamState, model: str, text: str) -> Response:
    usage = state.usage or Usage()
    provider_cost = state.provider_cost
    return Response(
        text=text,
        tool_calls=tuple(state.tool_calls),
        thinking=tuple(state.thinking),
        usage=usage,
        provider_cost=provider_cost,
        cost=provider_cost if provider_cost is not None else _estimate_response_cost(model, usage),
        stop_reason=state.stop_reason,
        model=state.model or model,
        raw=state.raw,
    )


class _PhysicalAttempt:
    """Thread-safe ownership of a reservation and its single physical outcome."""

    def __init__(
        self, execution: Execution, owner: LLM, request: Request, retry_number: int
    ) -> None:
        self.execution = execution
        self.owner = owner
        self.request = request
        self.retry_number = retry_number
        self.phase: Phase = "pending"
        self.op: MeterOperation | None = None
        self.facts: OperationRequest | None = None
        self.state = StreamState()
        self.response: Response | None = None
        self.record: Attempt | None = None
        self.started_at = 0.0
        self._reserve()

    def _reserve(self) -> None:
        scope = self.execution.scope
        if scope is None:
            return
        mode = "complete" if self.execution.path == "complete" else "stream"
        facts = request_facts(
            self.owner, self.request, mode, scope, parent_span_id=self.execution.parent_span_id
        )

        def sized_facts() -> OperationRequest:
            return request_facts(
                self.owner,
                self.request,
                mode,
                scope,
                sized=True,
                parent_span_id=facts.parent_span_id,
            )

        op = scope.open(facts, failure_request=sized_facts)
        with self.execution.lock:
            if self.phase == "aborted":
                op.abort()
            else:
                self.op, self.facts = op, facts

    def refresh(self, request: Request) -> None:
        self.request = request
        scope = self.execution.scope
        if scope is None or self.facts is None:
            return
        facts = request_facts(
            self.owner, request, "stream", scope, parent_span_id=self.facts.parent_span_id
        )
        if facts == self.facts:
            return
        with self.execution.lock:
            if self.phase != "pending":
                return
            if self.op is not None:
                self.op.abort()
        self._reserve()

    def start(self) -> bool:
        with self.execution.lock:
            if self.phase != "pending":
                return False
            self.started_at = time.time()
            if self.op is not None:
                self.op.mark_started()
            self.phase = "running"
            return True

    def _record(
        self, response: Response | None = None, error: BaseException | None = None
    ) -> Attempt:
        return Attempt(
            model=self.owner._model,
            status="ok" if error is None else "failed",
            usage=response.usage if response is not None else None,
            error=str(error) if error is not None else None,
            error_type=type(error).__name__ if error is not None else None,
            status_code=error.status_code if isinstance(error, APIError) else None,
            duration=time.time() - self.started_at,
            timestamp=self.started_at,
            retry_number=self.retry_number,
        )

    def finish(self, response: Response) -> None:
        with self.execution.lock:
            if self.phase != "running":
                return
            self.phase, self.response = "completed", response
            self.record = self._record(response=response)

    def settle(self) -> None:
        scope = self.execution.scope
        with self.execution.lock:
            if self.phase != "completed" or self.response is None:
                return
            response = self.response
        cost = _settlement_cost(scope, self.facts, response) if scope and self.facts else None
        with self.execution.lock:
            if self.phase != "completed":
                return
            if self.op is not None and cost is not None:
                self.op.settle(usage=response.usage, cost=cost)
            self.phase = "settled"

    def fail(self, error: BaseException) -> None:
        with self.execution.lock:
            if self.phase not in ("pending", "running", "completed"):
                return
            if self.phase == "pending":
                if self.op is not None:
                    self.op.abort()
                self.phase = "aborted"
                return
            self.phase = "failed"
            self.record = self._record(error=error)
            if self.op is not None:
                delivery: Delivery = (
                    error.delivery if isinstance(error, ProviderError) else "indeterminate"
                )
                self.op.fail(delivery)

    def abandon(self) -> None:
        if self.op is not None:
            self.op.mark_abandoned()
        self.fail(StreamAbandoned("stream abandoned before consumption finished"))


class StreamAbandoned(Exception):
    """A caller stopped consuming a physical stream."""


class _Chain:
    """Middleware around one candidate and its fallback chain, with shared attempt history."""

    def __init__(
        self, execution: Execution, owner: LLM, request: Request, arguments: Arguments
    ) -> None:
        self.execution, self.owner = execution, owner
        self.request, self.arguments = request, arguments
        self.response: Response | None = None

    async def items(self, pending: _PhysicalAttempt | None = None) -> AsyncIterator[Item]:
        execution = self.execution
        execution.visited.add(id(self.owner))
        before = dataclasses.replace(self.request, kwargs=dict(self.request.kwargs))
        self.request = await _run_abefore(self.owner._middleware, self.request)
        if pending is not None:
            pending.refresh(self.request)
        try:
            async with _managed(
                execution._retry_items(self.owner, self.request, pending)
            ) as source:
                async for item in source:
                    yield item
            assert execution.active is not None
            self.response = execution.active.response
        except self.owner._fallback_on as error:
            if not execution.can_recover(error):
                raise
            async with _managed(self._fallback_items(before, error)) as source:
                async for item in source:
                    yield item
        assert self.response is not None
        # Transport exhaustion is not acknowledgement by a sync consumer. When there are after
        # hooks, however, their established contract observes the already-settled response.
        if execution.path == "complete" or self.owner._middleware:
            assert execution.active is not None
            execution.active.settle()
        self.response = dataclasses.replace(self.response, attempts=execution.attempts())
        self.response = await _run_aafter(self.owner._middleware, self.request, self.response)

    async def _fallback_items(self, before: Request, error: Exception) -> AsyncIterator[Item]:
        arguments = self.arguments.rewritten(before, self.request)
        for owner in self.owner._fallbacks:
            if id(owner) in self.execution.visited:
                continue
            request = owner._prepare_call(
                self.request.messages, self.request.system, self.request.tools, arguments
            )
            candidate = _Chain(self.execution, owner, request, arguments)
            try:
                async with _managed(candidate.items()) as source:
                    async for item in source:
                        yield item
                self.response = candidate.response
                return
            except self.owner._fallback_on as next_error:
                if not self.execution.can_recover(next_error):
                    raise
                error = next_error
        raise error


class Execution:
    """A call's explicit lifecycle, shared by complete, both streams and sync wrappers."""

    def __init__(self, owner: LLM, request: Request, arguments: Arguments, path: Path) -> None:
        self.path: Path = path
        self.scope = current_meter()
        self.parent_span_id = current_span_id() or (self.scope.run_span_id if self.scope else None)
        self.lock = threading.RLock()
        self.visited: set[int] = set()
        self.physical: list[_PhysicalAttempt] = []
        self.active: _PhysicalAttempt | None = None
        self.delivered = False
        self.closed = False
        self.chain = _Chain(self, owner, request, arguments)
        self.pending = self._admit(owner, request, 0) if path != "complete" else None

    def _admit(self, owner: LLM, request: Request, retry_number: int) -> _PhysicalAttempt:
        attempt = _PhysicalAttempt(self, owner, request, retry_number)
        with self.lock:
            self.physical.append(attempt)
            self.active = attempt
            if self.closed:
                attempt.abandon()
        return attempt

    def can_recover(self, error: BaseException) -> bool:
        return (
            not self.delivered
            and not self.closed
            and not isinstance(error, AdmissionDenied | RequestError)
        )

    async def _physical_items(self, attempt: _PhysicalAttempt) -> AsyncIterator[Item]:
        iterator: AsyncIterator[Item] | None = None
        text: list[str] = []
        response: Response | None = None
        try:
            async with inference_slot():
                if not attempt.start():
                    raise StreamAbandoned("stream abandoned before its attempt started")
                iterator, attempt.state = dispatch(
                    attempt.owner._provider, attempt.request, self.path
                )
                try:
                    first = await iterator.__anext__()
                except StopAsyncIteration:
                    first = None
                if self.path == "complete":
                    response = cast("Response", first)
            if first is not None:
                if self.path != "complete":
                    self.delivered = True
                text.append(_item_text(first))
                yield first
            async for item in iterator:
                text.append(_item_text(item))
                yield item
            if response is None:
                response = _response(attempt.state, attempt.owner._model, "".join(text))
            attempt.finish(response)
        except BaseException as error:
            attempt.fail(error)
            raise
        finally:
            await close_async(iterator)

    async def _retry_items(
        self,
        owner: LLM,
        request: Request,
        pending: _PhysicalAttempt | None,
    ) -> AsyncIterator[Item]:
        retries = owner._retry.max_retries if owner._retry else 0
        for retry_number in range(retries + 1):
            attempt = (
                pending
                if retry_number == 0 and pending
                else self._admit(owner, request, retry_number)
            )
            try:
                async with _managed(self._physical_items(attempt)) as source:
                    async for item in source:
                        yield item
                return
            except Exception as error:
                if not self.can_recover(error) or owner._retry is None:
                    raise
                if not await _wait_before_retry(error, retry_number, owner._retry):
                    raise

    async def items(self) -> AsyncIterator[Item]:
        finished = False
        try:
            async with _managed(self.chain.items(self.pending)) as source:
                async for item in source:
                    yield item
            finished = True
        finally:
            if not finished:
                self.abandon()

    async def complete(self) -> Response:
        async for _ in self.items():
            pass
        return self.finalize("")

    def abandon(self) -> None:
        with self.lock:
            self.closed = True
            if self.active is not None:
                self.active.abandon()

    def finalize(self, text: str) -> Response:
        with self.lock:
            active = self.active
        if active is not None:
            active.settle()
        with self.lock:
            response = None if self.closed else self.chain.response
        if response is None:
            state = active.state if active is not None else StreamState()
            model = active.owner._model if active is not None else self.chain.owner._model
            response = _response(state, model, text)
        attempts = self.attempts()
        return (
            response
            if response.attempts == attempts
            else dataclasses.replace(response, attempts=attempts)
        )

    def attempts(self) -> tuple[Attempt, ...]:
        with self.lock:
            return tuple(attempt.record for attempt in self.physical if attempt.record is not None)


@asynccontextmanager
async def _managed(iterator: AsyncIterator[Item]) -> AsyncIterator[AsyncIterator[Item]]:
    """A delegating generator owns and closes the source it forwards."""
    try:
        yield iterator
    finally:
        await close_async(iterator)


def _item_text(item: Item) -> str:
    if isinstance(item, StreamEvent):
        return item.text if item.kind == "text" else ""
    return item.text if isinstance(item, Response) else item
