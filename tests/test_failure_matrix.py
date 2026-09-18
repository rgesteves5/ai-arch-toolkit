"""Failure disposition x scope x path x recovery: the public call contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from itertools import product

import pytest

from ai_arch_toolkit.core import (
    LLM,
    MeterScope,
    Money,
    Policy,
    Response,
    Result,
    RunConfig,
    State,
    Step,
    Usage,
)
from ai_arch_toolkit.core._exceptions import (
    APIError,
    ProviderTimeout,
    RateLimitError,
    RequestError,
    ResponseError,
    TransportError,
)
from ai_arch_toolkit.core._providers._base import BaseProvider, StreamState
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.core._step_engine import execute_step
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy

MODEL = "claude-sonnet-4-6"
USAGE = Usage(input_tokens=4, output_tokens=2)
KNOWN = Money.from_usd(0.000042)
BOUND = Money.from_usd(0.000507)  # ceil(35 serialized request chars/4) + max_tokens=32
FAILURES = (
    "429",
    "5xx",
    "4xx",
    "connect",
    "read_timeout",
    "cancel",
    "request",
    "response",
    "midstream",
    "abandon",
)
MODES = ("off", "measure", "soft", "strict")
PATHS = ("complete", "stream", "stream_events")
RECOVERIES = ("none", "retry", "fallback", "next", "step_retry", "step_fallback")


def failure(kind: str) -> BaseException:
    errors = {
        "429": RateLimitError(429, "busy"),
        "5xx": APIError(503, "busy"),
        "4xx": APIError(400, "bad"),
        "connect": TransportError("refused"),
        "read_timeout": ProviderTimeout("read"),
        "cancel": asyncio.CancelledError(),
        "request": RequestError("invalid request"),
        "response": ResponseError("unreadable"),
        "midstream": APIError(503, "stream failed"),
        "abandon": asyncio.CancelledError(),
    }
    return errors[kind]


class ScriptedProvider(BaseProvider):
    def __init__(self, kind: str | None) -> None:
        self.kind = kind
        self.calls = 0
        self.constructions = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def send(self) -> None:
        self.calls += 1
        self.entered.set()
        if self.calls != 1 or self.kind in (None, "request"):
            return
        if self.kind in ("cancel", "abandon"):
            await self.release.wait()
        raise failure(self.kind)

    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        await self.send()
        return Response(text="ok", model=MODEL, usage=USAGE)

    def stream(self, messages, *, system=None, tools=None, **kwargs):
        state = StreamState()
        state.model = MODEL

        async def chunks() -> AsyncIterator[str]:
            if self.calls == 0 and self.kind in ("midstream", "abandon"):
                self.calls += 1
                self.entered.set()
                yield "partial"
                if self.kind == "midstream":
                    raise failure(self.kind)
                await self.release.wait()
            else:
                await self.send()
            state.usage = USAGE
            yield "ok"

        return chunks(), state


class ConstructionFailureLLM(LLM):
    """Exercise request construction at the facade's pure preparation boundary.

    Provider-specific preparation moves to the same boundary in R02; the current provider
    interface has no separate prepare method.
    """

    def _prepare_call(self, messages, system, tools, arguments):
        provider = self._provider
        assert isinstance(provider, ScriptedProvider)
        provider.constructions += 1
        if provider.constructions == 1:
            raise RequestError("invalid request")
        return super()._prepare_call(messages, system, tools, arguments)


def configured(provider: ScriptedProvider, recovery: str, fallback: LLM | None = None) -> LLM:
    cls = ConstructionFailureLLM if provider.kind == "request" else LLM
    llm = cls(MODEL, api_key="test", max_tokens=32, fallback=fallback)
    llm._provider = provider
    if recovery == "retry":
        llm._retry = RetryConfig(max_retries=1, base_delay=0.001, max_delay=0.001)
    return llm


def meter_context(mode: str):
    if mode == "off":
        return nullcontext(None)
    controller = None
    if mode in ("soft", "strict"):
        policy = BudgetPolicy(max_cost=5.0, reserve="strict" if mode == "strict" else "none")
        controller = BudgetController(policy)
    return MeterScope(RunConfig(controller=controller, retain_meter_events=True))


async def invoke(llm: LLM, path: str, abandon: bool = False) -> str:
    if path == "complete":
        return (await llm.complete("hi")).text
    stream = llm.stream("hi") if path == "stream" else llm.stream_events("hi")
    async with stream:
        async for _ in stream:
            if abandon:
                break
    assert stream.response is not None
    return stream.response.text


async def observed(llm: LLM, provider: ScriptedProvider, kind: str, path: str) -> str:
    first_attempt = provider.calls == 0
    task = asyncio.create_task(invoke(llm, path, abandon=kind == "abandon" and first_attempt))
    if first_attempt and (kind == "cancel" or (kind == "abandon" and path == "complete")):
        await provider.entered.wait()
        task.cancel()
    try:
        return await task
    except BaseException as exc:
        return type(exc).__name__


async def recovered(llm, provider, backup, kind, path, recovery):
    if recovery.startswith("step_"):

        async def run(_):
            text = await observed(llm, provider, kind, path)
            return Result(value=text, error=None if text == "ok" else text)

        async def fallback(_):
            return Result(value=await invoke(backup, path))

        policy = Policy(
            max_cost=1.0,
            retry=RetryConfig(
                max_retries=1 if recovery == "step_retry" else 0, base_delay=0.001, max_delay=0.001
            ),
            fallback=Step(name="backup", fn=fallback),
            on_exhausted="fallback" if recovery == "step_fallback" else "halt",
        )
        result, _ = await execute_step(
            Step(name="call", fn=run, policy=policy), State().snapshot()
        )
        return result.value if result.is_ok else result.error
    outcome = await observed(llm, provider, kind, path)
    if recovery == "next":
        return await observed(llm, provider, "success", path)
    return outcome


def expectation(kind, path, recovery, mode):
    delivered = path != "complete" and kind in ("midstream", "abandon")
    external = kind in ("cancel", "abandon")
    retriable = kind in ("429", "5xx", "connect", "read_timeout", "midstream")
    succeeds = (
        recovery in ("next", "step_retry", "step_fallback")
        or (recovery == "retry" and retriable and not delivered and not external)
        or (recovery == "fallback" and kind != "request" and not delivered and not external)
    )
    primary = 2 if succeeds and recovery in ("next", "step_retry", "retry") else 1
    primary -= int(kind == "request")
    backup = int(succeeds and recovery in ("fallback", "step_fallback"))
    served = succeeds
    if mode == "measure" and recovery == "step_retry" and kind not in ("429", "request"):
        succeeds = False  # max_cost must fail closed on an unbounded measured failure.
    return succeeds, primary, backup, served


@pytest.mark.parametrize(
    "kind,mode,path,recovery",
    [
        pytest.param(*values, id="-".join(values))
        for values in product(FAILURES, MODES, PATHS, RECOVERIES)
    ],
)
async def test_failure_matrix(kind, mode, path, recovery):
    primary = ScriptedProvider(kind)
    secondary = ScriptedProvider(None)
    backup = configured(secondary, "none")
    llm = configured(primary, recovery, backup if recovery == "fallback" else None)
    success, primary_calls, secondary_calls, served = expectation(kind, path, recovery, mode)
    with meter_context(mode) as scope:
        result = await recovered(llm, primary, backup, kind, path, recovery)
        assert (result == "ok") == success, result
        if served and not success:
            assert "could not be priced (fail-closed)" in result
        elif not success:
            expected = (
                "partial"
                if kind == "abandon" and path != "complete"
                else type(failure(kind)).__name__
            )
            assert result == expected
        assert primary.calls == primary_calls
        assert secondary.calls == secondary_calls
        if scope is None:
            return
        snap = scope.snapshot()
        sent = primary_calls + secondary_calls
        assert snap.llm_calls == sent
        assert snap.cost == (KNOWN if served else Money.zero())
        uncertain = kind not in ("429", "request")
        bounded = uncertain and mode in ("soft", "strict")
        assert snap.unknown_cost_count == int(uncertain and not bounded)
        assert snap.uncertain_cost_count == int(bounded)
        assert snap.uncertain_cost == (BOUND if bounded else Money.zero())
        assert not scope.has_live_ops(scope.run_span_id)
        # The next operation's admission remains usable, including after an uncertain failure.
        assert await invoke(backup, path) == "ok"
