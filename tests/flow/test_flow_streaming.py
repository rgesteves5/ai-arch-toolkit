"""The streaming inside a flow (G-22, D54): each step's end carries its trace, every step that
starts ends, and the LLM calls of an iterated run stream their events."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import Any

import pytest

from ai_arch_toolkit.core import Response, StreamEvent, Usage, inference_limit, llm_events_to, tool
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.flow import Flow, FlowEvent, FlowStep
from tests.fake_provider import MODEL, FakeProvider, Reply, fake_llm
from tests.flow.event_grammar import steps_story

HELLO = Response(text="Hello", usage=Usage(input_tokens=3, output_tokens=2), model=MODEL)


def _streaming_llm() -> tuple[Any, FakeProvider]:
    return fake_llm(Reply(response=HELLO, chunks=["Hel", "lo"]))


def _asks(llm: Any, name: str = "ask") -> Step:
    async def fn(snap: StateSnapshot) -> Result:
        response = await llm.complete("hi")
        return Result(value=response.text, artifacts={name: response.text})

    return Step(name=name, fn=fn)


def _open_streams(provider: FakeProvider) -> list[int]:
    opened: list[int] = []
    original = provider.open_stream

    def counting(prepared: Any) -> Any:
        opened.append(1)
        return original(prepared)

    provider.open_stream = counting  # type: ignore[method-assign]
    return opened


async def _drain(flow: Flow) -> list[FlowEvent]:
    events: list[FlowEvent] = []
    async with flow.iter(State()) as execution:
        async for event in execution:
            events.append(event)
    return events


class TestTheChannel:
    """The core's channel: while bound, ``LLM.complete`` streams into it (D54)."""

    async def test_a_bound_sink_gets_each_event_of_a_call_with_its_id(self) -> None:
        llm, _ = _streaming_llm()
        seen: list[tuple[StreamEvent, str]] = []

        with llm_events_to(lambda event, call: seen.append((event, call))):
            response = await llm.complete("hi")

        assert [event.text for event, _ in seen] == ["Hel", "lo"]
        assert len({call for _, call in seen}) == 1
        assert response.text == "Hello"
        assert response.usage.output_tokens == 2

    async def test_two_calls_have_two_ids(self) -> None:
        llm, _ = _streaming_llm()
        calls: list[str] = []

        with llm_events_to(lambda _event, call: calls.append(call)):
            await llm.complete("one")
            await llm.complete("two")

        assert len(set(calls)) == 2

    async def test_without_a_sink_complete_does_not_stream(self) -> None:
        llm, provider = _streaming_llm()
        opened = _open_streams(provider)

        assert (await llm.complete("hi")).text == "Hello"
        with llm_events_to(None):
            await llm.complete("hi")

        assert opened == []


class TestTokensInAFlow:
    async def test_an_iterated_step_streams_the_models_text_between_its_start_and_end(
        self,
    ) -> None:
        llm, _ = _streaming_llm()
        flow = Flow(_asks(llm), name="one")

        events = await _drain(flow)

        kinds = [(event.type, event.step_name) for event in events]
        assert kinds == [
            ("flow_start", ""),
            ("step_start", "ask"),
            ("llm_event", "ask"),
            ("llm_event", "ask"),
            ("step_end", "ask"),
            ("flow_end", ""),
        ]
        streamed = [event for event in events if event.type == "llm_event"]
        assert [event.llm_event.text for event in streamed if event.llm_event] == ["Hel", "lo"]
        assert len({event.llm_call for event in streamed}) == 1
        end = events[4]
        assert end.result is not None and end.result.value == "Hello"

    async def test_run_does_not_stream(self) -> None:
        llm, provider = _streaming_llm()
        opened = _open_streams(provider)

        result = await Flow(_asks(llm), name="one").run(State())

        assert result.results["ask"].value == "Hello"
        assert opened == []

    async def test_a_nested_run_streams_under_the_outer_step(self) -> None:
        llm, _ = _streaming_llm()
        inner = Flow(_asks(llm, "inner_ask"), name="inner")

        async def outer_fn(snap: StateSnapshot) -> Result:
            result = await inner.run(State())
            return Result(value=result.results["inner_ask"].value)

        events = await _drain(Flow(Step(name="delegate", fn=outer_fn), name="outer"))

        streamed = [event for event in events if event.type == "llm_event"]
        assert [event.step_name for event in streamed] == ["delegate", "delegate"]
        assert [event.flow_name for event in streamed] == ["outer", "outer"]

    async def test_a_sync_call_from_a_thread_streams_too(self) -> None:
        llm, _ = _streaming_llm()

        async def in_thread(snap: StateSnapshot) -> Result:
            response = await asyncio.to_thread(llm.complete_sync, "hi")
            return Result(value=response.text)

        events = await _drain(Flow(Step(name="threaded", fn=in_thread), name="t"))

        streamed = [event for event in events if event.type == "llm_event"]
        assert [event.llm_event.text for event in streamed if event.llm_event] == ["Hel", "lo"]
        assert {event.step_name for event in streamed} == {"threaded"}
        steps_story(events[1:-1])  # the thread's events came inside the step


class TestEveryStepThatStartsEnds:
    async def test_a_step_end_carries_the_trace_entry(self) -> None:
        llm, _ = _streaming_llm()
        flow = Flow(_asks(llm), name="one")

        async with flow.iter(State()) as execution:
            events = [event async for event in execution]

        assert execution.result is not None
        (entry,) = execution.result.trace.steps
        (end,) = [event for event in events if event.type == "step_end"]
        assert end.step_trace is entry
        assert entry.name == "ask"

    async def test_an_engine_exception_ends_the_started_steps_before_it_arrives(self) -> None:
        async def slow(snap: StateSnapshot) -> Result:
            await asyncio.sleep(10)
            return Result(value="late")

        async def broken(snap: StateSnapshot) -> Any:
            await asyncio.sleep(0.01)
            return "not a Result"  # the engine trips on it: a bug that leaves the engine

        flow = Flow(
            FlowStep(step=Step(name="slow", fn=slow)),
            FlowStep(step=Step(name="broken", fn=broken)),
            FlowStep(step=Step(name="join", fn=slow), after=("slow", "broken")),
            name="dag",
        )
        events: list[FlowEvent] = []

        with pytest.raises(AttributeError):
            async for event in flow.iter(State()):
                events.append(event)

        story, carried = steps_story(events[1:])
        assert sorted(name for name, _ in story) == ["broken", "slow"]
        assert all(error and "the flow stopped" in error for _, error in story)
        assert all(trace is not None and trace.policy_decisions == ("halt",) for trace in carried)


class TestAgentsStream:
    async def test_an_iterated_react_agent_streams_its_answer(self) -> None:
        llm, _ = _streaming_llm()

        @tool
        def noop() -> str:
            """Do nothing."""
            return ""

        agent = Agent(ReasoningSpec(strategy="react"), llm, ToolGroup(noop))
        async with agent.iter("Say hello.") as execution:
            events = [event async for event in execution]

        streamed = [event for event in events if event.type == "llm_event"]
        assert [event.llm_event.text for event in streamed if event.llm_event] == ["Hel", "lo"]
        assert {event.step_name for event in streamed} == {"llm_call"}
        assert execution.result is not None and execution.result.text == "Hello"


class TestWhatTheReviewFound:
    """Regressions an independent review found in the first cut of A05."""

    async def test_a_call_left_running_by_its_step_drops_its_events_after_the_step_ends(
        self,
    ) -> None:
        inner, _ = fake_llm(Reply(response=HELLO, chunks=["Hel", "lo"], delay=0.2))

        async def tools(snap: StateSnapshot) -> Result:
            # What the tool executor does at a tool's deadline: stop waiting, the thread runs on.
            try:
                await asyncio.wait_for(asyncio.to_thread(inner.complete_sync, "x"), 0.05)
            except TimeoutError:
                return Result(value="tool timed out")
            return Result(value="done")

        async def after(snap: StateSnapshot) -> Result:
            await asyncio.sleep(0.4)
            return Result(value="next")

        events = await _drain(Flow(Step(name="tools", fn=tools), Step(name="next", fn=after)))

        steps_story(events[1:-1])  # no llm_event outside its step
        assert [event for event in events if event.type == "llm_event"] == []

    def test_a_call_left_running_finishes_after_a_sync_iteration_has_closed_its_loop(
        self,
    ) -> None:
        inner, _ = _streaming_llm()
        go = threading.Event()
        answers: list[str] = []
        failures: list[BaseException] = []

        def later() -> None:
            go.wait()
            try:
                answers.append(inner.complete_sync("hi").text)
            except BaseException as exc:  # reported below
                failures.append(exc)

        threads: list[threading.Thread] = []

        async def leaves_a_thread(snap: StateSnapshot) -> Result:
            context = contextvars.copy_context()  # the step's, with its LLM channel
            thread = threading.Thread(target=context.run, args=(later,))
            thread.start()
            threads.append(thread)
            return Result(value="left")

        with Flow(Step(name="leave", fn=leaves_a_thread)).iter_sync(State()) as execution:
            for _ in execution:
                pass
        go.set()
        threads[0].join(5)

        assert failures == []
        assert answers == ["Hello"]

    async def test_events_queued_when_the_engine_raises_reach_the_consumer_before_the_cuts(
        self,
    ) -> None:
        gate = asyncio.Event()

        async def ok(snap: StateSnapshot) -> Result:
            await gate.wait()
            return Result(value="a")

        async def buggy(snap: StateSnapshot) -> Any:
            await gate.wait()
            return "not a Result"

        async def slow(snap: StateSnapshot) -> Result:
            await asyncio.sleep(10)
            return Result(value="c")

        flow = Flow(
            FlowStep(step=Step(name="a", fn=ok)),
            FlowStep(step=Step(name="b", fn=buggy)),
            FlowStep(step=Step(name="c", fn=slow)),
            FlowStep(step=Step(name="z", fn=slow), after=("a",)),
            name="dag",
            max_parallelism=2,
        )
        events: list[FlowEvent] = []

        with pytest.raises(AttributeError):
            async for event in flow.iter(State()):
                events.append(event)
                if sum(e.type == "step_start" for e in events) == 2 and not gate.is_set():
                    gate.set()
                    await asyncio.sleep(0)
                    continue
                for _ in range(5):  # a consumer that does some async work per event
                    await asyncio.sleep(0)

        steps_story(events[1:])  # every step_end has its step_start before it

    async def test_inference_limit_holds_for_the_whole_call_in_an_iterated_run(self) -> None:
        holds = [asyncio.Event(), asyncio.Event()]
        llm, provider = fake_llm(
            Reply(response=HELLO, chunks=["Hel", "lo"], hold=holds[0], hold_after=1),
            Reply(response=HELLO, chunks=["Hel", "lo"], hold=holds[1], hold_after=1),
        )
        join = Step(name="join", fn=lambda snap: asyncio.sleep(0, Result(value=1)))
        flow = Flow(
            FlowStep(step=_asks(llm, "left")),
            FlowStep(step=_asks(llm, "right")),
            FlowStep(step=join, after=("left", "right")),  # a DAG: the two run at once
            name="dag",
        )
        loop = asyncio.get_running_loop()
        for hold in holds:
            loop.call_later(0.1, hold.set)

        with inference_limit(1):
            await _drain(flow)

        assert provider.peak == 1
