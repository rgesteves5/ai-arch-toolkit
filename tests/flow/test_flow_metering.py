"""The flow opens a run-scoped meter; LLM calls inside steps are metered and projected."""

from __future__ import annotations

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.toolkit.flow._flow import Flow

MODEL = "claude-sonnet-4-6"


class FakeProvider:
    async def complete(self, messages, *, system=None, tools=None, **kwargs) -> Response:
        return Response(text="ok", usage=Usage(input_tokens=12, output_tokens=4), model=MODEL)


def make_llm() -> LLM:
    llm = LLM(MODEL, api_key="test")
    llm._provider = FakeProvider()  # type: ignore[assignment]
    return llm


async def test_flow_meters_an_llm_call_inside_a_step():
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")

    result = await Flow(Step(name="s", fn=call_model), name="metered").run(State())
    meter = result.trace.metadata["meter"]
    assert meter["llm_calls"] == 1
    assert meter["input_tokens"] == 12 and meter["output_tokens"] == 4
    assert meter["cost"] > 0 and not meter["over_budget"]


async def test_flow_accumulates_metering_across_steps():
    llm = make_llm()

    async def call_model(snap: StateSnapshot) -> Result:
        await llm.complete("hi")
        return Result(value="done")

    flow = Flow(
        Step(name="a", fn=call_model),
        Step(name="b", fn=call_model),
        name="two-step",
    )
    meter = (await flow.run(State())).trace.metadata["meter"]
    assert meter["llm_calls"] == 2 and meter["input_tokens"] == 24 and meter["output_tokens"] == 8


async def test_flow_without_an_llm_reports_zero_metering():
    async def noop(snap: StateSnapshot) -> Result:
        return Result(value="done")

    result = await Flow(Step(name="s", fn=noop), name="plain").run(State())
    meter = result.trace.metadata["meter"]
    assert meter["llm_calls"] == 0 and meter["cost"] == 0.0 and not meter["over_budget"]
