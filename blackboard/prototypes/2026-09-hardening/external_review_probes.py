"""Verify behavioural claims made by two other reviewers (no network)."""
from __future__ import annotations

import asyncio

from ai_arch_toolkit.core import LLM, Result, State, Step, ToolCall, ToolGroup, tool
from ai_arch_toolkit.toolkit.agents import ReasoningSpec
from ai_arch_toolkit.toolkit.flow import Flow

print("== 1. tool argument validation: None and wrong element types ==")


@tool
def typed(value: int, items: list[int]) -> str:
    """Typed tool."""
    return f"value={value!r} items={items!r}"


group = ToolGroup(typed)
for args in ({"value": None, "items": ["wrong"]}, {"value": "7", "items": [1]}, {"value": "x", "items": [1]}):
    r = group.execute(ToolCall(id="1", name="typed", input=args))
    print(f"  {args} -> ok={r.ok} text={r.to_model_text()[:90]!r}")

print("== 2. ReasoningSpec.from_mapping with a dict policy / malformed schema ==")
try:
    spec = ReasoningSpec.from_mapping({"strategy": "react", "policy": {"timeout": 1, "retry": 2}})
    print("  policy given as dict ->", spec.policy)
except Exception as exc:  # noqa: BLE001
    print("  raised:", type(exc).__name__, exc)
try:
    spec = ReasoningSpec.from_mapping({"strategy": "react", "output_schema": 123})
    print("  output_schema=123 ->", spec.output_schema)
except Exception as exc:  # noqa: BLE001
    print("  raised:", type(exc).__name__, exc)
try:
    spec = ReasoningSpec.from_mapping({"strategy": "react", "bogus_key": 1})
    print("  unknown key -> accepted; knobs:", dict(spec.knobs) if hasattr(spec, "knobs") else None)
except Exception as exc:  # noqa: BLE001
    print("  unknown key raised:", type(exc).__name__, exc)

print("== 3. shallow immutability ==")
state = State(operational={"items": [1, 2]})
snap = state.snapshot()
snap["items"].append(99)
print("  mutate list from snapshot -> state sees:", state.get("items"))
spec = ReasoningSpec(strategy="react", knobs={"max_turns": 3})
try:
    spec.knobs["max_turns"] = 999  # type: ignore[index]
    print("  frozen ReasoningSpec.knobs mutated ->", dict(spec.knobs))
except TypeError as exc:
    print("  knobs mutation blocked:", exc)

print("== 4. in-place mutation of a snapshot value: parallel siblings vs sequential steps ==")


async def isolation() -> None:
    from ai_arch_toolkit.toolkit.flow import FlowStep

    async def start(snapshot) -> Result:
        return Result(value="s")

    async def writer(snapshot) -> Result:
        snapshot["shared"].append("written in place")  # bypasses Result.artifacts and merge
        return Result(value="w")

    async def reader(snapshot) -> Result:
        await asyncio.sleep(0.02)
        return Result(value="r", artifacts={"reader_saw": list(snapshot["shared"])})

    parallel = Flow(
        FlowStep(step=Step(name="start", fn=start)),
        FlowStep(step=Step(name="writer", fn=writer), after=("start",)),
        FlowStep(step=Step(name="reader", fn=reader), after=("start",)),
        name="parallel",
    )
    res = await parallel.run(State(operational={"shared": []}))
    print("  parallel wave : sibling saw", res.state.get("reader_saw"), "| state.shared =", res.state.get("shared"))

    sequential = Flow(Step(name="writer", fn=writer), Step(name="reader", fn=reader), name="sequential")
    res = await sequential.run(State(operational={"shared": []}))
    print("  sequential    : next step saw", res.state.get("reader_saw"), "| state.shared =", res.state.get("shared"))


asyncio.run(isolation())

print("== 5. LLM(fallback=other) mutates other ==")
c = LLM("claude-haiku-4-5", api_key="x")
b = LLM("claude-sonnet-4-6", api_key="x", fallback=c)
print("  before: b has", len(b._fallbacks), "fallback(s)")
a = LLM("claude-opus-5", api_key="x", fallback=b)
print("  after building a(fallback=b): b has", len(b._fallbacks), "fallback(s); a has", len(a._fallbacks))
