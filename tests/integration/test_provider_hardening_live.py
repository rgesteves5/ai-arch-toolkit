"""R02 live checks: what each adapter's hardening can only prove against the real API.

Owner-run only (``pytest -m live_api``; never in CI). Each test makes one to three small calls to
the provider's cheapest suitable model with a low output limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ai_arch_toolkit import LLM, ToolGroup, run_tools, tool
from ai_arch_toolkit.core import APIError, MeterScope, Response, RunConfig
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetPolicy
from tests.integration.conftest import (
    skip_no_anthropic,
    skip_no_gemini,
    skip_no_meta,
    skip_no_openai,
    skip_no_xai,
)

pytestmark = [pytest.mark.integration, pytest.mark.live_api]


@tool
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a * b)


PROMPT = (
    "Compute 2+3 with the add tool and 4*5 with the multiply tool. Call both tools now, in one "
    "turn, then answer with both results."
)


@dataclass(frozen=True, slots=True)
class Live:
    """How one provider is exercised: its cheapest reasoning model and its limits."""

    model: str
    max_tokens: int
    thinking: dict[str, Any]
    oversized: dict[str, Any]  # request options the API refuses with a 4xx
    no_thinking: dict[str, Any] = field(default_factory=dict)


OPENAI = Live(
    model="gpt-5-nano",
    max_tokens=1024,
    thinking={"thinking": True, "thinking_effort": "low"},
    oversized={"max_tokens": 10_000_000},
)
# grok-4.3: the cheapest Grok whose reasoning effort can be set, down to none
# (https://docs.x.ai/developers/models/grok-4.3); a temperature outside 0-2 is refused.
XAI = Live(
    model="grok-4.3",
    max_tokens=1024,
    thinking={"thinking_effort": "low"},
    no_thinking={"thinking_effort": "none"},
    oversized={"temperature": 5.0},
)
# gemini-3.1-flash-lite: the cheapest stable Gemini 3 (https://ai.google.dev/gemini-api/docs/models),
# whose thinking level goes down to minimal
# (https://ai.google.dev/gemini-api/docs/generate-content/thinking).
GEMINI = Live(
    model="gemini-3.1-flash-lite",
    max_tokens=1024,
    thinking={"thinking": True, "thinking_effort": "low"},
    no_thinking={"thinking_effort": "minimal"},
    oversized={"max_tokens": 10_000_000},
)
# muse-spark-1.3: the model the M01 probes ran on (the contributor tier is cheaper but may need
# another agreement); Muse Spark always reasons, from "minimal" up
# (https://dev.meta.ai/docs/reasoning). Meta tunes it for temperature=1.0.
META = Live(
    model="muse-spark-1.3",
    max_tokens=1024,
    thinking={"thinking": True, "thinking_effort": "low", "temperature": 1.0},
    no_thinking={"thinking_effort": "minimal", "temperature": 1.0},
    oversized={"max_tokens": 10_000_000},
)
# claude-haiku-4-5: the cheapest Claude; it thinks on a budget, which the effort sets (low: 2048
# tokens, added to max_tokens) (https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting).
ANTHROPIC = Live(
    model="claude-haiku-4-5",
    max_tokens=1024,
    thinking={"thinking": True, "thinking_effort": "low"},
    oversized={"max_tokens": 10_000_000},
)


async def _parallel_turn(llm: LLM, group: ToolGroup, *, streamed: bool) -> Response:
    if not streamed:
        return await llm.complete(PROMPT, tools=group)
    stream = llm.stream_events(PROMPT, tools=group)
    async for _ in stream:
        pass
    assert stream.response is not None
    return stream.response


async def _parallel_calls_replay(live: Live, *, streamed: bool) -> None:
    group = ToolGroup(add, multiply)
    async with LLM(live.model, max_tokens=live.max_tokens) as llm:
        first = await _parallel_turn(llm, group, streamed=streamed)
        assert {call.name for call in first.tool_calls} == {"add", "multiply"}, first
        results = await run_tools(first, group)
        history = [{"role": "user", "content": PROMPT}, first.to_message(), *results]
        final = await llm.complete(history, tools=group)
    assert "5" in final.text and "20" in final.text, final.text


async def _thinking_on_and_off(live: Live) -> None:
    async with LLM(live.model, max_tokens=live.max_tokens) as llm:
        off = await llm.complete("Reply with the single word: ready.", **live.no_thinking)
        on = await llm.complete("Reply with the single word: ready.", **live.thinking)
    assert "ready" in off.text.lower()
    assert "ready" in on.text.lower()


async def _a_4xx_under_a_cost_cap_leaves_the_next_call_admitted(live: Live) -> None:
    controller = BudgetController(BudgetPolicy(max_cost=0.05))
    async with LLM(live.model, max_tokens=live.max_tokens) as llm:
        with MeterScope(RunConfig(controller=controller)) as scope:
            with pytest.raises(APIError) as refused:
                await llm.complete("hi", **live.oversized)
            assert 400 <= refused.value.status_code < 500
            answer = await llm.complete("Reply with the single word: ready.")
    assert "ready" in answer.text.lower()
    assert scope.snapshot().llm_calls == 2


@skip_no_openai
@pytest.mark.timeout(120)
@pytest.mark.parametrize("streamed", [False, True], ids=["complete", "stream"])
async def test_openai_parallel_calls_replay(streamed: bool) -> None:
    await _parallel_calls_replay(OPENAI, streamed=streamed)


@skip_no_openai
@pytest.mark.timeout(120)
async def test_openai_thinking_on_and_off() -> None:
    await _thinking_on_and_off(OPENAI)


@skip_no_openai
@pytest.mark.timeout(120)
async def test_openai_a_4xx_under_a_cost_cap_leaves_the_next_call_admitted() -> None:
    await _a_4xx_under_a_cost_cap_leaves_the_next_call_admitted(OPENAI)


@skip_no_xai
@pytest.mark.timeout(120)
@pytest.mark.parametrize("streamed", [False, True], ids=["complete", "stream"])
async def test_xai_parallel_calls_replay(streamed: bool) -> None:
    await _parallel_calls_replay(XAI, streamed=streamed)


@skip_no_xai
@pytest.mark.timeout(120)
async def test_xai_thinking_on_and_off() -> None:
    await _thinking_on_and_off(XAI)


@skip_no_xai
@pytest.mark.timeout(120)
async def test_xai_a_4xx_under_a_cost_cap_leaves_the_next_call_admitted() -> None:
    await _a_4xx_under_a_cost_cap_leaves_the_next_call_admitted(XAI)


@skip_no_gemini
@pytest.mark.timeout(120)
@pytest.mark.parametrize("streamed", [False, True], ids=["complete", "stream"])
async def test_gemini_parallel_calls_replay(streamed: bool) -> None:
    await _parallel_calls_replay(GEMINI, streamed=streamed)


@skip_no_gemini
@pytest.mark.timeout(120)
async def test_gemini_thinking_on_and_off() -> None:
    await _thinking_on_and_off(GEMINI)


@skip_no_gemini
@pytest.mark.timeout(120)
async def test_gemini_a_4xx_under_a_cost_cap_leaves_the_next_call_admitted() -> None:
    await _a_4xx_under_a_cost_cap_leaves_the_next_call_admitted(GEMINI)


@skip_no_meta
@pytest.mark.timeout(180)
@pytest.mark.parametrize("streamed", [False, True], ids=["complete", "stream"])
async def test_meta_parallel_calls_replay(streamed: bool) -> None:
    await _parallel_calls_replay(META, streamed=streamed)


@skip_no_meta
@pytest.mark.timeout(180)
async def test_meta_thinking_on_and_off() -> None:
    await _thinking_on_and_off(META)


@skip_no_meta
@pytest.mark.timeout(180)
async def test_meta_a_4xx_under_a_cost_cap_leaves_the_next_call_admitted() -> None:
    await _a_4xx_under_a_cost_cap_leaves_the_next_call_admitted(META)


@skip_no_anthropic
@pytest.mark.timeout(120)
@pytest.mark.parametrize("streamed", [False, True], ids=["complete", "stream"])
async def test_anthropic_parallel_calls_replay(streamed: bool) -> None:
    await _parallel_calls_replay(ANTHROPIC, streamed=streamed)


@skip_no_anthropic
@pytest.mark.timeout(120)
async def test_anthropic_thinking_on_and_off() -> None:
    await _thinking_on_and_off(ANTHROPIC)


@skip_no_anthropic
@pytest.mark.timeout(120)
async def test_anthropic_a_4xx_under_a_cost_cap_leaves_the_next_call_admitted() -> None:
    await _a_4xx_under_a_cost_cap_leaves_the_next_call_admitted(ANTHROPIC)
