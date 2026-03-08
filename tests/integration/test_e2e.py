from __future__ import annotations

import pytest

from ai_arch_toolkit.core import LLM, RateLimitMiddleware, State, ToolGroup, tool, tool_result
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from tests.integration.conftest import skip_no_openai

MODEL = "gpt-4.1-nano"


# ---------------------------------------------------------------------------
# Test 1: LLM complete + stream
# ---------------------------------------------------------------------------


@skip_no_openai
@pytest.mark.timeout(30)
@pytest.mark.integration
async def test_llm_complete_and_stream():
    llm = LLM(MODEL)
    resp = await llm.complete("Say hello in one word")
    assert resp.text
    assert resp.usage.input_tokens > 0
    assert resp.attempts

    chunks: list[str] = []
    stream = llm.stream("Count 1 to 3")
    async for chunk in stream:
        chunks.append(chunk)
    final = stream.response
    assert final.text
    assert chunks
    await llm.close()


# ---------------------------------------------------------------------------
# Test 2: Tool calling round-trip
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> str:
    """Add two integers together.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


@skip_no_openai
@pytest.mark.timeout(30)
@pytest.mark.integration
async def test_tool_calling_round_trip():
    llm = LLM(MODEL)
    tools = ToolGroup(add)

    resp = await llm.complete("What is 3 + 4?", tools=tools)
    assert resp.tool_calls, "Expected at least one tool call"

    tc = resp.tool_calls[0]
    result = tools.execute(tc)
    assert result == "7"

    messages = [
        {"role": "user", "content": "What is 3 + 4?"},
        {
            "role": "assistant",
            "content": resp.text or None,
            "tool_calls": [
                {"id": t.id, "name": t.name, "input": t.input} for t in resp.tool_calls
            ],
        },
        tool_result(result, tool_use_id=tc.id, name=tc.name),
    ]
    final = await llm.complete(messages, tools=tools)
    assert "7" in final.text
    await llm.close()


# ---------------------------------------------------------------------------
# Test 3: ReAct flow e2e
# ---------------------------------------------------------------------------


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a * b)


@skip_no_openai
@pytest.mark.timeout(30)
@pytest.mark.integration
async def test_react_agent_e2e():
    llm = LLM(MODEL)
    flow = react_flow(llm, ToolGroup(multiply), max_iterations=5)
    state = State(operational=react_initial_state("What is 7 times 8?"))

    result = await flow.run(state)
    assert "56" in state["response"].text
    assert result.trace.steps
    await llm.close()


# ---------------------------------------------------------------------------
# Test 4: Middleware chain (rate limit)
# ---------------------------------------------------------------------------


@skip_no_openai
@pytest.mark.timeout(30)
@pytest.mark.integration
async def test_middleware_rate_limit():
    llm = LLM(MODEL, middleware=[RateLimitMiddleware(120)])
    resp = await llm.complete("Say hi")
    assert resp.text
    await llm.close()


# ---------------------------------------------------------------------------
# Test 5: Fallback chain e2e
# ---------------------------------------------------------------------------


@skip_no_openai
@pytest.mark.timeout(30)
@pytest.mark.integration
async def test_fallback_chain():
    llm = LLM(MODEL, fallback=MODEL)
    resp = await llm.complete("Say hello")
    assert resp.text
    assert len(resp.attempts) >= 1
    await llm.close()
