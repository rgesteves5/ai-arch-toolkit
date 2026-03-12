"""Tests for generate_review_flow factory."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response, Usage
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._generate_review import (
    generate_review_flow,
    generate_review_initial_state,
)


def _make_response(text: str = "", cost: float = 0.001) -> Response:
    return Response(
        text=text,
        tool_calls=(),
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=cost,
    )


class TestGenerateReviewFlow:
    async def test_accepted_on_first_attempt(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(return_value=_make_response("The answer is 42."))

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(return_value=_make_response("ACCEPT"))

        flow = generate_review_flow(gen_llm, review_llm, max_cycles=3)
        state = State(operational=generate_review_initial_state("What is 6*7?"))
        await flow.run(state)

        assert state.get("accepted") is True
        assert state.get("answer") == "The answer is 42."
        gen_llm.complete.assert_called_once()
        review_llm.complete.assert_called_once()

    async def test_retries_then_accepts(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(
            side_effect=[
                _make_response("41"),
                _make_response("42"),
            ]
        )

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(
            side_effect=[
                _make_response("RETRY: check your arithmetic"),
                _make_response("ACCEPT"),
            ]
        )

        flow = generate_review_flow(gen_llm, review_llm, max_cycles=3)
        state = State(operational=generate_review_initial_state("What is 6*7?"))
        await flow.run(state)

        assert state.get("accepted") is True
        assert state.get("answer") == "42"
        assert gen_llm.complete.call_count == 2

    async def test_feedback_injected_into_retry(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(
            side_effect=[
                _make_response("wrong"),
                _make_response("right"),
            ]
        )

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(
            side_effect=[
                _make_response("RETRY: missing justification"),
                _make_response("ACCEPT"),
            ]
        )

        flow = generate_review_flow(gen_llm, review_llm, max_cycles=3)
        state = State(operational=generate_review_initial_state("test"))
        await flow.run(state)

        second_call = gen_llm.complete.call_args_list[1]
        system_arg = second_call[1].get("system", "")
        assert "missing justification" in system_arg

    async def test_max_cycles_exhausted(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(return_value=_make_response("bad answer"))

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(return_value=_make_response("RETRY: wrong"))

        flow = generate_review_flow(gen_llm, review_llm, max_cycles=2)
        state = State(operational=generate_review_initial_state("test"))
        await flow.run(state)

        assert state.get("accepted") is not True
        # Fallback answer should still be available
        assert state.get("answer") == "bad answer"

    async def test_unacceptable_not_parsed_as_accept(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(return_value=_make_response("answer"))

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(
            side_effect=[
                _make_response("UNACCEPTABLE: completely wrong"),
                _make_response("ACCEPT"),
            ]
        )

        flow = generate_review_flow(gen_llm, review_llm, max_cycles=3)
        state = State(operational=generate_review_initial_state("test"))
        await flow.run(state)

        # Should have retried — "unacceptable" must not match as "accept"
        assert gen_llm.complete.call_count == 2
        assert state.get("accepted") is True

    async def test_with_review_tools(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(return_value=_make_response("42"))

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(return_value=_make_response("ACCEPT"))

        review_tools = ToolGroup()

        flow = generate_review_flow(gen_llm, review_llm, review_tools=review_tools, max_cycles=3)
        state = State(operational=generate_review_initial_state("What is 6*7?"))
        await flow.run(state)

        assert state.get("accepted") is True

    async def test_gen_kwargs_passed(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(return_value=_make_response("answer"))

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(return_value=_make_response("ACCEPT"))

        flow = generate_review_flow(gen_llm, review_llm, gen_kwargs={"temperature": 0.9})
        state = State(operational=generate_review_initial_state("test"))
        await flow.run(state)

        call_kwargs = gen_llm.complete.call_args[1]
        assert call_kwargs.get("temperature") == 0.9

    async def test_review_kwargs_passed(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(return_value=_make_response("answer"))

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(return_value=_make_response("ACCEPT"))

        flow = generate_review_flow(gen_llm, review_llm, review_kwargs={"temperature": 0.1})
        state = State(operational=generate_review_initial_state("test"))
        await flow.run(state)

        call_kwargs = review_llm.complete.call_args[1]
        assert call_kwargs.get("temperature") == 0.1

    async def test_flow_name(self) -> None:
        gen_llm = AsyncMock()
        gen_llm.complete = AsyncMock(return_value=_make_response("answer"))

        review_llm = AsyncMock()
        review_llm.complete = AsyncMock(return_value=_make_response("ACCEPT"))

        flow = generate_review_flow(gen_llm, review_llm)
        state = State(operational=generate_review_initial_state("test"))
        result = await flow.run(state)

        assert result.trace.flow_name == "generate_review"


class TestGenerateReviewInitialState:
    def test_creates_initial_state(self) -> None:
        init = generate_review_initial_state("Hello")
        assert init["task"] == "Hello"
        assert init["feedback"] == []
        assert init["accepted"] is False

    def test_multimodal_content(self) -> None:
        init = generate_review_initial_state(["text", "image"])
        assert isinstance(init["task"], str)
        assert init["accepted"] is False
