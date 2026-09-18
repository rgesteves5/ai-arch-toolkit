"""``run_react``: a ReAct loop inside a step, its answer, its response, and whether it failed."""

from __future__ import annotations

from unittest.mock import AsyncMock

from ai_arch_toolkit.core._response import Response
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import ReactRun, run_react


def _llm(*answers: Response | Exception) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=list(answers))
    return llm


async def test_the_loop_answers_with_its_last_response() -> None:
    response = Response(text="Paris")

    run = await run_react(
        _llm(response), ToolGroup(), "capital?", system="", max_iterations=2, llm_kwargs=None
    )

    assert run == ReactRun(answer="Paris", response=response, failed=False)


async def test_a_step_that_ended_in_error_marks_the_run_failed() -> None:
    run = await run_react(
        _llm(RuntimeError("down")),
        ToolGroup(),
        "capital?",
        system="",
        max_iterations=2,
        llm_kwargs=None,
    )

    assert run == ReactRun(answer="", response=None, failed=True)


async def test_the_system_prompt_and_llm_kwargs_reach_the_model() -> None:
    llm = _llm(Response(text="ok"))

    await run_react(
        llm, ToolGroup(), "task", system="Be brief.", max_iterations=1, llm_kwargs={"top_k": 3}
    )

    kwargs = llm.complete.await_args.kwargs
    assert (kwargs["system"], kwargs["top_k"]) == ("Be brief.", 3)
