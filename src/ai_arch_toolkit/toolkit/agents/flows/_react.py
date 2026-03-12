"""ReAct as a Flow — cyclic LLM call + tool execution loop."""

from __future__ import annotations

import asyncio
from typing import Any

from ai_arch_toolkit.core._content import Content, tool_result, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._response import Usage
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep


def react_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    system: str = "",
    max_iterations: int = 10,
    parallel_tool_calls: bool = True,
    timeout: float | None = None,
    policy: Policy | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    final_answer_hint: bool = True,
    strip_tools_on_final: bool = False,
    show_turn_counter: bool = False,
) -> Flow:
    """Create a ReAct Flow — cyclic LLM reasoning + tool execution.

    Args:
        llm: Language model to use.
        tools: Tool group for the agent.
        system: System prompt.
        max_iterations: Maximum reasoning iterations.
        parallel_tool_calls: Whether to execute tool calls in parallel.
        timeout: Overall timeout in seconds.
        policy: Optional execution policy for the flow.
        llm_kwargs: Additional kwargs passed to llm.complete().
        final_answer_hint: On the last turn, inject a message asking the model
            to provide a final text answer without calling tools. Fixes models
            that always emit tool calls and never produce a text response.
        strip_tools_on_final: On the last turn, pass an empty ToolGroup so the
            model physically cannot call tools. More aggressive than hint alone.
        show_turn_counter: Inject a ``[Turn N/M]`` user message each turn for
            debugging and transparency.
    """
    extra_kwargs = llm_kwargs or {}

    async def llm_call(snap: StateSnapshot) -> Result:
        """Call LLM with current messages and tools."""
        messages: list[dict[str, Any]] = snap.require("messages")
        total_usage: Usage = snap.get("total_usage", Usage())
        turn: int = snap.get("turn", 0) + 1
        is_final = turn >= max_iterations

        call_messages = list(messages)

        if show_turn_counter and not is_final:
            call_messages.append(user(f"[Turn {turn}/{max_iterations}]"))

        if is_final and final_answer_hint:
            label = f"[Turn {turn}/{max_iterations} — FINAL] " if show_turn_counter else ""
            call_messages.append(
                user(
                    f"{label}This is your last turn. Provide your final answer "
                    "as text. Do not call any tools."
                )
            )

        call_tools = ToolGroup() if (is_final and strip_tools_on_final) else tools

        try:
            response = await llm.complete(
                call_messages,
                system=system or None,
                tools=call_tools,
                **extra_kwargs,
            )
        except Exception as exc:
            return Result(error=str(exc))

        new_usage = Usage(
            input_tokens=total_usage.input_tokens + response.usage.input_tokens,
            output_tokens=total_usage.output_tokens + response.usage.output_tokens,
            cache_write_tokens=(
                total_usage.cache_write_tokens + response.usage.cache_write_tokens
            ),
            cache_read_tokens=(total_usage.cache_read_tokens + response.usage.cache_read_tokens),
        )

        return Result(
            value=response,
            artifacts={
                "response": response,
                "has_tool_calls": response.has_tool_calls,
                "needs_llm_call": False,
                "total_usage": new_usage,
                "turn": turn,
            },
            usage=response.usage,
            cost=response.cost or 0.0,
        )

    async def execute_tools(snap: StateSnapshot) -> Result:
        """Execute tool calls from the LLM response."""
        response = snap.require("response")
        messages: list[dict[str, Any]] = snap.require("messages")

        tool_result_dicts: list[dict[str, Any]] = []

        if parallel_tool_calls and len(response.tool_calls) > 1:

            async def _safe_execute(tc: Any) -> str:
                try:
                    return await tools.async_execute(tc)
                except Exception as exc:
                    return f"Error: {exc}"

            results = await asyncio.gather(*[_safe_execute(tc) for tc in response.tool_calls])
            for tc, result_str in zip(response.tool_calls, results, strict=True):
                tool_result_dicts.append(tool_result(result_str, tool_use_id=tc.id, name=tc.name))
        else:
            for tc in response.tool_calls:
                try:
                    result_str = await tools.async_execute(tc)
                except Exception as exc:
                    result_str = f"Error: {exc}"
                tool_result_dicts.append(tool_result(result_str, tool_use_id=tc.id, name=tc.name))

        updated_messages = [*messages, response.to_message(), *tool_result_dicts]

        return Result(
            value=tool_result_dicts,
            artifacts={
                "messages": updated_messages,
                "has_tool_calls": False,
                "needs_llm_call": True,
            },
        )

    def needs_llm(snap: StateSnapshot) -> bool:
        return snap.get("needs_llm_call", True)

    def has_tool_calls(snap: StateSnapshot) -> bool:
        return snap.get("has_tool_calls", False)

    flow_policy = policy
    if timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=timeout)

    return Flow(
        FlowStep(step=Step(name="llm_call", fn=llm_call), when=needs_llm),
        FlowStep(step=Step(name="execute_tools", fn=execute_tools), when=has_tool_calls),
        name="react",
        policy=flow_policy,
        max_iterations=max_iterations,
    )


def react_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a react_flow.

    Returns:
        Dict suitable for State(operational=...).
    """
    return {
        "messages": [user(task)],
        "has_tool_calls": False,
        "total_usage": Usage(),
    }
