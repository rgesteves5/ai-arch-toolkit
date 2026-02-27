"""ReActAgent — Thought → Action → Observation loop."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.agents._base import AgentEvent, BaseAgent, _add_usage
from ai_arch_toolkit.core._content import Content, tool_result, user
from ai_arch_toolkit.core._response import Usage


class ReActAgent(BaseAgent):
    """ReAct: iterative reasoning with tool use.

    Each iteration:
    1. Send conversation to LLM (with tool definitions).
    2. If the LLM returns tool calls, execute them and feed results back.
    3. Repeat until no tool calls remain or a stop condition is met.
    """

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        messages: list[dict[str, Any]] = [user(task)]
        total_usage = Usage()
        start = time.monotonic()

        for step_num in range(1, self.config.max_iterations + 1):
            # --- stop conditions ---
            if self._check_timeout(start):
                yield AgentEvent(type="step_end", step=step_num, stop_reason="timeout")
                return
            if self._check_budget(total_usage):
                yield AgentEvent(type="step_end", step=step_num, stop_reason="budget_exhausted")
                return

            # step_start
            yield AgentEvent(type="step_start", step=step_num)

            # LLM call
            try:
                response = await self.llm.complete(
                    messages,
                    system=self.config.system or None,
                    tools=self.tools,
                    tool_choice=self.config.tool_choice,
                )
            except Exception as exc:
                yield AgentEvent(type="error", step=step_num, error=str(exc))
                yield AgentEvent(type="step_end", step=step_num, stop_reason="error")
                return

            total_usage = _add_usage(total_usage, response.usage)

            # No tool calls → final answer
            if not response.has_tool_calls:
                yield AgentEvent(
                    type="step_end",
                    step=step_num,
                    response=response,
                    stop_reason="completed",
                )
                return

            # --- Execute tool calls ---
            tool_result_dicts: list[dict[str, Any]] = []

            if self.config.parallel_tool_calls and len(response.tool_calls) > 1:
                results = await self._execute_tools_parallel(response.tool_calls, step_num)
                for tc, (result_str, events) in zip(response.tool_calls, results, strict=True):
                    for evt in events:
                        yield evt
                    tool_result_dicts.append(
                        tool_result(result_str, tool_use_id=tc.id, name=tc.name)
                    )
            else:
                for tc in response.tool_calls:
                    result_str, events = await self._execute_one_tool(tc, step_num)
                    for evt in events:
                        yield evt
                    tool_result_dicts.append(
                        tool_result(result_str, tool_use_id=tc.id, name=tc.name)
                    )

            # step_end for this iteration (no stop_reason — run is still going)
            yield AgentEvent(type="step_end", step=step_num, response=response)

            # Update conversation
            messages.append(response.to_message())
            messages.extend(tool_result_dicts)

        # Max iterations exhausted
        yield AgentEvent(
            type="step_end",
            step=self.config.max_iterations,
            stop_reason="max_iterations",
        )

    async def _execute_one_tool(self, tc: Any, step_num: int) -> tuple[str, list[AgentEvent]]:
        """Execute a single tool call. Returns (result_str, events)."""
        events: list[AgentEvent] = []
        events.append(
            AgentEvent(
                type="tool_call",
                step=step_num,
                tool_name=tc.name,
                tool_call_id=tc.id,
                tool_args=dict(tc.input),
            )
        )

        try:
            result_str = await self.tools.async_execute(tc)
        except Exception as exc:
            events.append(
                AgentEvent(
                    type="error",
                    step=step_num,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    error=str(exc),
                )
            )
            result_str = f"Error: {exc}"
        else:
            events.append(
                AgentEvent(
                    type="tool_result",
                    step=step_num,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    result=result_str,
                )
            )

        return result_str, events

    async def _execute_tools_parallel(
        self, tool_calls: tuple[Any, ...], step_num: int
    ) -> list[tuple[str, list[AgentEvent]]]:
        """Execute multiple tool calls in parallel."""
        tasks = [self._execute_one_tool(tc, step_num) for tc in tool_calls]
        return list(await asyncio.gather(*tasks))
