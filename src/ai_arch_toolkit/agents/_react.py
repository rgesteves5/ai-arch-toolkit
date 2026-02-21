"""ReAct (Reason + Act) agent implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    ReActResult,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.llm._types import Message, ToolCall, ToolResult, Usage


def _execute_tool(tools: Any, tc: ToolCall) -> tuple[str, str | None]:
    """Execute a single tool call, returning (result, error_or_none)."""
    try:
        result_str = tools.execute(tc)
        return result_str, None
    except Exception as exc:
        err = f"Error executing tool '{tc.name}': {exc}"
        return err, str(exc)


async def _async_execute_tool(tools: Any, tc: ToolCall) -> tuple[str, str | None]:
    """Execute a single tool call async, returning (result, error_or_none)."""
    try:
        result_str = await tools.async_execute(tc)
        return result_str, None
    except Exception as exc:
        err = f"Error executing tool '{tc.name}': {exc}"
        return err, str(exc)


class ReActAgent(BaseAgent):
    """ReAct agent that interleaves reasoning and tool use.

    Implements the Thought -> Action -> Observation loop:
    1. Send messages to the LLM (with tool definitions).
    2. If the LLM returns tool calls, execute them and feed results back.
    3. Repeat until the LLM responds without tool calls or max iterations.
    """

    def _chat_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Build kwargs for client.chat, injecting tool_choice if set."""
        if self.config.tool_choice is not None:
            kwargs.setdefault("tool_choice", self.config.tool_choice)
        return kwargs

    def _execute_tools_sync(
        self, tool_calls: tuple[ToolCall, ...], step_num: int
    ) -> list[ToolResult]:
        """Execute tool calls, optionally in parallel."""
        # Fire tool_call events BEFORE dispatch
        for tc in tool_calls:
            _fire_event(
                self.config,
                "tool_call",
                step_number=step_num,
                tool_name=tc.name,
                tool_args=dict(tc.arguments),
            )

        parallel = self.config.parallel_tool_execution and len(tool_calls) > 1
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_execute_tool, self.tools, tc) for tc in tool_calls]
                outcomes = [f.result() for f in futures]
        else:
            outcomes = [_execute_tool(self.tools, tc) for tc in tool_calls]

        # Fire result/error events AFTER dispatch (from main thread)
        results: list[ToolResult] = []
        for tc, (result_str, error) in zip(tool_calls, outcomes, strict=True):
            if error:
                _fire_event(
                    self.config,
                    "error",
                    step_number=step_num,
                    tool_name=tc.name,
                    error=error,
                )
            _fire_event(
                self.config,
                "tool_result",
                step_number=step_num,
                tool_name=tc.name,
                result=result_str,
            )
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=result_str,
                )
            )
        return results

    async def _execute_tools_async(
        self, tool_calls: tuple[ToolCall, ...], step_num: int
    ) -> list[ToolResult]:
        """Execute tool calls async, optionally in parallel."""
        # Fire tool_call events BEFORE dispatch
        for tc in tool_calls:
            _fire_event(
                self.config,
                "tool_call",
                step_number=step_num,
                tool_name=tc.name,
                tool_args=dict(tc.arguments),
            )

        parallel = self.config.parallel_tool_execution and len(tool_calls) > 1
        if parallel:
            outcomes = await asyncio.gather(
                *[_async_execute_tool(self.tools, tc) for tc in tool_calls]
            )
        else:
            outcomes = [await _async_execute_tool(self.tools, tc) for tc in tool_calls]

        # Fire result/error events AFTER dispatch
        results: list[ToolResult] = []
        for tc, (result_str, error) in zip(tool_calls, outcomes, strict=True):
            if error:
                _fire_event(
                    self.config,
                    "error",
                    step_number=step_num,
                    tool_name=tc.name,
                    error=error,
                )
            _fire_event(
                self.config,
                "tool_result",
                step_number=step_num,
                tool_name=tc.name,
                result=result_str,
            )
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=result_str,
                )
            )
        return results

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the ReAct loop on the given task."""
        stream = kwargs.pop("stream", False)
        cancellation_token = self._resolve_cancellation_token(
            kwargs.pop("cancellation_token", None)
        )
        if stream:
            return self.run_stream(
                task,
                cancellation_token=cancellation_token,
                **kwargs,
            )
        messages: list[Message | ToolResult] = [Message(role="user", content=task)]
        steps: list[AgentStep] = []
        total_usage = Usage()
        budget = self._new_budget_manager()
        start = time.monotonic()
        chat_kwargs = self._chat_kwargs(**kwargs)

        for step_num in range(1, self.config.max_iterations + 1):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReActResult(
                        answer="[cancelled]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReActResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    ReActResult(
                        answer="[timeout exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=ReActResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    ReActResult(
                        answer="[token budget exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReActResult,
                )

            _fire_event(self.config, "step_start", step_number=step_num)

            response = self.client.chat(
                messages,
                system=self.config.system or None,
                tools=self.tools.definitions() or None,
                **chat_kwargs,
            )
            total_usage = _accumulate_usage(total_usage, response.usage)
            step_cost = self._observe_response(budget, response, step_number=step_num)

            if budget.exhausted_reason() is not None:
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        response=response,
                        usage=response.usage,
                        cost_usd=step_cost,
                    )
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return self._finalize_result(
                    ReActResult(
                        answer=response.text or "[token budget exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReActResult,
                )

            if not response.tool_calls:
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        response=response,
                        usage=response.usage,
                        cost_usd=step_cost,
                    )
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return self._finalize_result(
                    ReActResult(
                        answer=response.text,
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="completed",
                    ),
                    result_type=ReActResult,
                )

            if budget.exhausted_reason() is not None:
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        response=response,
                        usage=response.usage,
                        cost_usd=step_cost,
                    )
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return self._finalize_result(
                    ReActResult(
                        answer=response.text or "[token budget exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReActResult,
                )

            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReActResult(
                        answer="[cancelled]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReActResult,
                )
            tool_results = self._execute_tools_sync(response.tool_calls, step_num)

            steps.append(
                AgentStep(
                    step_number=step_num,
                    response=response,
                    tool_calls=response.tool_calls,
                    tool_results=tuple(tool_results),
                    usage=response.usage,
                    cost_usd=step_cost,
                )
            )
            _fire_event(self.config, "step_end", step_number=step_num)

            messages.append(response.to_message())
            messages.extend(tool_results)

        last_text = steps[-1].response.text if steps else ""
        return self._finalize_result(
            ReActResult(
                answer=last_text or "[max iterations reached]",
                steps=tuple(steps),
                total_usage=total_usage,
                stop_reason="max_iterations",
            ),
            result_type=ReActResult,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the ReAct loop asynchronously on the given task."""
        stream = kwargs.pop("stream", False)
        cancellation_token = self._resolve_cancellation_token(
            kwargs.pop("cancellation_token", None)
        )
        if stream:
            return self.async_run_stream(
                task,
                cancellation_token=cancellation_token,
                **kwargs,
            )
        messages: list[Message | ToolResult] = [Message(role="user", content=task)]
        steps: list[AgentStep] = []
        total_usage = Usage()
        budget = self._new_budget_manager()
        start = time.monotonic()
        chat_kwargs = self._chat_kwargs(**kwargs)

        for step_num in range(1, self.config.max_iterations + 1):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReActResult(
                        answer="[cancelled]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReActResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    ReActResult(
                        answer="[timeout exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=ReActResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    ReActResult(
                        answer="[token budget exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReActResult,
                )

            _fire_event(self.config, "step_start", step_number=step_num)

            response = await self.client.chat(
                messages,
                system=self.config.system or None,
                tools=self.tools.definitions() or None,
                **chat_kwargs,
            )
            total_usage = _accumulate_usage(total_usage, response.usage)
            step_cost = self._observe_response(budget, response, step_number=step_num)

            if budget.exhausted_reason() is not None:
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        response=response,
                        usage=response.usage,
                        cost_usd=step_cost,
                    )
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return self._finalize_result(
                    ReActResult(
                        answer=response.text or "[token budget exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReActResult,
                )

            if not response.tool_calls:
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        response=response,
                        usage=response.usage,
                        cost_usd=step_cost,
                    )
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return self._finalize_result(
                    ReActResult(
                        answer=response.text,
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="completed",
                    ),
                    result_type=ReActResult,
                )

            if budget.exhausted_reason() is not None:
                steps.append(
                    AgentStep(
                        step_number=step_num,
                        response=response,
                        usage=response.usage,
                        cost_usd=step_cost,
                    )
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return self._finalize_result(
                    ReActResult(
                        answer=response.text or "[token budget exceeded]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=ReActResult,
                )

            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    ReActResult(
                        answer="[cancelled]",
                        steps=tuple(steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=ReActResult,
                )
            tool_results = await self._execute_tools_async(response.tool_calls, step_num)

            steps.append(
                AgentStep(
                    step_number=step_num,
                    response=response,
                    tool_calls=response.tool_calls,
                    tool_results=tuple(tool_results),
                    usage=response.usage,
                    cost_usd=step_cost,
                )
            )
            _fire_event(self.config, "step_end", step_number=step_num)

            messages.append(response.to_message())
            messages.extend(tool_results)

        last_text = steps[-1].response.text if steps else ""
        return self._finalize_result(
            ReActResult(
                answer=last_text or "[max iterations reached]",
                steps=tuple(steps),
                total_usage=total_usage,
                stop_reason="max_iterations",
            ),
            result_type=ReActResult,
        )

    def run_stream(self, task: str, **kwargs: Any) -> Iterator[AgentStep]:
        """Run the ReAct loop, yielding each step as it completes."""
        kwargs.pop("stream", None)
        cancellation_token = self._resolve_cancellation_token(
            kwargs.pop("cancellation_token", None)
        )
        messages: list[Message | ToolResult] = [Message(role="user", content=task)]
        total_usage = Usage()
        budget = self._new_budget_manager()
        start = time.monotonic()
        chat_kwargs = self._chat_kwargs(**kwargs)

        for step_num in range(1, self.config.max_iterations + 1):
            if self._is_cancelled(cancellation_token):
                return
            if self._check_timeout(start):
                return
            if budget.exhausted_reason() is not None:
                return

            _fire_event(self.config, "step_start", step_number=step_num)

            response = self.client.chat(
                messages,
                system=self.config.system or None,
                tools=self.tools.definitions() or None,
                **chat_kwargs,
            )
            total_usage = _accumulate_usage(total_usage, response.usage)
            step_cost = self._observe_response(budget, response, step_number=step_num)

            if budget.exhausted_reason() is not None:
                yield AgentStep(
                    step_number=step_num,
                    response=response,
                    usage=response.usage,
                    cost_usd=step_cost,
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return

            if not response.tool_calls:
                yield AgentStep(
                    step_number=step_num,
                    response=response,
                    usage=response.usage,
                    cost_usd=step_cost,
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return

            if budget.exhausted_reason() is not None:
                yield AgentStep(
                    step_number=step_num,
                    response=response,
                    usage=response.usage,
                    cost_usd=step_cost,
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return

            if self._is_cancelled(cancellation_token):
                return
            tool_results = self._execute_tools_sync(response.tool_calls, step_num)

            yield AgentStep(
                step_number=step_num,
                response=response,
                tool_calls=response.tool_calls,
                tool_results=tuple(tool_results),
                usage=response.usage,
                cost_usd=step_cost,
            )
            _fire_event(self.config, "step_end", step_number=step_num)

            messages.append(response.to_message())
            messages.extend(tool_results)

    async def async_run_stream(self, task: str, **kwargs: Any) -> AsyncIterator[AgentStep]:
        """Run the ReAct loop asynchronously, yielding each step."""
        kwargs.pop("stream", None)
        cancellation_token = self._resolve_cancellation_token(
            kwargs.pop("cancellation_token", None)
        )
        messages: list[Message | ToolResult] = [Message(role="user", content=task)]
        total_usage = Usage()
        budget = self._new_budget_manager()
        start = time.monotonic()
        chat_kwargs = self._chat_kwargs(**kwargs)

        for step_num in range(1, self.config.max_iterations + 1):
            if self._is_cancelled(cancellation_token):
                return
            if self._check_timeout(start):
                return
            if budget.exhausted_reason() is not None:
                return

            _fire_event(self.config, "step_start", step_number=step_num)

            response = await self.client.chat(
                messages,
                system=self.config.system or None,
                tools=self.tools.definitions() or None,
                **chat_kwargs,
            )
            total_usage = _accumulate_usage(total_usage, response.usage)
            step_cost = self._observe_response(budget, response, step_number=step_num)

            if budget.exhausted_reason() is not None:
                yield AgentStep(
                    step_number=step_num,
                    response=response,
                    usage=response.usage,
                    cost_usd=step_cost,
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return

            if not response.tool_calls:
                yield AgentStep(
                    step_number=step_num,
                    response=response,
                    usage=response.usage,
                    cost_usd=step_cost,
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return

            if budget.exhausted_reason() is not None:
                yield AgentStep(
                    step_number=step_num,
                    response=response,
                    usage=response.usage,
                    cost_usd=step_cost,
                )
                _fire_event(self.config, "step_end", step_number=step_num)
                return

            if self._is_cancelled(cancellation_token):
                return
            tool_results = await self._execute_tools_async(response.tool_calls, step_num)

            yield AgentStep(
                step_number=step_num,
                response=response,
                tool_calls=response.tool_calls,
                tool_results=tuple(tool_results),
                usage=response.usage,
                cost_usd=step_cost,
            )
            _fire_event(self.config, "step_end", step_number=step_num)

            messages.append(response.to_message())
            messages.extend(tool_results)
