"""LLMCompiler agent — DAG-based parallel tool execution."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    LLMCompilerResult,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.llm._types import Message, Response, ToolCall, ToolResult, Usage

_PLAN_PROMPT = (
    "You are a planner that creates a DAG of tool calls. "
    "Return a JSON array where each element has:\n"
    '  {{"id": "1", "tool": "tool_name", "args": {{}}, '
    '"deps": []}}\n'
    "Use $N in args values to reference the output of task N.\n"
    "Available tools: {tools}\n\nTask: {task}"
)

_JOIN_PROMPT = (
    "Given the task and the results of all executed tool calls, "
    "synthesize a final answer.\n\n"
    "Task: {task}\n\n"
    "Results:\n{results}"
)

_REPLAN_CHECK_PROMPT = (
    "You are evaluating whether tool call results are sufficient to "
    "answer the task. If the results are sufficient, respond with "
    "FINISH. If more tool calls are needed, respond with REPLAN and "
    "explain what additional information is needed.\n\n"
    "Task: {task}\n\nResults:\n{results}"
)

_REPLAN_PROMPT = (
    "You are a planner that creates a DAG of tool calls. "
    "Some tasks have already been completed. Plan ONLY the additional "
    "tool calls needed.\n"
    "Return a JSON array where each element has:\n"
    '  {{"id": "1", "tool": "tool_name", "args": {{}}, '
    '"deps": []}}\n'
    "Use $N in args values to reference the output of task N.\n"
    "Available tools: {tools}\n\n"
    "Task: {task}\n\n"
    "Already completed:\n{completed}"
)


@dataclass
class _DAGTask:
    id: str
    tool_name: str
    tool_args: dict[str, str]
    deps: list[str]
    result: str = ""
    status: str = "pending"  # "pending" | "completed" | "failed"


class LLMCompilerAgent(BaseAgent):
    """LLMCompiler agent that plans and executes a DAG of tool calls.

    1. Planner: LLM generates a DAG as JSON.
    2. Scheduler: Finds ready tasks (all deps satisfied).
    3. Executor: Runs ready tasks in parallel.
    4. Repeat until all tasks done.
    5. Joiner: Evaluates sufficiency; re-plans if needed.
    6. Joiner: LLM synthesizes final answer.
    """

    def _build_plan_prompt(self, task: str) -> str:
        tool_names = ", ".join(t.name for t in self.tools.definitions())
        return _PLAN_PROMPT.format(tools=tool_names, task=task)

    def _build_join_prompt(self, task: str, dag: list[_DAGTask]) -> str:
        results_text = "\n".join(f"Task {dt.id} ({dt.tool_name}): {dt.result}" for dt in dag)
        return _JOIN_PROMPT.format(task=task, results=results_text)

    def _build_results_text(self, results_map: dict[str, str]) -> str:
        return "\n".join(f"Task {k}: {v}" for k, v in results_map.items())

    def _build_replan_prompt(self, task: str, results_map: dict[str, str]) -> str:
        tool_names = ", ".join(t.name for t in self.tools.definitions())
        completed = "\n".join(f"Task {k}: {v}" for k, v in results_map.items())
        return _REPLAN_PROMPT.format(tools=tool_names, task=task, completed=completed)

    # ------------------------------------------------------------------
    # DAG execution helpers (shared by initial plan & re-plans)
    # ------------------------------------------------------------------

    def _execute_dag_sync(
        self,
        dag: list[_DAGTask],
        step_num: int,
        results_map: dict[str, str],
        tool_results: list[ToolResult],
        start: float,
        total_usage: Usage,
        cancellation_token: object | None,
    ) -> tuple[int, Usage, bool, bool]:
        """Execute a DAG synchronously.

        Returns ``(next_step_num, usage, timed_out, cancelled)``.
        """
        while True:
            if self._is_cancelled(cancellation_token):
                return step_num, total_usage, False, True
            if self._check_timeout(start):
                return step_num, total_usage, True, False
            ready = _get_ready_tasks(dag, results_map)
            if not ready:
                break

            _fire_event(self.config, "step_start", step_number=step_num)

            for dt in ready:
                resolved = _substitute_args(dt.tool_args, results_map)
                _fire_event(
                    self.config,
                    "tool_call",
                    step_number=step_num,
                    tool_name=dt.tool_name,
                    tool_args=resolved,
                )

            if self.config.parallel_tool_execution and len(ready) > 1:
                if self._is_cancelled(cancellation_token):
                    return step_num, total_usage, False, True
                with ThreadPoolExecutor() as executor:
                    futures = {
                        dt.id: executor.submit(
                            self._exec_task,
                            dt,
                            results_map,
                        )
                        for dt in ready
                    }
                    for dt in ready:
                        if self._is_cancelled(cancellation_token):
                            return step_num, total_usage, False, True
                        dt.result = futures[dt.id].result()
                        if dt.status != "failed":
                            dt.status = "completed"
                        results_map[dt.id] = dt.result
            else:
                for dt in ready:
                    if self._is_cancelled(cancellation_token):
                        return step_num, total_usage, False, True
                    dt.result = self._exec_task(dt, results_map)
                    if dt.status != "failed":
                        dt.status = "completed"
                    results_map[dt.id] = dt.result

            for dt in ready:
                _fire_event(
                    self.config,
                    "tool_result",
                    step_number=step_num,
                    tool_name=dt.tool_name,
                    result=dt.result,
                )

            for dt in ready:
                tool_results.append(
                    ToolResult(
                        tool_call_id=dt.id,
                        name=dt.tool_name,
                        content=dt.result,
                    )
                )

            _fire_event(self.config, "step_end", step_number=step_num)
            step_num += 1

        return step_num, total_usage, False, False

    async def _execute_dag_async(
        self,
        dag: list[_DAGTask],
        step_num: int,
        results_map: dict[str, str],
        tool_results: list[ToolResult],
        start: float,
        total_usage: Usage,
        cancellation_token: object | None,
    ) -> tuple[int, Usage, bool, bool]:
        """Execute a DAG asynchronously.

        Returns ``(next_step_num, usage, timed_out, cancelled)``.
        """
        while True:
            if self._is_cancelled(cancellation_token):
                return step_num, total_usage, False, True
            if self._check_timeout(start):
                return step_num, total_usage, True, False
            ready = _get_ready_tasks(dag, results_map)
            if not ready:
                break

            _fire_event(self.config, "step_start", step_number=step_num)

            for dt in ready:
                resolved = _substitute_args(dt.tool_args, results_map)
                _fire_event(
                    self.config,
                    "tool_call",
                    step_number=step_num,
                    tool_name=dt.tool_name,
                    tool_args=resolved,
                )

            if self.config.parallel_tool_execution and len(ready) > 1:
                if self._is_cancelled(cancellation_token):
                    return step_num, total_usage, False, True
                outcomes = await asyncio.gather(
                    *[self._async_exec_task(dt, results_map) for dt in ready]
                )
                for dt, result in zip(ready, outcomes, strict=True):
                    if self._is_cancelled(cancellation_token):
                        return step_num, total_usage, False, True
                    dt.result = result
                    if dt.status != "failed":
                        dt.status = "completed"
                    results_map[dt.id] = dt.result
            else:
                for dt in ready:
                    if self._is_cancelled(cancellation_token):
                        return step_num, total_usage, False, True
                    dt.result = await self._async_exec_task(dt, results_map)
                    if dt.status != "failed":
                        dt.status = "completed"
                    results_map[dt.id] = dt.result

            for dt in ready:
                _fire_event(
                    self.config,
                    "tool_result",
                    step_number=step_num,
                    tool_name=dt.tool_name,
                    result=dt.result,
                )

            for dt in ready:
                tool_results.append(
                    ToolResult(
                        tool_call_id=dt.id,
                        name=dt.tool_name,
                        content=dt.result,
                    )
                )

            _fire_event(self.config, "step_end", step_number=step_num)
            step_num += 1

        return step_num, total_usage, False, False

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the LLMCompiler loop.

        Args:
            task: The task to solve.
            **kwargs: Extra arguments forwarded to the LLM client.
                ``max_replans`` (int): Maximum joiner re-plan iterations
                (default ``0`` for backward compatibility).
        """
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
        max_replans: int = kwargs.pop("max_replans", 0)
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        all_steps: list[AgentStep] = []
        start = time.monotonic()

        def _record_repair_response(step_number: int, response: Response) -> None:
            nonlocal total_usage
            total_usage = _accumulate_usage(total_usage, response.usage)
            repair_cost = self._observe_response(budget, response, step_number=step_number)
            all_steps.append(
                AgentStep(
                    step_number=step_number,
                    response=response,
                    usage=response.usage,
                    cost_usd=repair_cost,
                    metadata={"repair_attempt": True},
                )
            )

        # Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = self.client.chat(
            [Message(role="user", content=self._build_plan_prompt(task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        plan_cost = self._observe_response(budget, plan_resp, step_number=1)
        all_steps.append(
            AgentStep(step_number=1, response=plan_resp, usage=plan_resp.usage, cost_usd=plan_cost)
        )
        dag = self._parse_dag_with_fallback(
            plan_resp.text,
            system=system,
            step_number=1,
            on_repair_response=_record_repair_response,
            **kwargs,
        )
        _fire_event(
            self.config,
            "plan_created",
            step_number=1,
            result=plan_resp.text,
        )
        _fire_event(self.config, "step_end", step_number=1)

        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[token budget exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=LLMCompilerResult,
            )

        # Execute DAG
        step_num = 2
        results_map: dict[str, str] = {}
        tool_results: list[ToolResult] = []

        step_num, total_usage, timed_out, cancelled = self._execute_dag_sync(
            dag,
            step_num,
            results_map,
            tool_results,
            start,
            total_usage,
            cancellation_token,
        )
        if cancelled:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[cancelled]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=LLMCompilerResult,
            )
        if timed_out:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=LLMCompilerResult,
            )

        # Re-plan loop
        for _replan_num in range(max_replans):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=LLMCompilerResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=LLMCompilerResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[token budget exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=LLMCompilerResult,
                )
            results_text = self._build_results_text(results_map)
            check_resp = self.client.chat(
                [
                    Message(
                        role="user",
                        content=_REPLAN_CHECK_PROMPT.format(
                            task=task,
                            results=results_text,
                        ),
                    )
                ],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, check_resp.usage)
            self._observe_response(budget, check_resp, step_number=step_num)

            if "FINISH" in check_resp.text.upper():
                break

            # Re-plan: generate new DAG with context
            _fire_event(
                self.config,
                "plan_created",
                step_number=step_num,
                result=check_resp.text,
            )
            replan_resp = self.client.chat(
                [
                    Message(
                        role="user",
                        content=self._build_replan_prompt(task, results_map),
                    )
                ],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, replan_resp.usage)
            replan_cost = self._observe_response(budget, replan_resp, step_number=step_num)
            new_dag = self._parse_dag_with_fallback(
                replan_resp.text,
                system=system,
                step_number=step_num,
                on_repair_response=_record_repair_response,
                **kwargs,
            )
            all_steps.append(
                AgentStep(
                    step_number=step_num,
                    response=replan_resp,
                    usage=replan_resp.usage,
                    cost_usd=replan_cost,
                )
            )
            step_num += 1

            step_num, total_usage, timed_out, cancelled = self._execute_dag_sync(
                new_dag,
                step_num,
                results_map,
                tool_results,
                start,
                total_usage,
                cancellation_token,
            )
            if cancelled:
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=LLMCompilerResult,
                )
            if timed_out:
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=LLMCompilerResult,
                )
            dag.extend(new_dag)

        # Joiner
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[cancelled]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=LLMCompilerResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=LLMCompilerResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[token budget exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=LLMCompilerResult,
            )
        _fire_event(self.config, "step_start", step_number=step_num)
        join_resp = self.client.chat(
            [
                Message(
                    role="user",
                    content=self._build_join_prompt(task, dag),
                )
            ],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, join_resp.usage)
        join_cost = self._observe_response(budget, join_resp, step_number=step_num)
        all_steps.append(
            AgentStep(
                step_number=step_num,
                response=join_resp,
                tool_results=tuple(tool_results),
                usage=join_resp.usage,
                cost_usd=join_cost,
            )
        )
        _fire_event(self.config, "step_end", step_number=step_num)

        return self._finalize_result(
            LLMCompilerResult(
                answer=join_resp.text,
                steps=tuple(all_steps),
                total_usage=total_usage,
                stop_reason="completed",
                metadata={"dag_size": len(dag)},
            ),
            result_type=LLMCompilerResult,
        )

    def _exec_task(
        self,
        dt: _DAGTask,
        results_map: dict[str, str],
    ) -> str:
        """Execute a single DAG task (pure execution, no events)."""
        resolved_args = _substitute_args(dt.tool_args, results_map)
        tc = ToolCall(
            id=dt.id,
            name=dt.tool_name,
            arguments=resolved_args,
        )
        try:
            return self.tools.execute(tc)
        except Exception as exc:
            dt.status = "failed"
            return f"Error: {exc}"

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the LLMCompiler loop asynchronously.

        Args:
            task: The task to solve.
            **kwargs: Extra arguments forwarded to the LLM client.
                ``max_replans`` (int): Maximum joiner re-plan iterations
                (default ``0`` for backward compatibility).
        """
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
        max_replans: int = kwargs.pop("max_replans", 0)
        system = self.config.system or None
        total_usage = Usage()
        budget = self._new_budget_manager()
        all_steps: list[AgentStep] = []
        start = time.monotonic()

        def _record_repair_response(step_number: int, response: Response) -> None:
            nonlocal total_usage
            total_usage = _accumulate_usage(total_usage, response.usage)
            repair_cost = self._observe_response(budget, response, step_number=step_number)
            all_steps.append(
                AgentStep(
                    step_number=step_number,
                    response=response,
                    usage=response.usage,
                    cost_usd=repair_cost,
                    metadata={"repair_attempt": True},
                )
            )

        # Plan
        _fire_event(self.config, "step_start", step_number=1)
        plan_resp = await self.client.chat(
            [Message(role="user", content=self._build_plan_prompt(task))],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, plan_resp.usage)
        plan_cost = self._observe_response(budget, plan_resp, step_number=1)
        all_steps.append(
            AgentStep(step_number=1, response=plan_resp, usage=plan_resp.usage, cost_usd=plan_cost)
        )
        dag = await self._aparse_dag_with_fallback(
            plan_resp.text,
            system=system,
            step_number=1,
            on_repair_response=_record_repair_response,
            **kwargs,
        )
        _fire_event(
            self.config,
            "plan_created",
            step_number=1,
            result=plan_resp.text,
        )
        _fire_event(self.config, "step_end", step_number=1)

        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[token budget exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=LLMCompilerResult,
            )

        # Execute DAG
        step_num = 2
        results_map: dict[str, str] = {}
        tool_results: list[ToolResult] = []

        step_num, total_usage, timed_out, cancelled = await self._execute_dag_async(
            dag,
            step_num,
            results_map,
            tool_results,
            start,
            total_usage,
            cancellation_token,
        )
        if cancelled:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[cancelled]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=LLMCompilerResult,
            )
        if timed_out:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=LLMCompilerResult,
            )

        # Re-plan loop
        for _replan_num in range(max_replans):
            if self._is_cancelled(cancellation_token):
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=LLMCompilerResult,
                )
            if self._check_timeout(start):
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=LLMCompilerResult,
                )
            if budget.exhausted_reason() is not None:
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[token budget exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="budget_exhausted",
                    ),
                    result_type=LLMCompilerResult,
                )
            results_text = self._build_results_text(results_map)
            check_resp = await self.client.chat(
                [
                    Message(
                        role="user",
                        content=_REPLAN_CHECK_PROMPT.format(
                            task=task,
                            results=results_text,
                        ),
                    )
                ],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, check_resp.usage)
            self._observe_response(budget, check_resp, step_number=step_num)

            if "FINISH" in check_resp.text.upper():
                break

            # Re-plan: generate new DAG with context
            _fire_event(
                self.config,
                "plan_created",
                step_number=step_num,
                result=check_resp.text,
            )
            replan_resp = await self.client.chat(
                [
                    Message(
                        role="user",
                        content=self._build_replan_prompt(task, results_map),
                    )
                ],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, replan_resp.usage)
            replan_cost = self._observe_response(budget, replan_resp, step_number=step_num)
            new_dag = await self._aparse_dag_with_fallback(
                replan_resp.text,
                system=system,
                step_number=step_num,
                on_repair_response=_record_repair_response,
                **kwargs,
            )
            all_steps.append(
                AgentStep(
                    step_number=step_num,
                    response=replan_resp,
                    usage=replan_resp.usage,
                    cost_usd=replan_cost,
                )
            )
            step_num += 1

            step_num, total_usage, timed_out, cancelled = await self._execute_dag_async(
                new_dag,
                step_num,
                results_map,
                tool_results,
                start,
                total_usage,
                cancellation_token,
            )
            if cancelled:
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[cancelled]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="cancelled",
                    ),
                    result_type=LLMCompilerResult,
                )
            if timed_out:
                return self._finalize_result(
                    LLMCompilerResult(
                        answer="[timeout exceeded]",
                        steps=tuple(all_steps),
                        total_usage=total_usage,
                        stop_reason="timeout",
                    ),
                    result_type=LLMCompilerResult,
                )
            dag.extend(new_dag)

        # Joiner
        if self._is_cancelled(cancellation_token):
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[cancelled]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="cancelled",
                ),
                result_type=LLMCompilerResult,
            )
        if self._check_timeout(start):
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[timeout exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="timeout",
                ),
                result_type=LLMCompilerResult,
            )
        if budget.exhausted_reason() is not None:
            return self._finalize_result(
                LLMCompilerResult(
                    answer="[token budget exceeded]",
                    steps=tuple(all_steps),
                    total_usage=total_usage,
                    stop_reason="budget_exhausted",
                ),
                result_type=LLMCompilerResult,
            )
        _fire_event(self.config, "step_start", step_number=step_num)
        join_resp = await self.client.chat(
            [
                Message(
                    role="user",
                    content=self._build_join_prompt(task, dag),
                )
            ],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, join_resp.usage)
        join_cost = self._observe_response(budget, join_resp, step_number=step_num)
        all_steps.append(
            AgentStep(
                step_number=step_num,
                response=join_resp,
                tool_results=tuple(tool_results),
                usage=join_resp.usage,
                cost_usd=join_cost,
            )
        )
        _fire_event(self.config, "step_end", step_number=step_num)

        return self._finalize_result(
            LLMCompilerResult(
                answer=join_resp.text,
                steps=tuple(all_steps),
                total_usage=total_usage,
                stop_reason="completed",
                metadata={"dag_size": len(dag)},
            ),
            result_type=LLMCompilerResult,
        )

    async def _async_exec_task(
        self,
        dt: _DAGTask,
        results_map: dict[str, str],
    ) -> str:
        """Execute a single DAG task asynchronously (no events)."""
        resolved_args = _substitute_args(dt.tool_args, results_map)
        tc = ToolCall(
            id=dt.id,
            name=dt.tool_name,
            arguments=resolved_args,
        )
        try:
            return await self.tools.async_execute(tc)
        except Exception as exc:
            dt.status = "failed"
            return f"Error: {exc}"

    def _parse_dag_with_fallback(
        self,
        text: str,
        *,
        system: str | None,
        step_number: int,
        on_repair_response: Callable[[int, Response], None] | None = None,
        **kwargs: Any,
    ) -> list[_DAGTask]:
        dag = _parse_dag(text)
        if dag:
            return dag
        repaired = text
        for _ in range(max(0, self.config.planner_repair_retries)):
            repair_resp = self.client.chat(
                [
                    Message(
                        role="user",
                        content=(
                            "Repair the following into a valid JSON array of "
                            "{\"id\":\"1\",\"tool\":\"name\",\"args\":{},\"deps\":[]} objects. "
                            "Return only JSON.\n\n"
                            f"{repaired}"
                        ),
                    )
                ],
                system=system,
                **kwargs,
            )
            if on_repair_response is not None:
                on_repair_response(step_number, repair_resp)
            repaired = repair_resp.text
            dag = _parse_dag(repaired)
            if dag:
                _fire_event(
                    self.config,
                    "plan_created",
                    step_number=step_number,
                    result=repaired,
                    metadata={"repair_used": True},
                )
                return dag
        msg = "Unable to parse DAG planner output"
        _fire_event(
            self.config,
            "error",
            step_number=step_number,
            error=msg,
            metadata={"raw_plan": text},
        )
        raise ValueError(msg)

    async def _aparse_dag_with_fallback(
        self,
        text: str,
        *,
        system: str | None,
        step_number: int,
        on_repair_response: Callable[[int, Response], None] | None = None,
        **kwargs: Any,
    ) -> list[_DAGTask]:
        dag = _parse_dag(text)
        if dag:
            return dag
        repaired = text
        for _ in range(max(0, self.config.planner_repair_retries)):
            repair_resp = await self.client.chat(
                [
                    Message(
                        role="user",
                        content=(
                            "Repair the following into a valid JSON array of "
                            "{\"id\":\"1\",\"tool\":\"name\","
                            "\"args\":{},\"deps\":[]} objects. "
                            "Return only JSON.\n\n"
                            f"{repaired}"
                        ),
                    )
                ],
                system=system,
                **kwargs,
            )
            if on_repair_response is not None:
                on_repair_response(step_number, repair_resp)
            repaired = repair_resp.text
            dag = _parse_dag(repaired)
            if dag:
                _fire_event(
                    self.config,
                    "plan_created",
                    step_number=step_number,
                    result=repaired,
                    metadata={"repair_used": True},
                )
                return dag
        msg = "Unable to parse DAG planner output"
        _fire_event(
            self.config,
            "error",
            step_number=step_number,
            error=msg,
            metadata={"raw_plan": text},
        )
        raise ValueError(msg)


def _parse_dag(text: str) -> list[_DAGTask]:
    """Parse a JSON DAG from LLM output."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            if isinstance(items, list):
                return [
                    _DAGTask(
                        id=str(item.get("id", str(i))),
                        tool_name=str(item.get("tool", "")),
                        tool_args={str(k): str(v) for k, v in item.get("args", {}).items()},
                        deps=[str(d) for d in item.get("deps", [])],
                    )
                    for i, item in enumerate(items)
                ]
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


def _get_ready_tasks(dag: list[_DAGTask], results: dict[str, str]) -> list[_DAGTask]:
    """Find tasks whose dependencies are all satisfied."""
    return [dt for dt in dag if dt.status == "pending" and all(d in results for d in dt.deps)]


def _substitute_args(args: dict[str, str], results: dict[str, str]) -> dict[str, object]:
    """Replace $N references in args with actual results."""
    resolved: dict[str, object] = {}
    for key, value in args.items():
        if isinstance(value, str):
            for ref_id, result in results.items():
                value = value.replace(f"${ref_id}", result)
        resolved[key] = value
    return resolved
