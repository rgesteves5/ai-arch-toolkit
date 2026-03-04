"""LLMCompilerAgent — Plan DAG → Parallel Execute → Join → Optional Replan."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import (
    AgentConfig,
    AgentEvent,
    BaseAgent,
    PhaseConfig,
    _resolve_llm,
    _resolve_tools,
)
from ai_arch_toolkit.toolkit.agents._react import ReActAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_PLANNER_SYSTEM = (
    "You are a task planner. Given a task and available tools, produce a DAG "
    "of sub-tasks. Each sub-task should be on its own line in this format:\n\n"
    "$1. Description of first task [deps: none]\n"
    "$2. Description of second task [deps: $1]\n"
    "$3. Description of third task [deps: $1, $2]\n\n"
    "Use 'none' for tasks with no dependencies. Reference earlier tasks with "
    "$N notation.\n\nAvailable tools:\n"
)

_DEFAULT_JOINER_SYSTEM = (
    "You are a synthesizer. Given the original task and results from all "
    "sub-tasks, produce a final answer. If the results are insufficient "
    "and you need a different plan, respond with exactly REPLAN on the first "
    "line followed by an explanation."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCompilerConfig:
    """Configuration specific to the LLMCompiler agent."""

    max_replans: int = 2
    planner_system: str = _DEFAULT_PLANNER_SYSTEM
    joiner_system: str = _DEFAULT_JOINER_SYSTEM
    planner: PhaseConfig | None = None
    executor: PhaseConfig | None = None
    joiner: PhaseConfig | None = None

    def __post_init__(self) -> None:
        if self.max_replans < 0:
            raise ValueError(f"max_replans must be >= 0, got {self.max_replans}")


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

_DAG_RE = re.compile(r"\$(\d+)\.\s*(.+?)\s*\[deps:\s*(.*?)\]")


@dataclass(slots=True)
class _DAGTask:
    """A single node in the task DAG. Mutable — result/done/failed are updated."""

    id: int
    description: str
    deps: tuple[int, ...]
    result: str = ""
    done: bool = False
    failed: bool = False


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_dag(text: str) -> list[_DAGTask]:
    """Parse planner output into a list of DAG tasks."""
    tasks: list[_DAGTask] = []
    for match in _DAG_RE.finditer(text):
        task_id = int(match.group(1))
        description = match.group(2).strip()
        deps_str = match.group(3).strip()
        if deps_str.lower() == "none" or not deps_str:
            deps: tuple[int, ...] = ()
        else:
            deps = tuple(int(d.strip().lstrip("$")) for d in deps_str.split(","))
        tasks.append(_DAGTask(id=task_id, description=description, deps=deps))
    return tasks


def _ready_tasks(tasks: list[_DAGTask]) -> list[_DAGTask]:
    """Return tasks whose dependencies are all done and that are not yet done.

    Tasks whose dependencies include a failed task are skipped (marked
    failed themselves) since they cannot produce meaningful results.
    Failure is propagated transitively in a single pass.
    """
    # Propagate failure transitively until stable
    changed = True
    while changed:
        changed = False
        failed_ids = {t.id for t in tasks if t.failed}
        for t in tasks:
            if not t.done and not t.failed and any(d in failed_ids for d in t.deps):
                t.failed = True
                t.done = True
                t.result = "Skipped: dependency failed"
                changed = True

    done_ids = {t.id for t in tasks if t.done}
    return [t for t in tasks if not t.done and all(d in done_ids for d in t.deps)]


def _substitute_refs(text: str, tasks: list[_DAGTask]) -> str:
    """Replace $N references in text with completed task results."""
    task_map = {t.id: t for t in tasks}

    def _replacer(m: re.Match[str]) -> str:
        tid = int(m.group(1))
        t = task_map.get(tid)
        if t and t.done:
            return t.result
        return m.group(0)

    return re.sub(r"\$(\d+)", _replacer, text)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LLMCompilerAgent(BaseAgent):
    """LLMCompiler: plan a DAG → execute tasks in parallel → join → replan.

    Three phases per iteration:
    1. **Plan** — LLM generates a DAG of sub-tasks with dependencies.
    2. **Execute** — Tasks are executed in topological order; independent
       tasks run in parallel via ``asyncio.gather``.
    3. **Join** — LLM synthesizes results into a final answer or requests
       a replan.
    """

    __slots__ = ("compiler",)

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
        compiler: LLMCompilerConfig | None = None,
    ) -> None:
        super().__init__(llm, tools, config=config)
        self.compiler = compiler or LLMCompilerConfig()

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        task_str = task if isinstance(task, str) else str(task)
        step_num = 0
        start = time.monotonic()
        user_sys = self.config.system

        # Resolve phase overrides once (loop-invariant).
        planner_llm = _resolve_llm(self.compiler.planner, self.llm)
        exec_tools = _resolve_tools(self.compiler.executor, self.tools)
        joiner_llm = _resolve_llm(self.compiler.joiner, self.llm)

        # Augment planner system with tool descriptions.
        # Use the executor's resolved tools so the plan matches what's available.
        tool_lines: list[str] = []
        for defn in exec_tools.definitions:
            desc = defn.get("description", "")
            tool_lines.append(f"- {defn['name']}: {desc}")
        augmented_planner = self.compiler.planner_system + "\n".join(tool_lines)
        if user_sys:
            augmented_planner = user_sys + "\n\n" + augmented_planner

        for _replan in range(self.compiler.max_replans + 1):
            # --- outer stop conditions ---
            if self._check_timeout(start):
                yield AgentEvent(type="step_end", step=step_num + 1, stop_reason="timeout")
                return

            # -----------------------------------------------------------
            # Phase 1: PLAN
            # -----------------------------------------------------------
            step_num += 1
            yield AgentEvent(type="step_start", step=step_num)

            plan_response = await planner_llm.complete(
                [user(task_str)],
                system=augmented_planner,
            )
            yield AgentEvent(type="step_end", step=step_num, response=plan_response)

            dag = _parse_dag(plan_response.text)

            # -----------------------------------------------------------
            # Phase 2: EXECUTE (topological parallel)
            # -----------------------------------------------------------
            # Mark tasks with invalid deps (referencing nonexistent IDs) as failed
            valid_ids = {t.id for t in dag}
            for t in dag:
                if any(d not in valid_ids for d in t.deps):
                    t.failed = True
                    t.done = True
                    t.result = "Skipped: invalid dependency"

            while True:
                ready = _ready_tasks(dag)
                if not ready:
                    break

                # Run all ready tasks in parallel — collect events
                collected: list[list[AgentEvent]] = list(
                    await asyncio.gather(
                        *[self._execute_dag_task(t, dag, task_str) for t in ready]
                    )
                )

                # Yield collected events sequentially per task
                for task_events in collected:
                    step_num += 1
                    yield AgentEvent(type="step_start", step=step_num)
                    for evt in task_events:
                        yield AgentEvent(
                            type=evt.type,
                            step=step_num,
                            tool_name=evt.tool_name,
                            tool_call_id=evt.tool_call_id,
                            tool_args=evt.tool_args,
                            result=evt.result,
                            error=evt.error,
                            response=evt.response,
                        )
                    yield AgentEvent(type="step_end", step=step_num)

            # -----------------------------------------------------------
            # Phase 3: JOIN
            # -----------------------------------------------------------
            step_num += 1
            yield AgentEvent(type="step_start", step=step_num)

            results_block = "\n".join(
                f"Task ${t.id} ({t.description}): {t.result}" for t in dag if t.done
            )
            join_msg = (
                f"Original task: {task_str}\n\n"
                f"Sub-task results:\n{results_block}\n\n"
                f"Produce a final answer, or respond with REPLAN if results are "
                f"insufficient."
            )

            joiner_system = self.compiler.joiner_system
            if user_sys:
                joiner_system = user_sys + "\n\n" + joiner_system

            join_response = await joiner_llm.complete(
                [user(join_msg)],
                system=joiner_system,
                output_schema=self.config.output_schema,
                **self.config.llm_kwargs,
            )

            if "REPLAN" not in join_response.text.split("\n")[0]:
                yield AgentEvent(
                    type="step_end",
                    step=step_num,
                    response=join_response,
                    stop_reason="completed",
                )
                return

            yield AgentEvent(type="step_end", step=step_num, response=join_response)

        # Max replans exhausted
        yield AgentEvent(
            type="step_end",
            step=step_num,
            stop_reason="max_iterations",
        )

    async def _execute_dag_task(
        self,
        dag_task: _DAGTask,
        dag: list[_DAGTask],
        task_str: str,
    ) -> list[AgentEvent]:
        """Execute a single DAG task via inner ReAct, collecting events."""
        description = _substitute_refs(dag_task.description, dag)
        context_parts = [f"Original task: {task_str}\nCurrent sub-task: {description}"]
        if self.config.system:
            context_parts.insert(0, self.config.system)
        context = "\n\n".join(context_parts)

        inner_config = AgentConfig(
            max_iterations=3,
            system=context,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            tool_choice=self.config.tool_choice,
            parallel_tool_calls=self.config.parallel_tool_calls,
            llm_kwargs=self.config.llm_kwargs,
        )
        exec_llm = _resolve_llm(self.compiler.executor, self.llm)
        exec_tools = _resolve_tools(self.compiler.executor, self.tools)
        inner = ReActAgent(exec_llm, exec_tools, config=inner_config)

        events: list[AgentEvent] = []
        last_answer = ""
        inner_stop: str | None = None

        async for event in inner._run_loop(description):
            if event.type == "step_end":
                if event.response is not None:
                    last_answer = event.response.text
                # Skip terminal step_end — outer loop handles it
                if event.stop_reason is not None:
                    inner_stop = event.stop_reason
                    continue
            events.append(event)

        dag_task.result = last_answer
        dag_task.done = True
        if inner_stop in ("error", "timeout", "budget_exhausted"):
            dag_task.failed = True
        return events
