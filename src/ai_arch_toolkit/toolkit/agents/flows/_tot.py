"""Tree of Thoughts as a Flow — generate, evaluate, expand search tree."""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Literal

from ai_arch_toolkit.core._budget import BudgetExceeded, BudgetPolicy, BudgetState
from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep

_NUMBERED_RE = re.compile(r"^\d+\.\s+(.+)", re.MULTILINE)
_SCORE_RE = re.compile(r"(\d+\.?\d*)")


def tot_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    system: str = "",
    n_candidates: int = 3,
    max_depth: int = 3,
    max_iterations: int = 10,
    strategy: Literal["dfs", "bfs"] = "dfs",
    evaluator_system: str = (
        "Evaluate the following reasoning step for the given task. "
        "Respond with a single score between 0.0 and 1.0."
    ),
    timeout: float | None = None,
    policy: Policy | None = None,
    budget_policy: BudgetPolicy | None = None,
    gen_llm: LLM | None = None,
    eval_llm: LLM | None = None,
    solver_llm: LLM | None = None,
) -> Flow:
    """Create a Tree of Thoughts Flow — DFS/BFS search over reasoning paths.

    Args:
        llm: Default language model.
        tools: Tool group (currently unused; reserved for future solve phase).
        system: Base system prompt.
        n_candidates: Number of candidate thoughts to generate per node.
        max_depth: Maximum depth of the search tree.
        max_iterations: Maximum search iterations.
        strategy: Search strategy — 'dfs' or 'bfs'.
        evaluator_system: System prompt for scoring thoughts.
        timeout: Overall timeout in seconds.
        policy: Optional execution policy.
        budget_policy: Optional cumulative runtime budget for the flow.
        gen_llm: Override LLM for generating candidate thoughts.
        eval_llm: Override LLM for evaluating thoughts.
        solver_llm: Override LLM for final solution.
    """
    generator_llm = gen_llm or llm
    evaluator_llm = eval_llm or llm
    solve_llm = solver_llm or llm

    async def search_step(snap: StateSnapshot) -> Result:
        """One iteration of tree search: select, generate, evaluate, expand."""
        task: str = snap.require("task")
        frontier: deque[tuple[str, int]] = snap.require("frontier")
        iteration: int = snap.get("iteration", 0)
        budget_state = _budget_state_from_snapshot(snap)

        async def _complete(model: LLM, *args: Any, **kwargs: Any):
            nonlocal budget_state
            if budget_state is not None:
                budget_state.check_llm_calls()
            response = await model.complete(*args, **kwargs)
            if budget_state is not None:
                budget_state = budget_state.record_response(response)
            return response

        def _budget_artifacts() -> dict[str, Any]:
            return {"budget_state": budget_state} if budget_state is not None else {}

        def _budget_error(exc: BudgetExceeded) -> Result:
            if budget_state is None:
                return Result(error=str(exc), artifacts={"budget_exceeded": exc.to_dict()})
            exceeded = budget_state.with_exceeded(exc)
            return Result(
                error=str(exc),
                artifacts={
                    "budget_exceeded": exc.to_dict(),
                    "budget_state": exceeded,
                },
            )

        if not frontier:
            return Result(
                value=None,
                artifacts={"search_done": True, "iteration": iteration},
            )

        # SELECT
        if strategy == "dfs":
            state, depth = frontier.pop()
        else:
            state, depth = frontier.popleft()

        # MAX DEPTH — solve directly
        if depth >= max_depth:
            try:
                response = await _complete(
                    solve_llm,
                    [
                        user(
                            f"Task: {task}\n\nReasoning so far:\n{state}\n\n"
                            "Provide the final answer."
                        )
                    ],
                    system=system or None,
                )
            except BudgetExceeded as exc:
                return _budget_error(exc)
            return Result(
                value=response.text,
                artifacts={
                    "answer": response.text,
                    "response": response,
                    "search_done": True,
                    "frontier": frontier,
                    "iteration": iteration + 1,
                    **_budget_artifacts(),
                },
                usage=response.usage,
                cost=response.cost or 0.0,
                confidence=1.0,
            )

        # GENERATE candidates
        try:
            gen_response = await _complete(
                generator_llm,
                [
                    user(
                        f"Task: {task}\n\nCurrent reasoning:\n{state}\n\n"
                        f"Generate {n_candidates} distinct next reasoning steps. "
                        f"Format: 1. Step\n2. Step\n..."
                    )
                ],
                system=system or None,
            )
        except BudgetExceeded as exc:
            return _budget_error(exc)
        candidates = _NUMBERED_RE.findall(gen_response.text)[:n_candidates]

        if not candidates:
            return Result(
                value=None,
                artifacts={
                    "search_done": not bool(frontier),
                    "frontier": frontier,
                    "iteration": iteration + 1,
                    **_budget_artifacts(),
                },
                usage=gen_response.usage,
                cost=gen_response.cost or 0.0,
            )

        # EVALUATE candidates
        scored: list[tuple[float, str]] = []
        total_cost = gen_response.cost or 0.0
        for candidate in candidates:
            try:
                eval_response = await _complete(
                    evaluator_llm,
                    [
                        user(
                            f"Task: {task}\n\nReasoning: {state}\n\n"
                            f"Next step: {candidate}\n\nScore (0.0-1.0):"
                        )
                    ],
                    system=evaluator_system,
                )
            except BudgetExceeded as exc:
                return _budget_error(exc)
            match = _SCORE_RE.search(eval_response.text)
            score = float(match.group(1)) if match else 0.5
            score = min(max(score, 0.0), 1.0)
            scored.append((score, candidate))
            total_cost += eval_response.cost or 0.0

        # HIGH CONFIDENCE — solve immediately
        best_score, best_thought = max(scored)
        if best_score >= 0.9:
            full_reasoning = f"{state}\n{best_thought}" if state else best_thought
            try:
                response = await _complete(
                    solve_llm,
                    [
                        user(
                            f"Task: {task}\n\nReasoning:\n{full_reasoning}\n\n"
                            f"Provide the final answer."
                        )
                    ],
                    system=system or None,
                )
            except BudgetExceeded as exc:
                return _budget_error(exc)
            return Result(
                value=response.text,
                artifacts={
                    "answer": response.text,
                    "response": response,
                    "search_done": True,
                    "frontier": frontier,
                    "iteration": iteration + 1,
                    **_budget_artifacts(),
                },
                usage=response.usage,
                cost=total_cost + (response.cost or 0.0),
                confidence=best_score,
            )

        # EXPAND top candidates into frontier
        sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
        for _score, thought in sorted_scored[:n_candidates]:
            new_state = f"{state}\n{thought}" if state else thought
            frontier.append((new_state, depth + 1))

        return Result(
            value=None,
            artifacts={
                "frontier": frontier,
                "search_done": False,
                "iteration": iteration + 1,
                **_budget_artifacts(),
            },
            cost=total_cost,
        )

    def search_not_done(snap: StateSnapshot) -> bool:
        return not snap.get("search_done", False)

    flow_policy = policy
    if timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=timeout)

    return Flow(
        FlowStep(step=Step(name="search_step", fn=search_step), when=search_not_done),
        name="tot",
        policy=flow_policy,
        budget_policy=budget_policy,
        max_iterations=max_iterations,
    )


def tot_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a tot_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {
        "task": task_str,
        "frontier": deque([(task_str, 0)]),
        "search_done": False,
        "iteration": 0,
    }


def _budget_state_from_snapshot(snap: StateSnapshot) -> BudgetState | None:
    value = snap.get("budget_state")
    return value if isinstance(value, BudgetState) else None
