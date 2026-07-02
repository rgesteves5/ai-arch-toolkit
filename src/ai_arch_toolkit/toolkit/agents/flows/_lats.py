"""LATS as a Flow — Monte Carlo Tree Search with ReAct rollouts."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._policy import Policy
from ai_arch_toolkit.core._state import State, StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.budget import BudgetPolicy
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep

_SCORE_RE = re.compile(r"(\d+\.?\d*)")


@dataclass(slots=True)
class _MCTSNode:
    """Monte Carlo Tree Search node."""

    state: str
    parent: _MCTSNode | None = None
    children: list[_MCTSNode] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    reflection: str = ""
    answer: str = ""


def _select_uct(node: _MCTSNode, exploration_weight: float) -> _MCTSNode:
    """Select leaf node using UCT."""
    while node.children:
        best = max(
            node.children,
            key=lambda c: (
                (c.value / c.visits if c.visits > 0 else 0.0)
                + exploration_weight * math.sqrt(math.log(node.visits + 1) / (c.visits + 1))
            ),
        )
        node = best
    return node


def _backprop(node: _MCTSNode, score: float) -> None:
    """Backpropagate score up the tree."""
    current: _MCTSNode | None = node
    while current is not None:
        current.visits += 1
        current.value += score
        current = current.parent


def lats_flow(
    llm: LLM,
    tools: ToolGroup,
    *,
    system: str = "",
    n_candidates: int = 5,
    max_rollouts: int = 10,
    exploration_weight: float = 1.41,
    max_react_iterations: int = 10,
    evaluator_fn: Callable[[str, str], float] | None = None,
    evaluator_system: str = (
        "Evaluate the following answer for the given task. "
        "Respond with a single score between 0.0 and 1.0."
    ),
    reflect_system: str = (
        "Analyze why this answer scored poorly and provide specific feedback for improvement."
    ),
    timeout: float | None = None,
    policy: Policy | None = None,
    budget_policy: BudgetPolicy | None = None,
    rollout_llm: LLM | None = None,
    rollout_tools: ToolGroup | None = None,
    eval_llm: LLM | None = None,
    solver_llm: LLM | None = None,
    reflector_llm: LLM | None = None,
) -> Flow:
    """Create a LATS Flow — MCTS with ReAct rollouts.

    Args:
        llm: Default language model.
        tools: Tool group for ReAct rollouts.
        system: Base system prompt.
        n_candidates: Number of candidates (used for node creation context).
        max_rollouts: Maximum MCTS rollouts.
        exploration_weight: UCT exploration constant.
        max_react_iterations: Max iterations per inner ReAct.
        evaluator_fn: Optional external evaluator(task, answer) → score.
        evaluator_system: System prompt for LLM-based evaluation.
        reflect_system: System prompt for reflection on low scores.
        timeout: Overall timeout in seconds.
        policy: Optional execution policy.
        budget_policy: Optional cumulative runtime budget for the flow.
        rollout_llm: Override LLM for rollouts.
        rollout_tools: Override tools for rollouts.
        eval_llm: Override LLM for evaluation.
        solver_llm: Override LLM for final solution.
        reflector_llm: Override LLM for reflection.
    """
    inner_llm = rollout_llm or llm
    inner_tools = rollout_tools or tools
    evaluator_llm = eval_llm or llm
    solve_llm = solver_llm or llm
    reflect_llm = reflector_llm or llm

    async def mcts_rollout(snap: StateSnapshot) -> Result:
        """One MCTS rollout: select, expand (ReAct), evaluate, backprop."""
        task: str = snap.require("task")
        root: _MCTSNode = snap.require("mcts_root")
        rollout_num: int = snap.get("rollout_num", 0)

        async def _complete(model: LLM, *args: Any, **kwargs: Any):
            return await model.complete(*args, **kwargs)

        # SELECT via UCT
        leaf = _select_uct(root, exploration_weight)

        # EXPAND via inner ReAct
        inner_system = system
        if leaf.reflection:
            inner_system += f"\n\nPrevious feedback:\n{leaf.reflection}"

        inner = react_flow(
            inner_llm,
            inner_tools,
            system=inner_system,
            max_iterations=max_react_iterations,
            budget_policy=budget_policy,
        )

        inner_initial = react_initial_state(leaf.state)
        state = State(operational=inner_initial)
        result = await inner.run(state)

        response = state.get("response")
        answer = response.text if response else ""

        # EVALUATE
        if evaluator_fn is not None:
            score = evaluator_fn(task, answer)
        else:
            eval_response = await _complete(
                evaluator_llm,
                [user(f"Task: {task}\n\nAnswer: {answer}\n\nScore (0.0-1.0):")],
                system=evaluator_system,
            )
            match = _SCORE_RE.search(eval_response.text)
            score = float(match.group(1)) if match else 0.5
            score = min(max(score, 0.0), 1.0)

        # Create child node
        child = _MCTSNode(
            state=f"{leaf.state}\nAttempt: {answer}",
            parent=leaf,
            answer=answer,
        )
        leaf.children.append(child)

        # BACKPROPAGATE
        _backprop(child, score)

        # Track best answer across rollouts
        best_answer: str = snap.get("best_answer", "")
        best_score: float = snap.get("best_score", 0.0)
        if score > best_score:
            best_answer = answer
            best_score = score

        artifacts: dict[str, Any] = {
            "mcts_root": root,
            "rollout_num": rollout_num + 1,
            "last_answer": answer,
            "last_score": score,
            "best_answer": best_answer,
            "best_score": best_score,
        }

        # HIGH SCORE — solve
        if score >= 0.9:
            sol_response = await _complete(
                solve_llm,
                [user(f"Task: {task}\n\nBest answer: {answer}\n\nProvide the final answer.")],
                system=system or None,
            )
            artifacts["answer"] = sol_response.text
            artifacts["response"] = sol_response
            artifacts["search_done"] = True
            return Result(
                value=sol_response.text,
                artifacts=artifacts,
                cost=result.total_cost + (sol_response.cost or 0.0),
                confidence=score,
            )

        # LOW SCORE — reflect
        if score < 0.5:
            ref_response = await _complete(
                reflect_llm,
                [
                    user(
                        f"Task: {task}\n\nAnswer: {answer}\n\n"
                        f"Score: {score:.2f}\n\nProvide feedback."
                    )
                ],
                system=reflect_system,
            )
            child.reflection = ref_response.text

        # Check if this is the last rollout — if so, solve with best answer
        if rollout_num + 1 >= max_rollouts:
            sol_response = await _complete(
                solve_llm,
                [user(f"Task: {task}\n\nBest answer: {best_answer}\n\nProvide the final answer.")],
                system=system or None,
            )
            artifacts["answer"] = sol_response.text
            artifacts["response"] = sol_response
            artifacts["search_done"] = True
            return Result(
                value=sol_response.text,
                artifacts=artifacts,
                cost=result.total_cost + (sol_response.cost or 0.0),
                confidence=best_score,
            )

        artifacts["search_done"] = False
        return Result(
            value=answer,
            artifacts=artifacts,
            cost=result.total_cost,
            confidence=score,
        )

    def search_not_done(snap: StateSnapshot) -> bool:
        return not snap.get("search_done", False)

    flow_policy = policy
    if timeout is not None and flow_policy is None:
        flow_policy = Policy(timeout=timeout)

    return Flow(
        FlowStep(step=Step(name="mcts_rollout", fn=mcts_rollout), when=search_not_done),
        name="lats",
        policy=flow_policy,
        budget_policy=budget_policy,
        max_iterations=max_rollouts,
    )


def lats_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a lats_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {
        "task": task_str,
        "mcts_root": _MCTSNode(state=task_str),
        "rollout_num": 0,
        "search_done": False,
    }
