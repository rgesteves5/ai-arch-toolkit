"""LATS as a Flow — Monte Carlo Tree Search with ReAct rollouts."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Unpack

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._common import FlowOptions
from ai_arch_toolkit.toolkit.agents.flows._keys import ANSWER, RESPONSE, TASK
from ai_arch_toolkit.toolkit.agents.flows._react import run_react
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


def _select_uct(node: _MCTSNode, exploration_weight: float, width: int) -> _MCTSNode:
    """Select the node to expand next: descend by UCT through nodes that have ``width`` children.

    A node takes up to ``width`` children, sibling attempts from its state, before the search goes
    below it; from then on UCT chooses among those siblings. A width below 1 acts as 1.
    """
    while node.children and len(node.children) >= width:
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
    llm_kwargs: dict[str, Any] | None = None,
    rollout_llm: LLM | None = None,
    rollout_tools: ToolGroup | None = None,
    eval_llm: LLM | None = None,
    solver_llm: LLM | None = None,
    reflector_llm: LLM | None = None,
    **options: Unpack[FlowOptions],
) -> Flow:
    """Create a LATS Flow — MCTS with ReAct rollouts.

    A node is a text state: the task plus the attempts that led to it. Each rollout selects, by
    UCT, the most promising node that has fewer than ``n_candidates`` children, runs one inner
    ReAct attempt from its state, scores the answer (``evaluator_fn``, else the evaluator LLM),
    adds it as a child and backpropagates the score. A node thus gets up to ``n_candidates``
    sibling attempts before the search goes below it (Zhou et al. 2024, Language Agent Tree
    Search). A low-scoring attempt gets a reflection, passed to the attempts expanded from it. The
    search stops at a score of 0.9 or after ``max_rollouts`` attempts; the solver then answers
    from the best attempt.

    Each attempt re-runs its tools from scratch: the flow has no environment reset, so tool side
    effects repeat on every rollout. Use it only with read-only, idempotent or sandboxed tools.

    Args:
        llm: Default language model.
        tools: Tool group for ReAct rollouts.
        system: Base system prompt.
        n_candidates: Sibling attempts a node gets before the search goes below it (the
            branching factor).
        max_rollouts: Maximum rollouts, one ReAct attempt each.
        exploration_weight: UCT exploration constant.
        max_react_iterations: Max iterations per inner ReAct.
        evaluator_fn: Optional external evaluator(task, answer) → score.
        evaluator_system: System prompt for LLM-based evaluation.
        reflect_system: System prompt for reflection on low scores.
        llm_kwargs: Additional kwargs passed to every phase's LLM call.
        rollout_llm: Override LLM for rollouts.
        rollout_tools: Override tools for rollouts.
        eval_llm: Override LLM for evaluation.
        solver_llm: Override LLM for final solution.
        reflector_llm: Override LLM for reflection.
        **options: The options of the ``Flow`` it builds (``FlowOptions``).
    """
    inner_llm = rollout_llm or llm
    inner_tools = rollout_tools if rollout_tools is not None else tools
    evaluator_llm = eval_llm or llm
    solve_llm = solver_llm or llm
    reflect_llm = reflector_llm or llm
    extra = llm_kwargs or {}

    async def mcts_rollout(snap: StateSnapshot) -> Result:
        """One MCTS rollout: select a node, expand it by one ReAct attempt, evaluate, backprop."""
        task: str = snap.require(TASK)
        root: _MCTSNode = snap.require("mcts_root")
        rollout_num: int = snap.get("rollout_num", 0)

        async def _complete(model: LLM, *args: Any, **kwargs: Any):
            return await model.complete(*args, **kwargs)

        # SELECT via UCT: the most promising node that still takes children
        node = _select_uct(root, exploration_weight, n_candidates)

        # EXPAND via inner ReAct: one more child of the node
        inner_system = system
        if node.reflection:
            inner_system += f"\n\nPrevious feedback:\n{node.reflection}"

        run = await run_react(
            inner_llm,
            inner_tools,
            node.state,
            system=inner_system,
            max_iterations=max_react_iterations,
            llm_kwargs=llm_kwargs,
            **options,
        )
        answer = run.answer

        # EVALUATE
        if evaluator_fn is not None:
            score = evaluator_fn(task, answer)
        else:
            eval_response = await _complete(
                evaluator_llm,
                [user(f"Task: {task}\n\nAnswer: {answer}\n\nScore (0.0-1.0):")],
                system=evaluator_system,
                **extra,
            )
            match = _SCORE_RE.search(eval_response.text)
            score = float(match.group(1)) if match else 0.5
            score = min(max(score, 0.0), 1.0)

        # Create child node
        child = _MCTSNode(
            state=f"{node.state}\nAttempt: {answer}",
            parent=node,
            answer=answer,
        )
        node.children.append(child)

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
                **extra,
            )
            artifacts[ANSWER] = sol_response.text
            artifacts[RESPONSE] = sol_response
            artifacts["search_done"] = True
            return Result(
                value=sol_response.text,
                artifacts=artifacts,
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
                **extra,
            )
            child.reflection = ref_response.text

        # Check if this is the last rollout — if so, solve with best answer
        if rollout_num + 1 >= max_rollouts:
            sol_response = await _complete(
                solve_llm,
                [user(f"Task: {task}\n\nBest answer: {best_answer}\n\nProvide the final answer.")],
                system=system or None,
                **extra,
            )
            artifacts[ANSWER] = sol_response.text
            artifacts[RESPONSE] = sol_response
            artifacts["search_done"] = True
            return Result(
                value=sol_response.text,
                artifacts=artifacts,
                confidence=best_score,
            )

        artifacts["search_done"] = False
        return Result(
            value=answer,
            artifacts=artifacts,
            confidence=score,
        )

    def search_not_done(snap: StateSnapshot) -> bool:
        return not snap.get("search_done", False)

    return Flow(
        FlowStep(step=Step(name="mcts_rollout", fn=mcts_rollout), when=search_not_done),
        name="lats",
        max_iterations=max_rollouts,
        **options,
    )


def lats_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a lats_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {
        TASK: task_str,
        "mcts_root": _MCTSNode(state=task_str),
        "rollout_num": 0,
        "search_done": False,
    }
