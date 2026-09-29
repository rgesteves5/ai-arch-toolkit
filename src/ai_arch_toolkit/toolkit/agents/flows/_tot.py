"""Tree of Thoughts as a Flow — generate, evaluate, expand search tree."""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Literal, Unpack

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._state import StateSnapshot
from ai_arch_toolkit.core._step import Result, Step
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._common import FlowOptions, parse_score
from ai_arch_toolkit.toolkit.agents.flows._keys import ANSWER, RESPONSE, TASK
from ai_arch_toolkit.toolkit.flow._flow import Flow, FlowStep

_NUMBERED_RE = re.compile(r"^\d+\.\s+(.+)", re.MULTILINE)

type _Node = tuple[str, int, float]  # (reasoning so far, depth, score)


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
    llm_kwargs: dict[str, Any] | None = None,
    gen_llm: LLM | None = None,
    eval_llm: LLM | None = None,
    solver_llm: LLM | None = None,
    **options: Unpack[FlowOptions],
) -> Flow:
    """Create a Tree of Thoughts Flow — DFS/BFS search over reasoning paths.

    As in Yao et al. 2023 (https://arxiv.org/abs/2305.10601), DFS expands the most promising
    child first, and BFS expands one level at a time, keeping the best ``n_candidates`` states of
    each. When the iterations or the states to expand run out, the flow answers from the best
    state found: the best-scored of the deepest.

    Args:
        llm: Default language model.
        tools: Tool group (currently unused; reserved for future solve phase).
        system: Base system prompt.
        n_candidates: Number of candidate thoughts to generate per node, and of states BFS keeps
            per level.
        max_depth: Maximum depth of the search tree.
        max_iterations: Maximum search iterations, each taking one state from the frontier.
        strategy: Search strategy — 'dfs' or 'bfs'.
        evaluator_system: System prompt for scoring thoughts.
        llm_kwargs: Additional kwargs passed to every phase's LLM call.
        gen_llm: Override LLM for generating candidate thoughts.
        eval_llm: Override LLM for evaluating thoughts.
        solver_llm: Override LLM for final solution.
        **options: The options of the ``Flow`` it builds (``FlowOptions``).
    """
    generator_llm = gen_llm or llm
    evaluator_llm = eval_llm or llm
    solve_llm = solver_llm or llm
    extra = llm_kwargs or {}

    async def search_step(snap: StateSnapshot) -> Result:
        """One iteration of tree search: select, generate, evaluate, expand."""
        task: str = snap.require(TASK)
        # A copy: the step returns the frontier it leaves instead of mutating the state's.
        frontier: deque[_Node] = deque(snap.require("frontier"))
        best_state: _Node = snap.get("best_state", (task, 0, 0.0))
        iteration: int = snap.get("iteration", 0)

        async def _complete(model: LLM, *args: Any, **kwargs: Any):
            return await model.complete(*args, **kwargs)

        async def _solve(reasoning: str, confidence: float) -> Result:
            response = await _complete(
                solve_llm,
                [
                    user(
                        f"Task: {task}\n\nReasoning so far:\n{reasoning}\n\n"
                        "Provide the final answer."
                    )
                ],
                system=system or None,
                **extra,
            )
            return Result(
                value=response.text,
                artifacts={
                    ANSWER: response.text,
                    RESPONSE: response,
                    "search_done": True,
                    "frontier": frontier,
                    "iteration": iteration + 1,
                },
                confidence=confidence,
            )

        # NOTHING TO EXPAND — solve from the best state found
        if not frontier:
            return await _solve(best_state[0], best_state[2])

        # SELECT
        state, depth, _score = frontier.pop() if strategy == "dfs" else frontier.popleft()

        # MAX DEPTH — solve directly
        if depth >= max_depth:
            return await _solve(state, 1.0)

        # GENERATE candidates
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
            **extra,
        )
        candidates = _NUMBERED_RE.findall(gen_response.text)[:n_candidates]

        # EVALUATE candidates
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            eval_response = await _complete(
                evaluator_llm,
                [
                    user(
                        f"Task: {task}\n\nReasoning: {state}\n\n"
                        f"Next step: {candidate}\n\nScore (0.0-1.0):"
                    )
                ],
                system=evaluator_system,
                **extra,
            )
            scored.append((parse_score(eval_response.text), candidate))

        # HIGH CONFIDENCE — solve immediately
        best_score, best_thought = max(scored, default=(0.0, ""))
        if best_score >= 0.9:
            return await _solve(f"{state}\n{best_thought}" if state else best_thought, best_score)

        # EXPAND the candidates into the frontier, best first (ties in the generator's order)
        children: list[_Node] = [
            (f"{state}\n{thought}" if state else thought, depth + 1, score)
            for score, thought in sorted(scored, key=lambda x: x[0], reverse=True)
        ]
        # DFS pushes them worst first, so that pop() takes the most promising child next.
        frontier.extend(reversed(children) if strategy == "dfs" else children)
        if strategy != "dfs" and frontier and frontier[0][1] > depth:
            # BFS has expanded a level: the next one keeps its best n_candidates states.
            ranked = sorted(frontier, key=lambda node: node[2], reverse=True)
            frontier = deque(ranked[:n_candidates])
        best_state = max([best_state, *children], key=lambda node: (node[1], node[2]))

        # OUT OF STATES OR ITERATIONS — solve from the best state found
        if not frontier or iteration + 1 >= max_iterations:
            return await _solve(best_state[0], best_state[2])

        return Result(
            value=None,
            artifacts={
                "frontier": frontier,
                "best_state": best_state,
                "search_done": False,
                "iteration": iteration + 1,
            },
        )

    def search_not_done(snap: StateSnapshot) -> bool:
        return not snap.get("search_done", False)

    return Flow(
        FlowStep(step=Step(name="search_step", fn=search_step), when=search_not_done),
        name="tot",
        max_iterations=max_iterations,
        **options,
    )


def tot_initial_state(task: Content) -> dict[str, Any]:
    """Create the initial operational state for a tot_flow."""
    task_str = task if isinstance(task, str) else str(task)
    return {
        TASK: task_str,
        "frontier": deque([(task_str, 0, 0.0)]),
        "search_done": False,
        "iteration": 0,
    }
