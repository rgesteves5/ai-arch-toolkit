"""LATSAgent — Language Agent Tree Search (MCTS-based)."""

from __future__ import annotations

import math
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
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

_DEFAULT_EVALUATOR_SYSTEM = (
    "You are an evaluator. Score the answer on a scale of 0.0 to 1.0 "
    "for correctness and completeness. Respond with ONLY a number."
)

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_score(text: str) -> float:
    """Extract a 0-1 score from text."""
    m = _SCORE_RE.search(text)
    if m:
        return min(1.0, max(0.0, float(m.group(1))))
    return 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class LATSConfig:
    """Configuration specific to the LATS agent."""

    n_candidates: int = 5
    max_rollouts: int = 10
    exploration_weight: float = 1.41
    evaluator_fn: Callable[[str, str], float] | None = None
    evaluator_system: str = _DEFAULT_EVALUATOR_SYSTEM
    rollout: PhaseConfig | None = None
    evaluator: PhaseConfig | None = None
    solver: PhaseConfig | None = None
    reflector: PhaseConfig | None = None


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Node:
    """A mutable node in the MCTS tree.

    Not frozen because MCTS requires in-place updates to visits, value,
    children, and reflection during backpropagation and expansion.
    """

    state: str
    parent: _Node | None = None
    children: list[_Node] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    reflection: str = ""
    answer: str = ""


def _uct_score(node: _Node, exploration_weight: float) -> float:
    """Upper Confidence bound for Trees."""
    if node.visits == 0:
        return float("inf")
    parent_visits = node.parent.visits if node.parent else 1
    exploitation = node.value / node.visits
    exploration = exploration_weight * math.sqrt(math.log(parent_visits) / node.visits)
    return exploitation + exploration


def _select_uct(root: _Node, exploration_weight: float) -> _Node:
    """Select the best leaf node using UCT."""
    node = root
    while node.children:
        node = max(node.children, key=lambda n: _uct_score(n, exploration_weight))
    return node


def _backprop(node: _Node, score: float) -> None:
    """Backpropagate a score up the tree."""
    current: _Node | None = node
    while current is not None:
        current.visits += 1
        current.value += score
        current = current.parent


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LATSAgent(BaseAgent):
    """Language Agent Tree Search: MCTS with ReAct rollouts.

    Each rollout:
    1. **Select** — UCT selects a leaf node.
    2. **Expand** — Inner ReActAgent attempts to solve from that state.
    3. **Evaluate** — Score the answer (LLM or external evaluator).
    4. **Backpropagate** — Update node statistics.

    Low-scoring rollouts trigger reflection for future guidance.
    """

    __slots__ = ("lats",)

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
        lats: LATSConfig | None = None,
    ) -> None:
        super().__init__(llm, tools, config=config)
        self.lats = lats or LATSConfig()

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        task_str = task if isinstance(task, str) else str(task)
        root = _Node(state=task_str)
        step_num = 0
        start = time.monotonic()
        rollout_llm = _resolve_llm(self.lats.rollout, self.llm)
        rollout_tools = _resolve_tools(self.lats.rollout, self.tools)
        eval_llm = _resolve_llm(self.lats.evaluator, self.llm)
        solver_llm = _resolve_llm(self.lats.solver, self.llm)
        reflector_llm = _resolve_llm(self.lats.reflector, self.llm)

        for _rollout in range(self.lats.max_rollouts):
            # --- outer stop conditions ---
            if self._check_timeout(start):
                yield AgentEvent(type="step_end", step=step_num + 1, stop_reason="timeout")
                return

            # SELECT
            node = _select_uct(root, self.lats.exploration_weight)

            step_num += 1
            yield AgentEvent(type="step_start", step=step_num)

            # EXPAND — run inner ReAct from this node's state
            inner_system = self.config.system or ""
            if node.reflection:
                inner_system += f"\n\nReflection from previous attempt:\n{node.reflection}"

            inner_config = AgentConfig(
                max_iterations=3,
                system=inner_system,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                tool_choice=self.config.tool_choice,
                parallel_tool_calls=self.config.parallel_tool_calls,
                llm_kwargs=self.config.llm_kwargs,
            )
            inner = ReActAgent(rollout_llm, rollout_tools, config=inner_config)

            last_answer = ""
            inner_stop: str | None = None

            async for event in inner._run_loop(node.state):
                if event.type == "step_end":
                    if event.stop_reason is not None:
                        inner_stop = event.stop_reason
                    if event.response is not None:
                        last_answer = event.response.text

                # Re-number and yield inner events, skip terminal step_end
                if event.type == "step_end" and event.stop_reason is not None:
                    if event.stop_reason in ("timeout", "error", "budget_exhausted"):
                        yield AgentEvent(
                            type=event.type,
                            step=step_num,
                            response=event.response,
                            stop_reason=event.stop_reason,
                            error=event.error,
                        )
                    continue

                yield AgentEvent(
                    type=event.type,
                    step=step_num,
                    tool_name=event.tool_name,
                    tool_call_id=event.tool_call_id,
                    tool_args=event.tool_args,
                    result=event.result,
                    error=event.error,
                    response=event.response,
                )

            if inner_stop in ("timeout", "error", "budget_exhausted"):
                yield AgentEvent(
                    type="step_end",
                    step=step_num,
                    stop_reason=inner_stop,
                )
                return

            # EVALUATE
            if self.lats.evaluator_fn:
                score = self.lats.evaluator_fn(task_str, last_answer)
            else:
                eval_response = await eval_llm.complete(
                    [
                        user(
                            f"Task: {task_str}\n\nAnswer: {last_answer}\n\n"
                            f"Score this answer 0.0 to 1.0."
                        )
                    ],
                    system=self.lats.evaluator_system,
                )
                score = _parse_score(eval_response.text)

            # Create child node
            child = _Node(state=f"{node.state}\nAttempt: {last_answer}", parent=node)
            child.answer = last_answer
            node.children.append(child)

            yield AgentEvent(type="step_end", step=step_num)

            # BACKPROPAGATE
            _backprop(child, score)

            if score >= 0.9:
                step_num += 1
                yield AgentEvent(type="step_start", step=step_num)
                # Final answer via solver
                final_response = await solver_llm.complete(
                    [
                        user(
                            f"Task: {task_str}\n\nBest answer: {last_answer}\n\n"
                            f"Produce a final, polished answer."
                        )
                    ],
                    system=self.config.system or None,
                    output_schema=self.config.output_schema,
                    **self.config.llm_kwargs,
                )
                yield AgentEvent(
                    type="step_end",
                    step=step_num,
                    response=final_response,
                    stop_reason="completed",
                )
                return

            # REFLECT on low scores
            if score < 0.5:
                reflect_response = await reflector_llm.complete(
                    [
                        user(
                            f"Task: {task_str}\n\nAttempted answer: {last_answer}\n\n"
                            f"Score: {score:.2f}\n\nWhat went wrong and how to improve?"
                        )
                    ],
                )
                child.reflection = reflect_response.text

        # All rollouts exhausted
        yield AgentEvent(
            type="step_end",
            step=step_num,
            stop_reason="max_iterations",
        )
