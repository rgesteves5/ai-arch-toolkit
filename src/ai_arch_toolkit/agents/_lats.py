"""LATS (Language Agent Tree Search) agent — MCTS-based."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentConfig,
    AgentResult,
    AgentStep,
    BaseAgent,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.agents._parsing import parse_numbered_items, parse_score
from ai_arch_toolkit.agents._react import ReActAgent
from ai_arch_toolkit.llm._types import Message, Usage

_EXPAND_PROMPT = (
    "Given the current state, generate {n} distinct candidate "
    "actions or approaches to try next. Number each (1., 2., etc.).\n\n"
    "Problem: {task}\n\n"
    "Current state:\n{state}\n\n"
    "Previous reflections:\n{reflections}"
)

_EVALUATE_PROMPT = (
    "Evaluate how promising this approach is for solving the problem. "
    "Return a score between 0.0 (terrible) and 1.0 (excellent). "
    "Return ONLY the numeric score.\n\n"
    "Problem: {task}\n\nApproach:\n{approach}"
)

_REFLECT_PROMPT = (
    "The following approach failed to solve the problem. "
    "Analyze what went wrong and suggest what to try differently.\n\n"
    "Problem: {task}\n\n"
    "Failed approach: {approach}\n\n"
    "Result: {result}"
)


@dataclass
class _TreeNode:
    state: str
    value: float = 0.0
    visits: int = 0
    children: list[_TreeNode] = field(default_factory=list)
    parent: _TreeNode | None = field(default=None, repr=False)
    action: str = ""
    reflection: str = ""
    is_terminal: bool = False
    is_success: bool = False
    depth: int = 0

    def uct(self, c: float) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore

    def best_child(self, c: float) -> _TreeNode:
        """Select best child by UCT score."""
        return max(self.children, key=lambda n: n.uct(c))


class LATSAgent(BaseAgent):
    """LATS agent using Monte Carlo Tree Search.

    1. SELECT: Traverse from root using UCT.
    2. EXPAND: Generate candidate actions via LLM.
    3. EVALUATE: Score each candidate via LLM-as-judge.
    4. SIMULATE: Run best candidate with inner ReActAgent.
    5. BACKPROPAGATE: Update values from leaf to root.
    6. REFLECT: On failure, LLM generates reflection.
    7. Repeat for max_iterations MCTS iterations.
    """

    def _build_expand_prompt(self, task: str, node: _TreeNode, num_expansions: int) -> str:
        reflections = _collect_reflections(node)
        return _EXPAND_PROMPT.format(
            n=num_expansions,
            task=task,
            state=node.state,
            reflections=reflections or "(none)",
        )

    def _build_reflect_prompt(self, task: str, approach: str, result: str) -> str:
        return _REFLECT_PROMPT.format(
            task=task,
            approach=approach,
            result=result,
        )

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the LATS MCTS loop."""
        c = kwargs.pop("exploration_weight", math.sqrt(2))
        num_expansions = kwargs.pop("num_expansions", 3)
        evaluator = kwargs.pop("evaluator", None)
        max_iters = self.config.max_iterations
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        start = time.monotonic()

        root = _TreeNode(state="(start)", depth=0)
        best_terminal: _TreeNode | None = None

        for iteration in range(1, max_iters + 1):
            if self._check_timeout(start):
                break
            _fire_event(
                self.config,
                "step_start",
                step_number=iteration,
            )

            # SELECT
            node = root
            while node.children and not node.is_terminal:
                node = node.best_child(c)

            if node.is_terminal and node.is_success:
                _fire_event(
                    self.config,
                    "step_end",
                    step_number=iteration,
                )
                if best_terminal is None or node.value > best_terminal.value:
                    best_terminal = node
                continue

            # EXPAND
            expand_prompt = self._build_expand_prompt(task, node, num_expansions)
            expand_resp = self.client.chat(
                [Message(role="user", content=expand_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, expand_resp.usage)
            all_steps.append(
                AgentStep(
                    step_number=iteration,
                    response=expand_resp,
                )
            )
            candidates = parse_numbered_items(expand_resp.text, num_expansions)

            # EVALUATE each candidate
            scored: list[tuple[str, float]] = []
            for cand in candidates:
                eval_prompt = _EVALUATE_PROMPT.format(task=task, approach=cand)
                eval_resp = self.client.chat(
                    [Message(role="user", content=eval_prompt)],
                    system=system,
                    **kwargs,
                )
                total_usage = _accumulate_usage(total_usage, eval_resp.usage)
                score = parse_score(eval_resp.text)
                scored.append((cand, score))

            # Create child nodes
            for cand, score in scored:
                child = _TreeNode(
                    state=cand,
                    value=score,
                    visits=0,
                    parent=node,
                    action=cand,
                    depth=node.depth + 1,
                )
                node.children.append(child)

            if not node.children:
                _fire_event(
                    self.config,
                    "step_end",
                    step_number=iteration,
                )
                continue

            # SIMULATE best child
            best_child = max(node.children, key=lambda n: n.value)
            inner_config = AgentConfig(
                max_iterations=3,
                system=best_child.state,
                max_tokens=self.config.max_tokens,
            )
            inner = ReActAgent(self.client, self.tools, config=inner_config)
            inner_result = inner.run(task, **kwargs)
            total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
            all_steps.extend(inner_result.steps)

            # Evaluate simulation result
            sim_score = (
                evaluator(inner_result.answer)
                if evaluator
                else _evaluate_result(inner_result.answer)
            )
            best_child.is_terminal = True

            if sim_score >= 0.5:
                best_child.is_success = True
                best_child.state = inner_result.answer
                _backpropagate(best_child, sim_score)
                if best_terminal is None or sim_score > best_terminal.value:
                    best_terminal = best_child
            else:
                # REFLECT
                reflect_prompt = self._build_reflect_prompt(
                    task, best_child.action, inner_result.answer
                )
                reflect_resp = self.client.chat(
                    [
                        Message(
                            role="user",
                            content=reflect_prompt,
                        )
                    ],
                    system=system,
                    **kwargs,
                )
                total_usage = _accumulate_usage(total_usage, reflect_resp.usage)
                best_child.reflection = reflect_resp.text
                _backpropagate(best_child, sim_score)
                _fire_event(
                    self.config,
                    "reflection",
                    step_number=iteration,
                    result=reflect_resp.text,
                )

            _fire_event(
                self.config,
                "step_end",
                step_number=iteration,
            )

        if best_terminal:
            answer = best_terminal.state
        elif self._check_timeout(start):
            answer = "[timeout exceeded]"
        else:
            answer = "[no solution found]"
        return AgentResult(
            answer=answer,
            steps=tuple(all_steps),
            total_usage=total_usage,
        )

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the LATS MCTS loop asynchronously."""
        c = kwargs.pop("exploration_weight", math.sqrt(2))
        num_expansions = kwargs.pop("num_expansions", 3)
        evaluator = kwargs.pop("evaluator", None)
        max_iters = self.config.max_iterations
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        start = time.monotonic()

        root = _TreeNode(state="(start)", depth=0)
        best_terminal: _TreeNode | None = None

        for iteration in range(1, max_iters + 1):
            if self._check_timeout(start):
                break
            _fire_event(
                self.config,
                "step_start",
                step_number=iteration,
            )

            # SELECT
            node = root
            while node.children and not node.is_terminal:
                node = node.best_child(c)

            if node.is_terminal and node.is_success:
                _fire_event(
                    self.config,
                    "step_end",
                    step_number=iteration,
                )
                if best_terminal is None or node.value > best_terminal.value:
                    best_terminal = node
                continue

            # EXPAND
            expand_prompt = self._build_expand_prompt(task, node, num_expansions)
            expand_resp = await self.client.chat(
                [Message(role="user", content=expand_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, expand_resp.usage)
            all_steps.append(
                AgentStep(
                    step_number=iteration,
                    response=expand_resp,
                )
            )
            candidates = parse_numbered_items(expand_resp.text, num_expansions)

            # EVALUATE
            scored: list[tuple[str, float]] = []
            for cand in candidates:
                eval_prompt = _EVALUATE_PROMPT.format(task=task, approach=cand)
                eval_resp = await self.client.chat(
                    [Message(role="user", content=eval_prompt)],
                    system=system,
                    **kwargs,
                )
                total_usage = _accumulate_usage(total_usage, eval_resp.usage)
                score = parse_score(eval_resp.text)
                scored.append((cand, score))

            for cand, score in scored:
                child = _TreeNode(
                    state=cand,
                    value=score,
                    visits=0,
                    parent=node,
                    action=cand,
                    depth=node.depth + 1,
                )
                node.children.append(child)

            if not node.children:
                _fire_event(
                    self.config,
                    "step_end",
                    step_number=iteration,
                )
                continue

            # SIMULATE
            best_child = max(node.children, key=lambda n: n.value)
            inner_config = AgentConfig(
                max_iterations=3,
                system=best_child.state,
                max_tokens=self.config.max_tokens,
            )
            inner = ReActAgent(self.client, self.tools, config=inner_config)
            inner_result = await inner.async_run(task, **kwargs)
            total_usage = _accumulate_usage(total_usage, inner_result.total_usage)
            all_steps.extend(inner_result.steps)

            sim_score = (
                evaluator(inner_result.answer)
                if evaluator
                else _evaluate_result(inner_result.answer)
            )
            best_child.is_terminal = True

            if sim_score >= 0.5:
                best_child.is_success = True
                best_child.state = inner_result.answer
                _backpropagate(best_child, sim_score)
                if best_terminal is None or sim_score > best_terminal.value:
                    best_terminal = best_child
            else:
                reflect_prompt = self._build_reflect_prompt(
                    task, best_child.action, inner_result.answer
                )
                reflect_resp = await self.client.chat(
                    [
                        Message(
                            role="user",
                            content=reflect_prompt,
                        )
                    ],
                    system=system,
                    **kwargs,
                )
                total_usage = _accumulate_usage(total_usage, reflect_resp.usage)
                best_child.reflection = reflect_resp.text
                _backpropagate(best_child, sim_score)
                _fire_event(
                    self.config,
                    "reflection",
                    step_number=iteration,
                    result=reflect_resp.text,
                )

            _fire_event(
                self.config,
                "step_end",
                step_number=iteration,
            )

        if best_terminal:
            answer = best_terminal.state
        elif self._check_timeout(start):
            answer = "[timeout exceeded]"
        else:
            answer = "[no solution found]"
        return AgentResult(
            answer=answer,
            steps=tuple(all_steps),
            total_usage=total_usage,
        )


def _backpropagate(node: _TreeNode, value: float) -> None:
    """Update value and visits from node back to root."""
    current: _TreeNode | None = node
    while current is not None:
        current.visits += 1
        current.value += value
        current = current.parent


def _collect_reflections(node: _TreeNode) -> str:
    """Gather all reflections from ancestors."""
    reflections: list[str] = []
    current: _TreeNode | None = node
    while current is not None:
        if current.reflection:
            reflections.append(current.reflection)
        current = current.parent
    return "\n".join(reversed(reflections))


def _evaluate_result(answer: str) -> float:
    """Simple heuristic evaluation of an answer."""
    if not answer or answer.startswith("["):
        return 0.0
    return 0.7
