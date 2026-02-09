"""Tree of Thoughts agent implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.agents._base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    _accumulate_usage,
    _fire_event,
)
from ai_arch_toolkit.agents._parsing import parse_numbered_items, parse_score
from ai_arch_toolkit.llm._types import Message, Usage

_GENERATE_PROMPT = (
    "Given the current state of reasoning, generate {k} distinct "
    "next thoughts or steps to continue solving the problem. "
    "Number each thought (1., 2., etc.).\n\n"
    "Problem: {task}\n\n"
    "Current state:\n{state}"
)

_EVALUATE_PROMPT = (
    "Evaluate how promising this thought is for solving the problem. "
    "Return a score between 0.0 (terrible) and 1.0 (excellent). "
    "Return ONLY the numeric score.\n\n"
    "Problem: {task}\n\nThought:\n{thought}"
)


@dataclass
class _ThoughtNode:
    state: str
    score: float = 0.0
    depth: int = 0
    children: list[_ThoughtNode] = field(default_factory=list)
    parent: _ThoughtNode | None = field(default=None, repr=False)
    is_terminal: bool = False


class TreeOfThoughtsAgent(BaseAgent):
    """Tree of Thoughts agent using BFS or DFS tree search.

    1. Generate K candidate thoughts at each node.
    2. Score each via LLM-as-judge.
    3. Keep top N (beam width) for BFS or explore best for DFS.
    4. Recurse to max depth.
    5. Return the best terminal thought.
    """

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the Tree of Thoughts search."""
        max_depth = kwargs.pop("max_depth", 3)
        branching_factor = kwargs.pop("branching_factor", 3)
        beam_width = kwargs.pop("beam_width", 2)
        strategy = kwargs.pop("search_strategy", "bfs")
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        start = time.monotonic()

        root = _ThoughtNode(state="(start)", depth=0)
        best_node = root

        if strategy == "dfs":
            best_node, total_usage, all_steps = self._dfs(
                task,
                root,
                max_depth,
                branching_factor,
                system,
                total_usage,
                all_steps,
                best_node,
                start=start,
                **kwargs,
            )
        else:
            best_node, total_usage, all_steps = self._bfs(
                task,
                root,
                max_depth,
                branching_factor,
                beam_width,
                system,
                total_usage,
                all_steps,
                start=start,
                **kwargs,
            )

        return AgentResult(
            answer=best_node.state,
            steps=tuple(all_steps),
            total_usage=total_usage,
        )

    def _bfs(
        self,
        task: str,
        root: _ThoughtNode,
        max_depth: int,
        branching_factor: int,
        beam_width: int,
        system: str | None,
        total_usage: Usage,
        all_steps: list[AgentStep],
        *,
        start: float,
        **kwargs: Any,
    ) -> tuple[_ThoughtNode, Usage, list[AgentStep]]:
        """BFS tree search with beam width pruning."""
        current_level = [root]
        best_node = root

        for depth in range(1, max_depth + 1):
            if self._check_timeout(start):
                break
            _fire_event(self.config, "step_start", step_number=depth)
            next_level: list[_ThoughtNode] = []

            for node in current_level:
                children, usage, steps = self._expand_and_score(
                    task,
                    node,
                    branching_factor,
                    depth,
                    system,
                    **kwargs,
                )
                total_usage = _accumulate_usage(total_usage, usage)
                all_steps.extend(steps)
                next_level.extend(children)

            # Keep top beam_width
            next_level.sort(key=lambda n: n.score, reverse=True)
            current_level = next_level[:beam_width]

            if current_level and current_level[0].score > best_node.score:
                best_node = current_level[0]

            _fire_event(self.config, "step_end", step_number=depth)

        if self._check_timeout(start) and best_node.state == "(start)":
            return (
                _ThoughtNode(state="[timeout exceeded]"),
                total_usage,
                all_steps,
            )
        return best_node, total_usage, all_steps

    def _dfs(
        self,
        task: str,
        node: _ThoughtNode,
        max_depth: int,
        branching_factor: int,
        system: str | None,
        total_usage: Usage,
        all_steps: list[AgentStep],
        best_node: _ThoughtNode,
        *,
        start: float,
        **kwargs: Any,
    ) -> tuple[_ThoughtNode, Usage, list[AgentStep]]:
        """DFS tree search."""
        if node.depth >= max_depth or self._check_timeout(start):
            return best_node, total_usage, all_steps

        depth = node.depth + 1
        _fire_event(self.config, "step_start", step_number=depth)

        children, usage, steps = self._expand_and_score(
            task, node, branching_factor, depth, system, **kwargs
        )
        total_usage = _accumulate_usage(total_usage, usage)
        all_steps.extend(steps)

        _fire_event(self.config, "step_end", step_number=depth)

        # Sort children by score, explore best first
        children.sort(key=lambda n: n.score, reverse=True)
        for child in children:
            if child.score > best_node.score:
                best_node = child
            best_node, total_usage, all_steps = self._dfs(
                task,
                child,
                max_depth,
                branching_factor,
                system,
                total_usage,
                all_steps,
                best_node,
                start=start,
                **kwargs,
            )

        return best_node, total_usage, all_steps

    def _expand_and_score(
        self,
        task: str,
        node: _ThoughtNode,
        branching_factor: int,
        depth: int,
        system: str | None,
        **kwargs: Any,
    ) -> tuple[list[_ThoughtNode], Usage, list[AgentStep]]:
        """Generate and score child thoughts for a node."""
        total_usage = Usage()
        steps: list[AgentStep] = []

        # Generate thoughts
        gen_prompt = _GENERATE_PROMPT.format(k=branching_factor, task=task, state=node.state)
        gen_resp = self.client.chat(
            [Message(role="user", content=gen_prompt)],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, gen_resp.usage)
        steps.append(AgentStep(step_number=depth, response=gen_resp))

        thoughts = parse_numbered_items(gen_resp.text, branching_factor)
        children: list[_ThoughtNode] = []

        for thought in thoughts:
            # Score each thought
            eval_prompt = _EVALUATE_PROMPT.format(task=task, thought=thought)
            eval_resp = self.client.chat(
                [Message(role="user", content=eval_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, eval_resp.usage)
            score = parse_score(eval_resp.text)
            child = _ThoughtNode(
                state=thought,
                score=score,
                depth=depth,
                parent=node,
            )
            node.children.append(child)
            children.append(child)

        return children, total_usage, steps

    async def async_run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the Tree of Thoughts search asynchronously."""
        max_depth = kwargs.pop("max_depth", 3)
        branching_factor = kwargs.pop("branching_factor", 3)
        beam_width = kwargs.pop("beam_width", 2)
        strategy = kwargs.pop("search_strategy", "bfs")
        system = self.config.system or None
        total_usage = Usage()
        all_steps: list[AgentStep] = []
        start = time.monotonic()

        root = _ThoughtNode(state="(start)", depth=0)
        best_node = root

        if strategy == "dfs":
            best_node, total_usage, all_steps = await self._async_dfs(
                task,
                root,
                max_depth,
                branching_factor,
                system,
                total_usage,
                all_steps,
                best_node,
                start=start,
                **kwargs,
            )
        else:
            best_node, total_usage, all_steps = await self._async_bfs(
                task,
                root,
                max_depth,
                branching_factor,
                beam_width,
                system,
                total_usage,
                all_steps,
                start=start,
                **kwargs,
            )

        return AgentResult(
            answer=best_node.state,
            steps=tuple(all_steps),
            total_usage=total_usage,
        )

    async def _async_bfs(
        self,
        task: str,
        root: _ThoughtNode,
        max_depth: int,
        branching_factor: int,
        beam_width: int,
        system: str | None,
        total_usage: Usage,
        all_steps: list[AgentStep],
        *,
        start: float,
        **kwargs: Any,
    ) -> tuple[_ThoughtNode, Usage, list[AgentStep]]:
        current_level = [root]
        best_node = root

        for depth in range(1, max_depth + 1):
            if self._check_timeout(start):
                break
            _fire_event(self.config, "step_start", step_number=depth)
            next_level: list[_ThoughtNode] = []

            for node in current_level:
                children, usage, steps = await self._async_expand_and_score(
                    task,
                    node,
                    branching_factor,
                    depth,
                    system,
                    **kwargs,
                )
                total_usage = _accumulate_usage(total_usage, usage)
                all_steps.extend(steps)
                next_level.extend(children)

            next_level.sort(key=lambda n: n.score, reverse=True)
            current_level = next_level[:beam_width]

            if current_level and current_level[0].score > best_node.score:
                best_node = current_level[0]

            _fire_event(self.config, "step_end", step_number=depth)

        if self._check_timeout(start) and best_node.state == "(start)":
            return (
                _ThoughtNode(state="[timeout exceeded]"),
                total_usage,
                all_steps,
            )
        return best_node, total_usage, all_steps

    async def _async_dfs(
        self,
        task: str,
        node: _ThoughtNode,
        max_depth: int,
        branching_factor: int,
        system: str | None,
        total_usage: Usage,
        all_steps: list[AgentStep],
        best_node: _ThoughtNode,
        *,
        start: float,
        **kwargs: Any,
    ) -> tuple[_ThoughtNode, Usage, list[AgentStep]]:
        if node.depth >= max_depth or self._check_timeout(start):
            return best_node, total_usage, all_steps

        depth = node.depth + 1
        _fire_event(self.config, "step_start", step_number=depth)

        children, usage, steps = await self._async_expand_and_score(
            task,
            node,
            branching_factor,
            depth,
            system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, usage)
        all_steps.extend(steps)

        _fire_event(self.config, "step_end", step_number=depth)

        children.sort(key=lambda n: n.score, reverse=True)
        for child in children:
            if child.score > best_node.score:
                best_node = child
            best_node, total_usage, all_steps = await self._async_dfs(
                task,
                child,
                max_depth,
                branching_factor,
                system,
                total_usage,
                all_steps,
                best_node,
                start=start,
                **kwargs,
            )

        return best_node, total_usage, all_steps

    async def _async_expand_and_score(
        self,
        task: str,
        node: _ThoughtNode,
        branching_factor: int,
        depth: int,
        system: str | None,
        **kwargs: Any,
    ) -> tuple[list[_ThoughtNode], Usage, list[AgentStep]]:
        total_usage = Usage()
        steps: list[AgentStep] = []

        gen_prompt = _GENERATE_PROMPT.format(k=branching_factor, task=task, state=node.state)
        gen_resp = await self.client.chat(
            [Message(role="user", content=gen_prompt)],
            system=system,
            **kwargs,
        )
        total_usage = _accumulate_usage(total_usage, gen_resp.usage)
        steps.append(AgentStep(step_number=depth, response=gen_resp))

        thoughts = parse_numbered_items(gen_resp.text, branching_factor)
        children: list[_ThoughtNode] = []

        for thought in thoughts:
            eval_prompt = _EVALUATE_PROMPT.format(task=task, thought=thought)
            eval_resp = await self.client.chat(
                [Message(role="user", content=eval_prompt)],
                system=system,
                **kwargs,
            )
            total_usage = _accumulate_usage(total_usage, eval_resp.usage)
            score = parse_score(eval_resp.text)
            child = _ThoughtNode(
                state=thought,
                score=score,
                depth=depth,
                parent=node,
            )
            node.children.append(child)
            children.append(child)

        return children, total_usage, steps
