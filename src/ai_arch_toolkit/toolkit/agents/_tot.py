"""ToTAgent — Tree of Thoughts with DFS/BFS search."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from ai_arch_toolkit.core._content import Content, user
from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._response import Usage
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents._base import (
    AgentConfig,
    AgentEvent,
    BaseAgent,
    PhaseConfig,
    _add_usage,
    _resolve_llm,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_EVALUATOR_SYSTEM = (
    "You are a thought evaluator. Score the given thought on a scale of 0.0 to 1.0 "
    "for how likely it is to lead to a correct solution. Respond with ONLY a number."
)

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")
_ITEM_RE = re.compile(r"^\d+[.)]\s*(.+)", re.MULTILINE)


def _parse_numbered_items(text: str) -> list[str]:
    """Extract numbered list items from text."""
    items = _ITEM_RE.findall(text)
    return items if items else [line.strip() for line in text.strip().splitlines() if line.strip()]


def _parse_score(text: str) -> float:
    """Extract a 0-1 score from text."""
    m = _SCORE_RE.search(text)
    if m:
        return min(1.0, max(0.0, float(m.group(1))))
    return 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ToTConfig:
    """Configuration specific to the Tree of Thoughts agent."""

    n_candidates: int = 3
    max_depth: int = 3
    evaluator_system: str = _DEFAULT_EVALUATOR_SYSTEM
    strategy: Literal["dfs", "bfs"] = "dfs"
    generator: PhaseConfig | None = None
    evaluator: PhaseConfig | None = None
    solver: PhaseConfig | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ToTAgent(BaseAgent):
    """Tree of Thoughts: generate-evaluate-expand search.

    At each node, generates candidate next thoughts, evaluates them,
    and expands the best into the frontier. Supports DFS and BFS strategies.
    """

    __slots__ = ("tot",)

    def __init__(
        self,
        llm: LLM,
        tools: ToolGroup,
        *,
        config: AgentConfig | None = None,
        tot: ToTConfig | None = None,
    ) -> None:
        super().__init__(llm, tools, config=config)
        self.tot = tot or ToTConfig()

    async def _run_loop(self, task: Content, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        task_str = task if isinstance(task, str) else str(task)
        step_num = 0
        total_usage = Usage()
        start = time.monotonic()
        frontier: deque[tuple[str, int]] = deque()  # (state, depth)
        frontier.append((task_str, 0))
        gen_llm = _resolve_llm(self.tot.generator, self.llm)
        eval_llm = _resolve_llm(self.tot.evaluator, self.llm)
        solver_llm = _resolve_llm(self.tot.solver, self.llm)

        while frontier and step_num < self.config.max_iterations:
            # --- stop conditions ---
            if self._check_timeout(start):
                yield AgentEvent(type="step_end", step=step_num + 1, stop_reason="timeout")
                return
            if self._check_budget(total_usage):
                yield AgentEvent(
                    type="step_end", step=step_num + 1, stop_reason="budget_exhausted"
                )
                return

            # Select next node
            if self.tot.strategy == "dfs":
                state, depth = frontier.pop()
            else:
                state, depth = frontier.popleft()

            step_num += 1
            yield AgentEvent(type="step_start", step=step_num)

            if depth >= self.tot.max_depth:
                # At max depth, produce final answer from this state
                final_response = await solver_llm.complete(
                    [
                        user(
                            f"Based on the following reasoning, produce a final answer.\n\n"
                            f"Reasoning:\n{state}\n\nTask: {task_str}"
                        )
                    ],
                    output_schema=self.config.output_schema,
                    **self.config.llm_kwargs,
                )
                total_usage = _add_usage(total_usage, final_response.usage)
                yield AgentEvent(
                    type="step_end",
                    step=step_num,
                    response=final_response,
                    stop_reason="completed",
                )
                return

            # Generate candidates
            gen_response = await gen_llm.complete(
                [
                    user(
                        f"Generate {self.tot.n_candidates} candidate next thoughts for "
                        f"solving the task.\n\nTask: {task_str}\n\n"
                        f"Current reasoning:\n{state}\n\n"
                        f"List each candidate as a numbered item."
                    )
                ],
                system=self.config.system or None,
            )
            total_usage = _add_usage(total_usage, gen_response.usage)
            candidates = _parse_numbered_items(gen_response.text)

            # Evaluate candidates
            scored: list[tuple[float, str]] = []
            for candidate in candidates:
                eval_response = await eval_llm.complete(
                    [
                        user(
                            f"Task: {task_str}\n\n"
                            f"Current reasoning:\n{state}\n\n"
                            f"Candidate thought: {candidate}\n\n"
                            f"Score this thought 0.0 to 1.0."
                        )
                    ],
                    system=self.tot.evaluator_system,
                )
                total_usage = _add_usage(total_usage, eval_response.usage)
                score = _parse_score(eval_response.text)
                scored.append((score, candidate))

            yield AgentEvent(type="step_end", step=step_num, response=gen_response)

            # Check for high-confidence solution
            if scored:
                best_score, best_thought = max(scored)
                if best_score >= 0.9:
                    step_num += 1
                    yield AgentEvent(type="step_start", step=step_num)
                    final_response = await solver_llm.complete(
                        [
                            user(
                                f"Based on the following reasoning, produce a final answer.\n\n"
                                f"Reasoning:\n{state}\n{best_thought}\n\nTask: {task_str}"
                            )
                        ],
                        output_schema=self.config.output_schema,
                        **self.config.llm_kwargs,
                    )
                    total_usage = _add_usage(total_usage, final_response.usage)
                    yield AgentEvent(
                        type="step_end",
                        step=step_num,
                        response=final_response,
                        stop_reason="completed",
                    )
                    return

                # Expand top candidates into frontier
                for _score, thought in sorted(scored, reverse=True)[: self.tot.n_candidates]:
                    new_state = f"{state}\n{thought}"
                    frontier.append((new_state, depth + 1))

        # Max iterations exhausted
        yield AgentEvent(
            type="step_end",
            step=step_num,
            stop_reason="max_iterations",
        )
