"""The state keys the strategies and the agent runner share, and what each one holds.

A strategy's flow reads the task from ``TASK`` (ReAct and the single call read the conversation
from ``MESSAGES`` instead) and leaves its answer under ``ANSWER``, with the model response it came
from under ``RESPONSE``, also when it runs out of turns. The runner reads those two alone; a flow
that leaves no ``ANSWER`` answers with its last step's value (``extract_text``). The keys private
to one strategy (``feedback``, ``plan_text``, ``mcts_root``, …) stay in its module.
"""

from __future__ import annotations

from typing import Final

__all__ = ["ANSWER", "MESSAGES", "RESPONSE", "TASK"]

TASK: Final = "task"
"""The task, as text."""

MESSAGES: Final = "messages"
"""The conversation so far: a list of messages, the task first."""

ANSWER: Final = "answer"
"""The strategy's answer, as text (empty when the model gave none)."""

RESPONSE: Final = "response"
"""The model ``Response`` that ``ANSWER`` came from (``None`` when there was none)."""
