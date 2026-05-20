"""Mini test: verify python_repl works with Gemini Flash Lite and Claude Haiku.

Runs a simple BBEH-style problem through react_flow with python_repl to check
that tool results are actually visible to the model (not "None").

Usage:
    set -a && source .env && set +a
    uv run python scripts/test_repl_models.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback

from ai_arch_toolkit.core import LLM
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.tools._python import python_repl

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM = (
    "You are an expert problem solver with access to a python_repl tool.\n"
    "Use python_repl for ALL computation. Never compute in your head.\n"
    "End with 'The answer is: X' (not in a tool call)."
)

# Simple problems where the model MUST use python_repl to get the right answer
PROBLEMS = [
    {
        "question": "What is sorted(['banana', 'apple', 'cherry'])? Give the sorted list.",
        "expected": "apple",  # substring check
    },
    {
        "question": (
            "Alice, Bob, and Claire are playing a game. Alice has a ball, Bob has a hat, "
            "Claire has a shoe. Alice and Bob swap items. What does Bob have now?"
        ),
        "expected": "ball",
    },
    {
        "question": "What is 17 * 23 + 42 - 15 * 3?",
        "expected": "388",  # 391 + 42 - 45 = 388
    },
]

MODELS = [
    ("gemini-3.1-flash-lite-preview", 0.0),
    # ("claude-haiku-4-5-20251001", 0.0),  # billing blocked
]


async def test_model(model_name: str, temperature: float) -> None:
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")

    llm = LLM(model_name, temperature=temperature, max_tokens=4096)
    tools = ToolGroup(python_repl)
    flow = react_flow(
        llm,
        tools,
        system=SYSTEM,
        max_iterations=5,
    )

    for i, problem in enumerate(PROBLEMS, 1):
        print(f"\n--- Problem {i}: {problem['question'][:60]}...")
        state = State(operational=react_initial_state(problem["question"]))
        try:
            result = await flow.run(state)
            response = state.get("response")
            answer = response.text if response else "(no response)"
            history = state.get("history", [])

            # Show tool calls and results
            for msg in history:
                role = msg.get("role", "?")
                if role == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "tool_use":
                                print(
                                    f"  TOOL CALL: {part['name']}({json.dumps(part.get('input', {}))[:100]})"
                                )
                    elif isinstance(content, str) and content.strip():
                        print(f"  ASSISTANT: {content[:120]}")
                elif role == "tool":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "tool_result":
                                text = part.get("text", part.get("content", ""))
                                print(f"  TOOL RESULT: {str(text)[:120]}")
                    elif isinstance(content, str):
                        print(f"  TOOL RESULT: {content[:120]}")

            # Check answer
            has_expected = problem["expected"].lower() in answer.lower()
            status = "PASS" if has_expected else "FAIL"
            print(f"  ANSWER: {answer[:200]}")
            print(f"  STATUS: {status} (expected '{problem['expected']}' in answer)")
            print(f"  COST: ${result.total_cost:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()


async def main():
    # First verify python_repl works locally
    print("Local python_repl sanity check:")
    print(f"  python_repl('2+2') = {python_repl('2+2')!r}")
    print(f"  python_repl('x=5\\nprint(x)') = {python_repl('x=5\nprint(x)')!r}")
    print(f"  python_repl('sorted([3,1,2])') = {python_repl('sorted([3,1,2])')!r}")

    for model_name, temp in MODELS:
        try:
            await test_model(model_name, temp)
        except Exception as e:
            print(f"\nFATAL ERROR with {model_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
