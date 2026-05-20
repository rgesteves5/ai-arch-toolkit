"""Custom Inspect AI solvers wrapping ai-arch-toolkit flows."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from inspect_ai.solver import TaskState, solver

from ai_arch_toolkit.core import LLM, tool
from ai_arch_toolkit.core._content import user
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._generate_review import (
    generate_review_flow,
    generate_review_initial_state,
)
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.agents.flows._self_discovery import (
    self_discovery_flow,
    self_discovery_initial_state,
)
from ai_arch_toolkit.toolkit.tools._math import math_eval
from ai_arch_toolkit.toolkit.tools._python import python_repl

from ._table_parse import table_parse
from ._thinking_systems import load_thinking_systems, make_thinking_system_tool

logger = logging.getLogger(__name__)

# Thread pool for xAI/grok calls — gRPC needs its own event loop.
_XAI_EXECUTOR = ThreadPoolExecutor(max_workers=4)

SYSTEM = (
    "You are an expert problem solver. Read the question carefully. "
    "Think step by step, then give your final answer."
)

THINKING_SYSTEM = (
    "You are an expert problem solver with access to tools.\n\n"
    "## Workflow\n"
    "1. Call thinking_system() to browse available reasoning strategies.\n"
    "2. Pick the most relevant strategy and call "
    "thinking_system(ts_names=[...]) for detailed guidance.\n"
    "3. Apply the strategy. Use python_repl for booleans, lists, sorting, "
    "counting, and arithmetic. Use math_eval for single math expressions. "
    "Use think as a scratchpad.\n"
    "4. Give your final answer in a normal message (not a tool call). "
    "Use the prefix 'The answer is:' for your final answer."
)

SYSTEM_TS_ONLY = (
    "You are an expert problem solver with access to tools.\n\n"
    "## Workflow\n"
    "1. Call thinking_system() to browse available reasoning strategies.\n"
    "2. Pick the most relevant strategy and call "
    "thinking_system(ts_names=[...]) for detailed guidance.\n"
    "3. Apply the strategy step by step in your reasoning.\n"
    "4. Give your final answer in a normal message (not a tool call). "
    "Use the prefix 'The answer is:' for your final answer."
)

SYSTEM_PYEVAL_ONLY = (
    "You are an expert problem solver with access to tools.\n\n"
    "## Workflow\n"
    "1. Analyze the problem and decide what computation would help.\n"
    "2. Use python_repl to evaluate Python expressions: arithmetic, "
    "booleans, lists, sorting, counting, comparisons, comprehensions.\n"
    "   Examples: python_repl(\"sorted(['c','a','b'])\"), "
    'python_repl("len([x for x in range(10) if x % 2 == 0])"), '
    'python_repl("15 * 3 + 42 // 6").\n'
    "3. Give your final answer in a normal message (not a tool call). "
    "Use the prefix 'The answer is:' for your final answer."
)

SYSTEM_TS_PYEVAL = (
    "You are an expert problem solver with access to tools.\n\n"
    "## Workflow\n"
    "1. Call thinking_system() to browse available reasoning strategies.\n"
    "2. Pick the most relevant strategy and call "
    "thinking_system(ts_names=[...]) for detailed guidance.\n"
    "3. Apply the strategy. Use python_repl for any computation: "
    "arithmetic, booleans, lists, sorting, counting, comparisons, "
    "comprehensions.\n"
    "   Examples: python_repl(\"sorted(['c','a','b'])\"), "
    'python_repl("not (True) and (True or False)"), '
    'python_repl("15 * 3 + 42 // 6").\n'
    "4. Give your final answer in a normal message (not a tool call). "
    "Use the prefix 'The answer is:' for your final answer."
)


SYSTEM_FULL = (
    "You are an expert problem solver. Use tools — never do math or logic in your head.\n\n"
    "## Tools\n"
    "- **python_repl(code)**: Multi-line Python. Variables, loops, string methods, "
    "list/dict ops, regex, math. Use for ALL computation.\n"
    "- **thinking_system()**: Browse reasoning strategies. Call with ts_names=['name'] "
    "for detailed guidance on a specific strategy.\n"
    "- **table_parse(data)**: Parse column-major bracket data into structured table.\n"
    "- **think(thought)**: Scratchpad.\n\n"
    "## Quick Guide\n"
    "- Swaps/tracking → python_repl with dict\n"
    "- Sorting/counting/boolean/arithmetic → python_repl\n"
    "- Tables → table_parse then python_repl\n"
    "- Complex reasoning → thinking_system first\n"
    "- Simple knowledge → answer directly\n\n"
    "## Rules\n"
    "1. Be efficient — use 1-3 tool calls max, not more.\n"
    "2. You MUST end with 'The answer is: X' (not in a tool call).\n"
    "3. Give ONLY the answer after the prefix — no explanation."
)


def _make_llm(model: str, max_completion_tokens: int = 16384) -> LLM:
    """Create an LLM with the right token parameter for the model family."""
    is_new_openai = model.startswith(("gpt-5", "o3", "o4"))
    is_anthropic = model.startswith("claude-")
    is_grok = model.startswith("grok-")

    # Temperature: gpt-5-nano forces 1.0, reasoning models ignore it
    if model.startswith("gpt-5"):
        temp = 1.0  # gpt-5 family only supports temperature=1
    elif model.startswith(("o3", "o4")):
        temp = 1.0  # reasoning models use default
    else:
        temp = 0.0

    llm = LLM(model, temperature=temp)

    # Token limit parameter varies by provider
    if is_new_openai:
        llm._defaults.pop("max_tokens", None)
        llm._defaults["max_completion_tokens"] = max_completion_tokens
    elif is_anthropic or is_grok:
        llm._defaults["max_tokens"] = max_completion_tokens
    # Gemini uses max_tokens by default, which is already set

    return llm


def _is_grok(model: str) -> bool:
    return model.startswith("grok-")


def _run_in_thread(coro_fn, *args):
    """Run an async function in a separate thread with its own event loop.

    Used for xAI/grok: the gRPC SDK creates its own asyncio event loop,
    which conflicts with Inspect AI's loop. By running in a fresh thread,
    gRPC gets an isolated loop.
    """

    def _target():
        return asyncio.run(coro_fn(*args))

    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_XAI_EXECUTOR, _target)


async def _grok_baseline_complete(
    model: str,
    question: str,
    system: str,
    extra_kwargs: dict[str, object],
):
    """Create a fresh LLM and complete — called inside executor thread."""
    llm = _make_llm(model)
    return await llm.complete([user(question)], system=system, **extra_kwargs)


async def _grok_react_run(
    model: str,
    question: str,
    tools: ToolGroup,
    system: str,
    max_iterations: int,
):
    """Create a fresh LLM + flow and run — called inside executor thread."""
    llm = _make_llm(model)
    flow = react_flow(llm, tools, system=system, max_iterations=max_iterations)
    flow_state = State(operational=react_initial_state(question))
    result = await flow.run(flow_state)
    response = flow_state.get("response")
    answer = response.text if response else ""
    return answer, result.total_cost


async def _grok_self_discovery_run(
    model: str,
    question: str,
    tools: ToolGroup,
    system: str,
    flow_kwargs: dict[str, object] | None = None,
):
    """Create a fresh LLM + self_discovery flow — called inside executor thread."""
    llm = _make_llm(model)
    flow = self_discovery_flow(llm, tools, system=system, **(flow_kwargs or {}))
    flow_state = State(operational=self_discovery_initial_state(question))
    result = await flow.run(flow_state)
    response = flow_state.get("response")
    answer = response.text if response else ""
    return answer, result.total_cost


async def _react_solve(
    state: TaskState,
    model: str,
    llm: LLM,
    tools: ToolGroup,
    system: str,
    max_iterations: int,
    solver_name: str,
    *,
    strip_tools_on_final: bool = False,
    show_turn_counter: bool = False,
) -> TaskState:
    """Shared logic for all ReAct-based solvers, with grok thread isolation."""
    question = state.input_text
    try:
        if _is_grok(model):
            answer, cost = await _run_in_thread(
                _grok_react_run, model, question, tools, system, max_iterations
            )
        else:
            flow = react_flow(
                llm,
                tools,
                system=system,
                max_iterations=max_iterations,
                strip_tools_on_final=strip_tools_on_final,
                show_turn_counter=show_turn_counter,
            )
            flow_state = State(operational=react_initial_state(question))
            result = await flow.run(flow_state)
            response = flow_state.get("response")
            answer = response.text if response else ""
            cost = result.total_cost
        state.output = _make_output(answer)
        state.metadata["cost"] = cost
    except Exception as exc:
        logger.warning("%s error: %s", solver_name, exc)
        state.output = _make_output(f"Error: {exc}")
        state.metadata["cost"] = 0.0
    return state


@tool
def think(thought: str) -> str:
    """Use as a scratchpad for intermediate reasoning steps.

    Args:
        thought: Your working notes, intermediate steps, or scratch calculations.
    """
    return ""


@solver
def baseline_solver(model: str = "gpt-5-nano", thinking: bool = False, **kwargs: object):
    """Raw LLM call — no tools, no orchestration."""
    llm = _make_llm(model)
    use_thread = _is_grok(model)

    async def solve(state: TaskState, generate) -> TaskState:
        question = state.input_text
        try:
            extra_kwargs: dict[str, object] = {}
            if thinking and model.startswith("claude-"):
                extra_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
            if use_thread:
                response = await _run_in_thread(
                    _grok_baseline_complete, model, question, SYSTEM, extra_kwargs
                )
            else:
                response = await llm.complete([user(question)], system=SYSTEM, **extra_kwargs)
            state.output = _make_output(response.text)
            state.metadata["cost"] = response.cost or 0.0
        except Exception as exc:
            logger.warning("baseline_solver error: %s", exc)
            state.output = _make_output(f"Error: {exc}")
            state.metadata["cost"] = 0.0
        return state

    return solve


def _catalog_to_modules(catalog: dict[str, dict[str, str]]) -> tuple[str, ...]:
    """Convert a thinking systems catalog into Self-Discovery module descriptions."""
    return tuple(
        f"{name.replace('_', ' ').title()}: {entry['summary']}" for name, entry in catalog.items()
    )


@solver
def self_discovery_solver(
    model: str = "gpt-5-nano",
    thinking_systems_path: str | None = None,
    use_ts_modules: bool = False,
    **kwargs: object,
):
    """Self-discovery flow — select, adapt, operationalize, solve.

    Args:
        model: Model name.
        thinking_systems_path: Optional path to custom thinking systems YAML.
        use_ts_modules: When True, use thinking systems as Self-Discovery modules
            instead of the abstract defaults. The concrete strategies become what
            SELECT picks from, so the full pipeline (select, adapt, operationalize)
            works with domain-specific reasoning.
    """
    llm = _make_llm(model)
    catalog = load_thinking_systems(thinking_systems_path)
    ts_tool = make_thinking_system_tool(catalog)
    tools = ToolGroup(think, ts_tool, python_repl)
    use_thread = _is_grok(model)

    flow_kwargs: dict[str, object] = {}
    if use_ts_modules:
        flow_kwargs["modules"] = _catalog_to_modules(catalog)

    async def solve(state: TaskState, generate) -> TaskState:
        question = state.input_text
        try:
            if use_thread:
                answer, cost = await _run_in_thread(
                    _grok_self_discovery_run,
                    model,
                    question,
                    tools,
                    SYSTEM_TS_PYEVAL,
                    flow_kwargs or None,
                )
            else:
                flow = self_discovery_flow(llm, tools, system=SYSTEM_TS_PYEVAL, **flow_kwargs)
                flow_state = State(operational=self_discovery_initial_state(question))
                result = await flow.run(flow_state)
                response = flow_state.get("response")
                answer = response.text if response else ""
                cost = result.total_cost
            state.output = _make_output(answer)
            state.metadata["cost"] = cost
        except Exception as exc:
            logger.warning("self_discovery_solver error: %s", exc)
            state.output = _make_output(f"Error: {exc}")
            state.metadata["cost"] = 0.0
        return state

    return solve


@solver
def react_tools_solver(
    model: str = "gpt-5-nano",
    max_iterations: int = 8,
    thinking_systems_path: str | None = None,
    **kwargs: object,
):
    """ReAct loop with think + math_eval + thinking_system tools."""
    llm = _make_llm(model)
    catalog = load_thinking_systems(thinking_systems_path)
    ts_tool = make_thinking_system_tool(catalog)
    tools = ToolGroup(think, math_eval, python_repl, ts_tool)

    async def solve(state: TaskState, generate) -> TaskState:
        return await _react_solve(
            state, model, llm, tools, THINKING_SYSTEM, max_iterations, "react_tools_solver"
        )

    return solve


@solver
def react_ts_only_solver(
    model: str = "gpt-5-nano",
    max_iterations: int = 8,
    thinking_systems_path: str | None = None,
    **kwargs: object,
):
    """ReAct loop with thinking_system tool only."""
    llm = _make_llm(model)
    catalog = load_thinking_systems(thinking_systems_path)
    ts_tool = make_thinking_system_tool(catalog)
    tools = ToolGroup(ts_tool)

    async def solve(state: TaskState, generate) -> TaskState:
        return await _react_solve(
            state, model, llm, tools, SYSTEM_TS_ONLY, max_iterations, "react_ts_only_solver"
        )

    return solve


@solver
def react_pyeval_only_solver(
    model: str = "gpt-5-nano",
    max_iterations: int = 8,
    **kwargs: object,
):
    """ReAct loop with python_repl tool only."""
    llm = _make_llm(model)
    tools = ToolGroup(python_repl)

    async def solve(state: TaskState, generate) -> TaskState:
        return await _react_solve(
            state,
            model,
            llm,
            tools,
            SYSTEM_PYEVAL_ONLY,
            max_iterations,
            "react_pyeval_only_solver",
        )

    return solve


@solver
def react_ts_pyeval_solver(
    model: str = "gpt-5-nano",
    max_iterations: int = 8,
    thinking_systems_path: str | None = None,
    strip_tools_on_final: bool = True,
    **kwargs: object,
):
    """ReAct loop with thinking_system + python_repl tools."""
    llm = _make_llm(model)
    catalog = load_thinking_systems(thinking_systems_path)
    ts_tool = make_thinking_system_tool(catalog)
    tools = ToolGroup(ts_tool, python_repl)

    async def solve(state: TaskState, generate) -> TaskState:
        return await _react_solve(
            state,
            model,
            llm,
            tools,
            SYSTEM_TS_PYEVAL,
            max_iterations,
            "react_ts_pyeval_solver",
            strip_tools_on_final=strip_tools_on_final,
        )

    return solve


@solver
def react_full_solver(
    model: str = "gpt-5-nano",
    max_iterations: int = 8,
    thinking_systems_path: str | None = None,
    **kwargs: object,
):
    """Full ReAct with all tools: thinking_system + python_repl + table_parse + think."""
    llm = _make_llm(model)
    catalog = load_thinking_systems(thinking_systems_path)
    ts_tool = make_thinking_system_tool(catalog)
    tools = ToolGroup(ts_tool, python_repl, table_parse, think)

    async def solve(state: TaskState, generate) -> TaskState:
        return await _react_solve(
            state, model, llm, tools, SYSTEM_FULL, max_iterations, "react_full_solver"
        )

    return solve


SYSTEM_GEN_REVIEW_GEN = (
    "You are an expert problem solver with access to a python_repl tool.\n\n"
    "## Workflow\n"
    "1. Analyze the problem and decide what computation would help.\n"
    "2. Use python_repl to evaluate Python expressions: arithmetic, "
    "booleans, lists, sorting, counting, comparisons, comprehensions.\n"
    "3. Give your final answer in a normal message (not a tool call). "
    "Use the prefix 'The answer is:' for your final answer."
)

SYSTEM_GEN_REVIEW_REVIEW = (
    "You are a careful reviewer with access to a python_repl tool.\n\n"
    "Your job is to verify the proposed answer by independently computing it.\n"
    "Use python_repl to check the work. Then respond with:\n"
    "- ACCEPT if the answer is correct.\n"
    "- RETRY followed by what went wrong and what the correct answer should be."
)


@solver
def generate_review_solver(
    model: str = "gpt-5-nano",
    max_cycles: int = 3,
    max_gen_iterations: int = 5,
    max_review_iterations: int = 5,
    **kwargs: object,
):
    """Generate-Review loop: generate with python_repl, review with python_repl."""
    llm = _make_llm(model)
    tools = ToolGroup(python_repl)

    async def solve(state: TaskState, generate) -> TaskState:
        question = state.input_text
        try:
            flow = generate_review_flow(
                gen_llm=llm,
                review_llm=llm,
                gen_tools=tools,
                review_tools=tools,
                gen_system=SYSTEM_GEN_REVIEW_GEN,
                review_system=SYSTEM_GEN_REVIEW_REVIEW,
                max_cycles=max_cycles,
                max_gen_iterations=max_gen_iterations,
                max_review_iterations=max_review_iterations,
            )
            flow_state = State(operational=generate_review_initial_state(question))
            result = await flow.run(flow_state)
            answer = flow_state.get("answer", "")
            response = flow_state.get("response")
            if not answer and response:
                answer = response.text
            cost = result.total_cost
            state.output = _make_output(answer)
            state.metadata["cost"] = cost
        except Exception as exc:
            logger.warning("generate_review_solver error: %s", exc)
            state.output = _make_output(f"Error: {exc}")
            state.metadata["cost"] = 0.0
        return state

    return solve


def _make_output(text: str):
    """Create a ModelOutput from text."""
    from inspect_ai.model import ChatMessageAssistant, ModelOutput
    from inspect_ai.model._model_output import ChatCompletionChoice

    return ModelOutput(
        model="ai-arch-toolkit",
        choices=[
            ChatCompletionChoice(message=ChatMessageAssistant(content=text), stop_reason="stop")
        ],
        completion=text,
    )
