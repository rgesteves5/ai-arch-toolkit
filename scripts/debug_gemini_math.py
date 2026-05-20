"""Debug: trace the flow executor iteration by iteration."""

from __future__ import annotations

import asyncio
import time

from ai_arch_toolkit.core import LLM
from ai_arch_toolkit.core._state import State
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit.agents.flows._react import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.flow import _executor as executor_mod
from ai_arch_toolkit.toolkit.tools._python import python_repl

SYSTEM = (
    "You are an expert problem solver with access to a python_repl tool.\n"
    "Use python_repl for ALL computation. Never compute in your head.\n"
    "End with the exact text: The answer is: X"
)

# Monkey-patch _execute_sequential to trace iterations
_original = executor_mod._execute_sequential


async def _traced_sequential(flow, state, traces, results):
    has_conditions = any(fs.when is not None for fs in flow.steps)
    max_iter = flow.max_iterations
    print(f"\n[TRACE] Starting cyclic execution, max_iter={max_iter}")

    if not has_conditions:
        print("[TRACE] No conditions — pure sequential")
        return await _original(flow, state, traces, results)

    iteration = 0
    while max_iter is None or iteration < max_iter:
        any_executed = False
        print(f"\n[TRACE] === Iteration {iteration} ===")
        for fs in flow.steps:
            snapshot = state.snapshot()
            scoped = executor_mod._resolve_and_apply_scope(snapshot, fs, flow)

            if fs.when is not None:
                try:
                    condition_met = fs.when(scoped)
                except Exception as exc:
                    print(f"[TRACE]   {fs.step.name}: condition ERROR: {exc}")
                    traces.append(
                        executor_mod.StepTrace(
                            name=fs.step.name,
                            skipped=True,
                            skip_reason=f"condition error: {exc}",
                            started_at=time.monotonic(),
                        )
                    )
                    continue
                if not condition_met:
                    print(f"[TRACE]   {fs.step.name}: SKIPPED (condition not met)")
                    traces.append(
                        executor_mod.StepTrace(
                            name=fs.step.name,
                            skipped=True,
                            skip_reason="condition not met",
                            started_at=time.monotonic(),
                        )
                    )
                    continue

            print(f"[TRACE]   {fs.step.name}: EXECUTING...")
            result, step_trace = await executor_mod._run_step_with_scope(fs.step, scoped)
            traces.append(step_trace)
            any_executed = True
            if result is not None:
                results[fs.step.name] = result
                state.merge(result)
                arts = {k: repr(v)[:80] for k, v in result.artifacts.items()}
                print(f"[TRACE]   {fs.step.name}: DONE, artifacts={arts}")
                if result.is_error:
                    print(f"[TRACE]   {fs.step.name}: ERROR={result.error!r} - halting")
                    return

        if not any_executed:
            print(f"[TRACE] No steps executed in iteration {iteration} — breaking")
            break
        print(f"[TRACE] Iteration {iteration} complete, any_executed={any_executed}")
        iteration += 1

    print(f"[TRACE] Loop ended. Final iteration={iteration}")


async def run():
    llm = LLM("gemini-3.1-flash-lite-preview", temperature=0.0, max_tokens=4096)
    tools = ToolGroup(python_repl)
    flow = react_flow(
        llm,
        tools,
        system=SYSTEM,
        max_iterations=5,
        show_turn_counter=True,
        strip_tools_on_final=True,
    )

    state = State(operational=react_initial_state("What is 17 * 23 + 42 - 15 * 3?"))

    # Patch the executor
    executor_mod._execute_sequential = _traced_sequential

    result = await flow.run(state)

    # Restore
    executor_mod._execute_sequential = _original

    response = state.get("response")
    print("\n=== FINAL STATE ===")
    print(f"turn: {state.get('turn')}")
    print(f"needs_llm_call: {state.get('needs_llm_call')}")
    print(f"has_tool_calls: {state.get('has_tool_calls')}")
    msgs = state.get("messages", [])
    print(f"messages: {len(msgs)}")
    for i, m in enumerate(msgs):
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, str):
            print(f"  [{i}] {role}: {content[:100]}")
        elif isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "tool_use":
                        parts.append(f"tool_use({p['name']})")
                    elif p.get("type") == "tool_result":
                        parts.append(
                            f"tool_result({str(p.get('text', p.get('content', '')))[:50]})"
                        )
                    elif p.get("type") == "text":
                        parts.append(f"text({p.get('text', '')[:50]})")
                    else:
                        parts.append(str(p)[:50])
            print(f"  [{i}] {role}: {', '.join(parts)}")

    if response:
        print(f"\nresponse.text: {response.text!r}")
        print(f"response.tool_calls: {len(response.tool_calls)} calls")
        print(f"response.stop_reason: {response.stop_reason}")


asyncio.run(run())
