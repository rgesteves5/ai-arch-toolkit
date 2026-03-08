"""36 — Fallback Chains & Attempt Tracking.

Demonstrates the full fallback chain and attempt tracking features:

1. **Fallback chains** — multiple fallback LLMs (strings or instances),
   each with independent provider, retry, middleware, and temperature.
2. **Attempt tracking** — every LLM call (primary, retry, fallback) is
   recorded on ``Response.attempts`` so you can inspect what happened.
3. **Flow-level trace** — ``FlowResult.trace`` aggregates cost and timing
   across all flow steps.
4. **Custom fallback_on** — control which error types trigger fallback.

Requires: OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY / XAI_API_KEY).
"""

from ai_arch_toolkit import LLM, Attempt, State, ToolGroup
from ai_arch_toolkit.core import APIError, RetryConfig
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.tools import datetime_now, math_eval

# =====================================================================
# 1. Basic fallback chain — string shorthand
# =====================================================================

print("=" * 60)
print("1. BASIC FALLBACK CHAIN")
print("=" * 60)

llm = LLM(
    "gpt-4.1-nano",
    fallback="gpt-4.1-mini",  # single string fallback (backward compatible)
)
print(f"LLM: {llm!r}\n")

response = llm.complete_sync("What is 2 + 2? Reply in one word.")
print(f"Answer: {response.text}")
print(f"Attempts: {len(response.attempts)}")
for a in response.attempts:
    print(f"  - model={a.model} status={a.status} duration={a.duration:.3f}s")

# =====================================================================
# 2. Multi-model fallback chain with independent configs
# =====================================================================

print(f"\n{'=' * 60}")
print("2. MULTI-MODEL FALLBACK CHAIN")
print("=" * 60)

# Each fallback LLM is fully independent — own provider, retry, temperature
fast_fallback = LLM("gpt-4.1-nano", temperature=0.0)
creative_fallback = LLM("gpt-4.1-mini", temperature=0.7)

llm = LLM(
    "gpt-4.1-nano",
    temperature=0.0,
    fallback=[fast_fallback, creative_fallback],
)
print(f"LLM: {llm!r}\n")

response = llm.complete_sync("Name one color. Reply in one word.")
print(f"Answer: {response.text}")
print(f"Model used: {response.model}")
print(f"Total attempts: {len(response.attempts)}")

# =====================================================================
# 3. Fallback with retry — retry exhaustion triggers fallback
# =====================================================================

print(f"\n{'=' * 60}")
print("3. FALLBACK + RETRY CONFIG")
print("=" * 60)

llm = LLM(
    "gpt-4.1-nano",
    retry=RetryConfig(max_retries=2, base_delay=0.5),
    fallback="gpt-4.1-mini",
)
print(f"LLM: {llm!r}")
print("If primary fails, it retries 2 times, then tries the fallback.\n")

response = llm.complete_sync("What is the speed of light? One sentence.")
print(f"Answer: {response.text}")
print(f"Attempts: {len(response.attempts)}")

# =====================================================================
# 4. Attempt tracking — inspecting the full attempt history
# =====================================================================

print(f"\n{'=' * 60}")
print("4. ATTEMPT TRACKING IN DETAIL")
print("=" * 60)

llm = LLM("gpt-4.1-nano", fallback="gpt-4.1-mini")
response = llm.complete_sync("What is Python? One sentence.")


def print_attempts(attempts: tuple[Attempt, ...]) -> None:
    """Pretty-print attempt history."""
    for i, a in enumerate(attempts):
        status_icon = "OK" if a.status == "ok" else "FAIL"
        print(f"  [{i}] {status_icon} model={a.model}")
        print(f"      duration={a.duration:.4f}s  timestamp={a.timestamp:.0f}")
        print(f"      retry_number={a.retry_number}")
        if a.usage:
            print(f"      tokens: in={a.usage.input_tokens} out={a.usage.output_tokens}")
        if a.error:
            print(f"      error={a.error_type}: {a.error}")


print(f"Response: {response.text[:80]}...")
print(f"\nAttempt history ({len(response.attempts)} attempts):")
print_attempts(response.attempts)

# =====================================================================
# 5. Streaming with fallback + attempts
# =====================================================================

print(f"\n{'=' * 60}")
print("5. STREAMING WITH FALLBACK + ATTEMPTS")
print("=" * 60)

llm = LLM("gpt-4.1-nano", fallback="gpt-4.1-mini")

print("Streaming: ", end="")
stream = llm.stream_sync("Count from 1 to 5, separated by commas.")
for chunk in stream:
    print(chunk, end="", flush=True)
print()

resp = stream.response
print(f"\nStream finalized — {len(resp.attempts)} attempt(s):")
print_attempts(resp.attempts)

# =====================================================================
# 6. Flow with attempt tracking — trace
# =====================================================================

print(f"\n{'=' * 60}")
print("6. FLOW-LEVEL TRACE")
print("=" * 60)

tools = ToolGroup(datetime_now, math_eval)
llm = LLM("gpt-4.1-nano", fallback="gpt-4.1-mini")

flow = react_flow(
    llm,
    tools,
    system="Use tools to answer. Be concise.",
    max_iterations=5,
)

state = State(operational=react_initial_state("What is 42 * 17? Use the calculator tool."))
result = flow.run_sync(state)
print(f"Answer: {state['response'].text}")
print(f"Trace steps: {len(result.trace.steps)}")
print(f"Total cost: ${result.total_cost:.6f}")
print(f"Total duration: {result.total_duration:.2f}s")

# =====================================================================
# 7. Custom fallback_on — control which errors trigger fallback
# =====================================================================

print(f"\n{'=' * 60}")
print("7. CUSTOM fallback_on")
print("=" * 60)

# Only fall back on APIError and TimeoutError (not ConnectionError/OSError)
llm = LLM(
    "gpt-4.1-nano",
    fallback="gpt-4.1-mini",
    fallback_on=(APIError, TimeoutError),
)
print(f"fallback_on: {llm._fallback_on}")
print("Only APIError and TimeoutError will trigger fallback.")
print("ConnectionError and OSError will propagate immediately.\n")

response = llm.complete_sync("Say hello in Japanese. One word.")
print(f"Answer: {response.text}")
print(f"Attempts: {len(response.attempts)}")

# =====================================================================
# 8. Nested fallbacks are flattened
# =====================================================================

print(f"\n{'=' * 60}")
print("8. NESTED FALLBACK FLATTENING")
print("=" * 60)

inner = LLM("gpt-4.1-mini", fallback="gpt-4.1-nano")
outer = LLM("gpt-4.1-nano", fallback=inner)

print(f"outer: {outer!r}")
print(f"Chain length: {len(outer._fallbacks)} (flattened from nested)")
print(f"Chain: {[fb._model for fb in outer._fallbacks]}")
print("Nested fallbacks are extracted into a flat chain for full observability.")
