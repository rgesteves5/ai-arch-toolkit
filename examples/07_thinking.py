"""07 — Thinking / Extended Reasoning.

Extended thinking lets the model reason step-by-step before answering.
Control reasoning depth with thinking_budget (token count) or
thinking_effort (string level like "low", "medium", "high").
"""

from ai_arch_toolkit import LLM

llm = LLM("claude-haiku-4-5-20251001")

# --- Extended thinking with a token budget ---
print("=== Extended Thinking (thinking_budget=2048) ===")
resp = llm.complete_sync(
    "What are the philosophical implications of Gödel's incompleteness theorems?",
    thinking=True,
    thinking_budget=2048,
)
if resp.thinking:
    print(f"[Thinking ({len(resp.thinking[0].text)} chars)]: {resp.thinking[0].text}")
print("\nAnswer:", resp.text, "\n")

# --- Extended thinking with effort level ---
print("=== Extended Thinking (thinking_effort='medium') ===")
resp2 = llm.complete_sync(
    "Solve step by step: If a train travels 120 km in 1.5 hours, "
    "then stops for 30 minutes, then travels 80 km in 1 hour, "
    "what is the average speed for the entire journey?",
    thinking=True,
    thinking_effort="medium",
    max_tokens=16384,
)
if resp2.thinking:
    print(f"[Thinking ({len(resp2.thinking[0].text)} chars)]: {resp2.thinking[0].text}")
print("\nAnswer:", resp2.text)
