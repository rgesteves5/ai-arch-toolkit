"""07 — Thinking / Extended Reasoning (Anthropic).

Extended Thinking lets the model reason step-by-step before answering.
Claude 4.5 models use budget_tokens to control reasoning depth.
Claude 4.6 models also support adaptive effort levels ("low"/"medium"/"high").

Important: Anthropic requires max_tokens > budget_tokens. The provider
defaults to max_tokens=4096, so pass a larger max_tokens when using
big thinking budgets.
"""

from ai_arch_toolkit import Client, ThinkingConfig

client = Client("anthropic", model="claude-haiku-4-5")

# --- Extended thinking with a small budget ---
print("=== Extended Thinking (budget_tokens=2048) ===")
resp = client.chat(
    "What are the philosophical implications of Gödel's incompleteness theorems?",
    thinking=ThinkingConfig(budget_tokens=2048),
)
if resp.thinking:
    print(f"[Thinking ({len(resp.thinking)} chars)]: {resp.thinking[:200]}...")
print("\nAnswer:", resp.text[:300], "...\n")

# --- Extended thinking with a larger budget ---
# Note: max_tokens must be greater than budget_tokens (default max_tokens is 4096)
print("=== Extended Thinking (budget_tokens=4096) ===")
resp2 = client.chat(
    "Solve step by step: If a train travels 120 km in 1.5 hours, "
    "then stops for 30 minutes, then travels 80 km in 1 hour, "
    "what is the average speed for the entire journey?",
    thinking=ThinkingConfig(budget_tokens=4096),
    max_tokens=8192,
)
if resp2.thinking:
    print(f"[Thinking ({len(resp2.thinking)} chars)]: {resp2.thinking[:200]}...")
print("\nAnswer:", resp2.text)
