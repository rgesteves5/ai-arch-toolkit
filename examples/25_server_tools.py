"""25 — Server Tools (Web Search, Code Execution).

Server tools are executed by the LLM provider's infrastructure, not
locally. They enable capabilities like web search and code interpretation
without needing external API keys.

Supported by Anthropic (web search) and OpenAI (web search, code execution).
"""

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import web_search

llm = LLM("gpt-4.1-nano")

# Pass web_search() as a tool — the provider handles execution
result = llm.complete_sync(
    "What were the top tech news stories this week?",
    tools=[web_search()],
)

print("Answer:", result.text)
print(f"Tokens: {result.usage.input_tokens + result.usage.output_tokens}")
