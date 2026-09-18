"""25 — Server Tools (Web Search, Code Execution).

Server tools are executed by the LLM provider's infrastructure, not
locally. They enable capabilities like web search and code interpretation
without needing external API keys.

Anthropic and Gemini run web search and code execution, and Meta runs web
search; the OpenAI (Chat Completions) and xAI adapters raise RequestError.
"""

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import web_search

llm = LLM("claude-haiku-4-5")

# Pass web_search() as a tool — the provider handles execution
result = llm.complete_sync(
    "What were the top tech news stories this week?",
    tools=[web_search()],
)

print("Answer:", result.text)
print(f"Tokens: {result.usage.input_tokens + result.usage.output_tokens}")
