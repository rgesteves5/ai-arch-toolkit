"""49 — Local Web Search (a local model searching with Brave or Tavily).

A model on a local OpenAI-compatible server (Ollama here) searches the web with a toolkit tool:
``brave_search`` or ``tavily_search``, whichever has a key in the environment. The toolkit runs
the search, not the model's provider, so any model that calls tools can use it. Each search is
billed by the service on your key, and the run's meter counts it at the price table's
``[tools]`` entry: Brave $0.005 a search, Tavily $0.008 (a basic search, one credit).

Needs a local server with a model that calls tools (``ollama pull qwen3:8b``) and
BRAVE_SEARCH_API_KEY or TAVILY_API_KEY. Without either, the script says what is missing and
stops, without spending anything.
"""

import os
import urllib.error
import urllib.request

from ai_arch_toolkit import LLM, BudgetPolicy, ModelPricing, ToolGroup, pricing
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec
from ai_arch_toolkit.toolkit.tools import brave_search, tavily_search

BASE_URL = "http://localhost:11434/v1"  # Ollama; LM Studio serves http://localhost:1234/v1
MODEL = "qwen3:8b"  # any local model that calls tools


def search_tool():
    """The search tool that has a key in the environment, Brave first; ``None`` without one."""
    if os.environ.get("BRAVE_SEARCH_API_KEY", "").strip():
        return brave_search
    if os.environ.get("TAVILY_API_KEY", "").strip():
        return tavily_search
    return None


def server_answers() -> bool:
    """Whether the local server lists its models."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def main() -> None:
    search = search_tool()
    if search is None:
        print("Skipped: set BRAVE_SEARCH_API_KEY or TAVILY_API_KEY to search the web.")
        return
    if not server_answers():
        print(f"Skipped: no local server answers at {BASE_URL} (start Ollama, or edit BASE_URL).")
        return

    # A local model costs nothing, but a metered run refuses a model it cannot price.
    pricing.register(MODEL, ModelPricing())
    llm = LLM(MODEL, base_url=BASE_URL)  # a loopback server needs no API key
    agent = Agent(ReasoningSpec(strategy="react", max_iterations=4), llm, ToolGroup(search))

    # The budget bounds what the searches cost: at most three, and five cents.
    result = agent.run_sync(
        "What is the latest stable release of Python, and when did it come out? "
        "Search the web, and name the page you read it on.",
        budget_policy=BudgetPolicy(max_tool_calls=3, max_cost=0.05),
    )

    print(f"Searched with {search.__name__} through {MODEL}.")
    print("Answer:", result.text)
    if result.report is not None:
        print(f"Searches: {result.report.tool_calls}, cost: ${result.report.cost:.3f}")


if __name__ == "__main__":
    main()
