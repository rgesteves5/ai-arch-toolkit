"""26 — Self-Discovery Agent.

Self-Discovery selects reasoning modules, adapts them to the task,
creates a reasoning plan, then solves using that plan with tool access.

Four phases: Select → Adapt → Operationalize → Solve.
"""

from ai_arch_toolkit import LLM, AgentConfig, SelfDiscoveryAgent, ToolGroup, tool

llm = LLM("claude-sonnet-4-20250514")


@tool
def search(query: str) -> str:
    """Search for information on a topic.

    Args:
        query: The search query.
    """
    return f"Search result for '{query}': relevant information found."


@tool
def analyze(data: str) -> str:
    """Analyze data and extract insights.

    Args:
        data: The data to analyze.
    """
    return f"Analysis of '{data}': key patterns identified."


tools = ToolGroup(search, analyze)
agent = SelfDiscoveryAgent(
    llm,
    tools,
    config=AgentConfig(max_iterations=5),
)

result = agent.run_sync(
    "What are the main factors contributing to climate change, "
    "and how do they interact with each other?"
)

print("Answer:", result.answer)
print(f"Steps: {len(result.steps)}")
print(f"Stop reason: {result.stop_reason}")
