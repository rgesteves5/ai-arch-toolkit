"""26 — Self-Discovery Flow.

Self-Discovery selects reasoning modules, adapts them to the task,
creates a reasoning plan, then solves using that plan with tool access.

Four phases: Select → Adapt → Operationalize → Solve.
"""

from ai_arch_toolkit import LLM, State, ToolGroup, tool
from ai_arch_toolkit.toolkit.agents import self_discovery_flow, self_discovery_initial_state

llm = LLM("claude-sonnet-5")


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

flow = self_discovery_flow(llm, tools, max_react_iterations=5)

state = State(
    operational=self_discovery_initial_state(
        "What are the main factors contributing to climate change, "
        "and how do they interact with each other?"
    )
)
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
