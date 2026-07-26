"""27 — LLMCompiler Flow.

LLMCompiler plans a DAG of tasks, executes independent tasks in parallel,
then joins the results. Supports replanning if results are insufficient.

Three phases per iteration: Plan DAG → Parallel Execute → Join.
"""

from ai_arch_toolkit import LLM, State, ToolGroup, tool
from ai_arch_toolkit.toolkit.agents import llm_compiler_flow, llm_compiler_initial_state

llm = LLM("claude-sonnet-5")


@tool
def get_population(country: str) -> str:
    """Get the population of a country.

    Args:
        country: The country name.
    """
    populations = {"France": "67 million", "Germany": "83 million", "Japan": "125 million"}
    return populations.get(country, f"Population of {country}: unknown")


@tool
def get_gdp(country: str) -> str:
    """Get the GDP of a country.

    Args:
        country: The country name.
    """
    gdps = {"France": "$2.78 trillion", "Germany": "$4.07 trillion", "Japan": "$4.23 trillion"}
    return gdps.get(country, f"GDP of {country}: unknown")


tools = ToolGroup(get_population, get_gdp)

flow = llm_compiler_flow(llm, tools, max_replans=2)

state = State(
    operational=llm_compiler_initial_state(
        "Compare France, Germany, and Japan by population and GDP. "
        "Which country has the highest GDP per capita?"
    )
)
result = flow.run_sync(state)

print("Answer:", state["response"].text)
print(f"Steps: {len(result.trace.steps)}")
print(f"Cost: ${result.total_cost:.4f}")
