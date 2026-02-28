"""27 — LLMCompiler Agent.

LLMCompiler plans a DAG of tasks, executes independent tasks in parallel,
then joins the results. Supports replanning if results are insufficient.

Three phases per iteration: Plan DAG → Parallel Execute → Join.
"""

from ai_arch_toolkit import LLM, AgentConfig, LLMCompilerAgent, ToolGroup, tool

llm = LLM("claude-sonnet-4-20250514")


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
agent = LLMCompilerAgent(
    llm,
    tools,
    config=AgentConfig(max_iterations=5),
)

result = agent.run_sync(
    "Compare France, Germany, and Japan by population and GDP. "
    "Which country has the highest GDP per capita?"
)

print("Answer:", result.answer)
print(f"Steps: {len(result.steps)}")
print(f"Stop reason: {result.stop_reason}")
