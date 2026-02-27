"""12 — ReAct Agent across Multiple Models.

The same agent definition works with any provider. Just swap the LLM
model string — the agent loop, tool execution, and event handling all
stay the same.
"""

from ai_arch_toolkit.agents import AgentConfig, AgentResult, ReActAgent
from ai_arch_toolkit.core import LLM, ToolGroup, tool


@tool
def lookup(topic: str) -> str:
    """Look up a fact about a topic.

    Args:
        topic: The topic to look up.
    """
    facts = {
        "eiffel tower": "The Eiffel Tower is 330 metres tall, built in 1889.",
        "mount everest": "Mount Everest is 8,849 metres, the tallest mountain.",
        "great wall": "The Great Wall of China stretches over 21,000 km.",
    }
    for key, value in facts.items():
        if key in topic.lower():
            return value
    return f"No facts found for: {topic}"


tools = ToolGroup(lookup)
task = "How tall is the Eiffel Tower?"

models = [
    "gpt-4.1-nano",
    "claude-haiku-4-5-20251001",
    "gemini-2.0-flash-lite",
]


def run_with_model(model: str) -> AgentResult:
    llm = LLM(model)
    agent = ReActAgent(
        llm,
        tools,
        config=AgentConfig(system="Use the lookup tool to answer.", max_iterations=3),
    )
    return agent.run_sync(task)


for model in models:
    print(f"=== {model} ===")
    try:
        result = run_with_model(model)
        print(f"  Answer: {result.answer}")
        print(f"  Steps: {len(result.steps)}, Stop: {result.stop_reason}")
        print(f"  Tokens: {result.total_usage.input_tokens}+{result.total_usage.output_tokens}")
        print(f"  Cost: ${result.total_cost:.4f}")
    except Exception as e:
        print(f"  Skipped ({type(e).__name__}: {e})")
    print()
