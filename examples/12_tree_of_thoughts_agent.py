"""12 — Tree of Thoughts Agent (xAI).

A reasoning-only agent that explores multiple thought branches to solve
a classic river-crossing puzzle. No tools needed — pure LLM reasoning.
"""

from ai_arch_toolkit import (
    AgentConfig,
    Client,
    ToolRegistry,
    TreeOfThoughtsAgent,
)


def on_event(event):
    if event.type == "step_start":
        print(f"  Branch exploration — depth {event.step_number}")
    elif event.type == "step_end":
        score = event.metadata.get("score", "?")
        print(f"    Score: {score}")


client = Client("xai", model="grok-4-1-fast-reasoning")
agent = TreeOfThoughtsAgent(
    client,
    ToolRegistry(),  # empty registry — reasoning only
    config=AgentConfig(on_event=on_event),
)

task = (
    "A farmer needs to cross a river with a wolf, a goat, and a cabbage. "
    "The boat can only carry the farmer and one item at a time. "
    "If left alone, the wolf will eat the goat, and the goat will eat the cabbage. "
    "How can the farmer get everything across safely?"
)

print("Solving river-crossing puzzle with Tree of Thoughts...\n")
result = agent.run(
    task,
    max_depth=3,
    branching_factor=3,
    beam_width=2,
    search_strategy="bfs",
)

print(f"\nSolution:\n{result.answer}")
print(f"Steps explored: {len(result.steps)}")
