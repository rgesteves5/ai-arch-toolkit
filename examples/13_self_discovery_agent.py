"""13 — Self-Discovery Agent (Anthropic).

The SelfDiscoveryAgent works through 4 phases:
  SELECT  — pick relevant reasoning modules
  ADAPT   — tailor them to the task
  IMPLEMENT — build a structured reasoning plan
  SOLVE   — execute the plan to produce an answer
"""

from ai_arch_toolkit import (
    AgentConfig,
    Client,
    SelfDiscoveryAgent,
    ToolRegistry,
)


def on_event(event):
    phase_names = {1: "SELECT", 2: "ADAPT", 3: "IMPLEMENT", 4: "SOLVE"}
    if event.type == "step_start":
        phase = phase_names.get(event.step_number, f"Step {event.step_number}")
        print(f"  Phase: {phase}")


client = Client("anthropic", model="claude-haiku-4-5")
agent = SelfDiscoveryAgent(
    client,
    ToolRegistry(),  # reasoning only
    config=AgentConfig(on_event=on_event),
)

task = (
    "A company has 100 employees. 60 speak English, 50 speak Spanish, "
    "and 20 speak both. How many employees speak neither language? "
    "What percentage of Spanish speakers also speak English?"
)

custom_modules = [
    "Set Theory and Venn Diagrams",
    "Arithmetic Reasoning",
    "Percentage Calculation",
    "Step-by-step Decomposition",
]

print("Solving with Self-Discovery...\n")
result = agent.run(task, reasoning_modules=custom_modules)

print(f"\nAnswer:\n{result.answer}")
print(f"\nPhases completed: {len(result.steps)}")

# Inspect individual phase outputs
for i, step in enumerate(result.steps, 1):
    phase_names = {1: "SELECT", 2: "ADAPT", 3: "IMPLEMENT", 4: "SOLVE"}
    name = phase_names.get(i, f"Step {i}")
    preview = step.response.text[:100].replace("\n", " ")
    print(f"  {name}: {preview}...")
