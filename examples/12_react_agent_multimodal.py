"""12 — Multimodal ReAct Flow.

A react_flow that receives an image alongside text and uses Wikipedia
tools to research and complement its visual analysis.

The flow sees the image (Alice in Wonderland book cover), identifies
what it depicts, then uses wikipedia_search and wikipedia_article to
look up relevant facts and provide a well-rounded answer.
"""

from pathlib import Path

from ai_arch_toolkit import LLM, State, ToolGroup, image
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.tools import wikipedia_article, wikipedia_search

tools = ToolGroup(wikipedia_search, wikipedia_article)
llm = LLM("gpt-4.1-nano")

flow = react_flow(
    llm,
    tools,
    system=(
        "You are a knowledgeable visual assistant. "
        "First describe what you see in the image, then use the Wikipedia tools "
        "to look up relevant facts and provide a well-rounded answer."
    ),
    max_iterations=5,
)

# Load a local image (Alice in Wonderland cover from Project Gutenberg)
image_path = Path(__file__).parent / "alice_in_wonderland.jpg"
image_bytes = image_path.read_bytes()

# Pass multimodal content: text + image
task = [
    "What book is this the cover of? "
    "Tell me key facts about it — author, publication year, and legacy.",
    image(image_bytes, media_type="image/jpeg"),
]

state = State(operational=react_initial_state(task))

print("=== Multimodal ReAct Flow ===\n")

for event in flow.iter_sync(state):
    if event.type == "step_start":
        print(f"--- Step: {event.step_name} ---")
    elif event.type == "step_end" and event.result:
        print(f"  Completed: {event.step_name}")
    elif event.type == "flow_end":
        print(f"\n[flow complete — cost: ${event.trace.total_cost:.4f}]")

print(f"\nAnswer: {state['response'].text}")
print("\nDone.")
