"""12 — Multimodal ReAct Agent.

A ReActAgent that receives an image alongside text and uses Wikipedia
tools to research and complement its visual analysis.

The agent sees the image (Alice in Wonderland book cover), identifies
what it depicts, then uses wikipedia_search and wikipedia_article to
look up relevant facts and provide a well-rounded answer.
"""

from pathlib import Path

from ai_arch_toolkit.core import LLM, ToolGroup, image
from ai_arch_toolkit.toolkit.agents import AgentConfig, ReActAgent
from ai_arch_toolkit.toolkit.tools import wikipedia_article, wikipedia_search

tools = ToolGroup(wikipedia_search, wikipedia_article)
llm = LLM("gpt-4.1-nano")

agent = ReActAgent(
    llm,
    tools,
    config=AgentConfig(
        system=(
            "You are a knowledgeable visual assistant. "
            "First describe what you see in the image, then use the Wikipedia tools "
            "to look up relevant facts and provide a well-rounded answer."
        ),
        max_iterations=5,
    ),
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

print("=== Multimodal ReAct Agent ===\n")

for event in agent.run_sync(task, stream=True):
    if event.type == "step_start":
        print(f"--- Step {event.step} ---")
    elif event.type == "tool_call":
        print(f"  Tool: {event.tool_name}({event.tool_args})")
    elif event.type == "tool_result":
        print(f"  Result: {event.result}")
    elif event.type == "error":
        print(f"  Error: {event.error}")
    elif event.type == "step_end":
        if event.response:
            print(f"  Answer: {event.response.text}")
        if event.stop_reason:
            print(f"\n[stop_reason={event.stop_reason}]")

print("\nDone.")
