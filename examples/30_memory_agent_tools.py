"""Memory tools with ReAct Flow — deliberate remember, recall, explore, forget.

Demonstrates:
  - memory_tools() creating a ToolGroup with 4 memory tools
  - react_flow using memory tools alongside regular tools
  - Flow remembering facts, recalling them later, and exploring the graph
  - Combining memory tools with other toolkit tools

Requires: OPENAI_API_KEY (or change the model to your provider).
"""

from __future__ import annotations

import asyncio

from ai_arch_toolkit import LLM, State
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.memory import GraphStore, Node, memory_tools
from ai_arch_toolkit.toolkit.memory.graph._networkx import NetworkXBackend


def make_embed_fn():
    """Create a simple hash-based embedding function.

    For a real app, replace with an embedding API (OpenAI, Cohere, etc.).
    This stand-in produces deterministic 64-dim vectors from text hashes.
    """

    async def embed(text: str) -> list[float]:
        h = hash(text)
        return [(((h >> i) & 0xFF) - 128) / 128.0 for i in range(64)]

    return embed


async def main() -> None:
    print("=== Memory Agent Tools ===\n")

    # --- 1. Create store and seed some knowledge ---
    store = GraphStore(NetworkXBackend(), embed=make_embed_fn())

    seeds = [
        Node(
            type="fact",
            content={"text": "Project Apollo uses Python 3.13", "subject": "Apollo"},
            source="external",
        ),
        Node(
            type="fact",
            content={"text": "The API rate limit is 100 requests per minute", "subject": "API"},
            source="external",
        ),
        Node(
            type="preference",
            content={"text": "Deploy to staging before production"},
            source="user_stated",
        ),
    ]
    for node in seeds:
        await store.add(node)

    # Connect related nodes
    nodes = await store.list()
    if len(nodes) >= 2:
        await store.connect(nodes[0].id, nodes[1].id, "SAME_PROJECT")

    print(f"Seeded {await store.count()} memories\n")

    # --- 2. Create tools ---
    mem_tools = memory_tools(store)
    print(f"Memory tools: {mem_tools}\n")

    # --- 3. Create flow with memory tools ---
    llm = LLM("gpt-4.1-nano")
    flow = react_flow(
        llm,
        mem_tools,
        system=(
            "You are a helpful assistant with access to a memory graph. "
            "Use recall to search your memory before answering questions. "
            "Use remember to store important new information. "
            "Use explore_memory to understand how memories relate to each other."
        ),
        max_iterations=6,
    )

    # --- 4. Run tasks that exercise memory ---
    tasks = [
        "What do you know about Project Apollo? Search your memory.",
        "Remember this: Project Apollo's deadline is March 15th, 2026.",
        "What's the deployment procedure? Check your memory for preferences.",
    ]

    for task in tasks:
        print(f"Task: {task}")
        state = State(operational=react_initial_state(task))
        result = flow.run_sync(state)
        print(f"Answer: {state['response'].text[:300]}")
        print(f"Steps: {len(result.trace.steps)}, Cost: ${result.total_cost:.4f}\n")

    # --- 5. Show final memory state ---
    print("--- Final Memory State ---")
    all_nodes = await store.list()
    print(f"Total nodes: {len(all_nodes)}")
    for node in all_nodes:
        text = " ".join(str(v) for v in node.content.values() if isinstance(v, str))
        print(f"  [{node.type}] src={node.source} | {text[:70]}")

    edges_count = 0
    for node in all_nodes:
        edges = await store.edges(node.id)
        edges_count += len(edges)
    print(f"Total edges: {edges_count}")


if __name__ == "__main__":
    asyncio.run(main())
