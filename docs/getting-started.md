# Getting Started

## Installation

```bash
# Pick the provider extra that matches the model you plan to use
uv add "ai-arch-toolkit[openai]"
# or: [anthropic], [gemini], [xai], [all]
```

Or with pip:

```bash
pip install "ai-arch-toolkit[openai]"
# or: [anthropic], [gemini], [xai], [all]
```

The base package has no provider SDK dependencies. If you want to call a model,
install the extra for that provider.

## Quick Start

### Simple LLM Call

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-4-20250514")
response = llm.complete_sync("What is the capital of France?")
print(response.text)
```

### Reusable Prompt Files

```python
from ai_arch_toolkit import LLM, load_prompt

template = load_prompt("prompts/story-writer.prompt.yaml")
rendered = template.render(genre="mystery", task="Write chapter one")

llm = LLM("gpt-4.1-nano")
response = llm.complete_sync("Begin.", system=rendered.text)
```

Prompt loading and rendering require no provider or API key. Install the `prompts` extra
for the complete YAML, Jinja, and JSON Schema feature set. See [Prompts](prompts.md).

### Using Tools

```python
from ai_arch_toolkit import LLM, tool, ToolGroup

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

llm = LLM("claude-sonnet-4-20250514")
tools = ToolGroup(get_weather)

response = llm.complete_sync(
    "What's the weather in Paris?",
    tools=tools,
)
print(response.text)
```

### ReAct Flow

```python
from ai_arch_toolkit import LLM, State, ToolGroup, tool
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Result for: {query}"

llm = LLM("claude-sonnet-4-20250514")
tools = ToolGroup(search)

flow = react_flow(llm, tools, max_iterations=5)
state = State(operational=react_initial_state("Find the population of Tokyo"))
result = flow.run_sync(state)
print(state["response"].text)
```

### Streaming

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-4-20250514")

# Text streaming
stream = llm.stream_sync("Tell me a joke")
for chunk in stream:
    print(chunk, end="")

# Rich event streaming (async)
import asyncio

async def main():
    stream = llm.stream_events("Tell me a joke")
    async for event in stream:
        if event.kind == "text":
            print(event.text, end="")

asyncio.run(main())
```

## Multi-Provider Support

Switch providers by changing the model name:

```python
from ai_arch_toolkit import LLM

# Anthropic
llm = LLM("claude-sonnet-4-20250514")

# OpenAI
llm = LLM("gpt-4o")

# Gemini
llm = LLM("gemini-2.0-flash")

# xAI
llm = LLM("grok-2")

# Local OpenAI-compatible server (Ollama, LM Studio, vLLM) — no API key needed on localhost
llm = LLM("gemma4:e4b", base_url="http://localhost:11434/v1")

# Force the adapter explicitly if needed
llm = LLM("my-model", provider="openai", base_url="http://localhost:8000/v1")
```

Unknown model names with `base_url=` set route to the OpenAI-compatible adapter
automatically. When `base_url` points at a loopback host (`localhost`, `127.x`,
`::1`) the API key is optional — local servers ignore it, and any cloud key in
your environment is **not** sent there. A remote endpoint (a hosted gateway or
proxy) still requires a key, so genuine misconfigurations fail fast.

Reasoning deltas from these servers (`reasoning_content` / `reasoning`) surface
as real-time `thinking` events in `stream_events()` (each event is a fragment —
`event.partial` is `True`) and as complete `Response.thinking` blocks.

## Agent Flows

The toolkit exposes these built-in flow factories:

- **`react_flow()`** — Thought-Action-Observation loop
- **`reflexion_flow()`** — ReAct with self-critique retry
- **`rewoo_flow()`** — Plan with placeholders, execute, solve
- **`plan_execute_flow()`** — Numbered plan, per-step ReAct, solve
- **`tot_flow()`** — Tree of Thoughts (DFS/BFS search)
- **`lats_flow()`** — Language Agent Tree Search (MCTS)
- **`self_discovery_flow()`** — Select reasoning modules, adapt, operationalize, solve
- **`llm_compiler_flow()`** — Plan a DAG, parallel execute, join
- **`generate_review_flow()`** — Generator-reviewer loop with retry feedback

Each factory returns a `Flow` and has a companion `*_initial_state(task)` helper.

## Graph

Build typed, directed graphs with algorithms and persistence:

Install the `graph` extra for this section, for example
`ai-arch-toolkit[openai,graph]` or `ai-arch-toolkit[all]`.

```python
from ai_arch_toolkit import Graph, GraphNode
from ai_arch_toolkit.core.graph._networkx import NetworkXBackend

g = Graph(NetworkXBackend())

# Add typed nodes
alice = g.add_sync(GraphNode(id="alice", type="person", content="Alice"))
bob = g.add_sync(GraphNode(id="bob", type="person", content="Bob"))
proj = g.add_sync(GraphNode(id="p1", type="project", content="Website"))

# Connect with typed edges
g.connect_sync("alice", "p1", "WORKS_ON")
g.connect_sync("bob", "p1", "WORKS_ON")
g.connect_sync("alice", "bob", "KNOWS")

# Query
print(g.has_sync("alice"))          # True
print(g.degree_sync("alice"))       # 2
print(g.node_count_sync())          # 3
print(g.get_stats_sync())           # {node_count: 3, edge_count: 3, ...}

# Algorithms
pr = g.pagerank_sync()              # {alice: 0.33, bob: 0.38, p1: 0.29}
desc = g.get_descendants_sync("alice")  # {bob, p1}

# Persistence
g.save_sync("my_graph.json")
```

See `examples/28_memory_graph_basics.py` for the memory layer built on top of this.

## Next Steps

- See `examples/` for complete working examples
- Read [Prompts](prompts.md) and the [Context Model](context-model.md)
- Read the [API docs](api.md) for detailed reference
- Check the [UV guide](uv-guide.md) for development setup
