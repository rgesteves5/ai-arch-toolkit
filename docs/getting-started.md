# Getting Started

## Installation

```bash
uv add ai-arch-toolkit
```

Or with pip:

```bash
pip install ai-arch-toolkit
```

## Quick Start

### Simple LLM Call

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-4-20250514")
response = llm.complete_sync("What is the capital of France?")
print(response.text)
```

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

### ReAct Agent

```python
from ai_arch_toolkit import LLM, ReActAgent, AgentConfig, ToolGroup, tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Result for: {query}"

llm = LLM("claude-sonnet-4-20250514")
tools = ToolGroup(search)
config = AgentConfig(max_iterations=5)

agent = ReActAgent(llm, tools, config=config)
result = agent.run_sync("Find the population of Tokyo")
print(result.answer)
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
```

## Agent Architectures

The toolkit includes several agent architectures:

- **ReActAgent** — Thought-Action-Observation loop
- **ReflexionAgent** — ReAct with self-critique retry
- **ReWOOAgent** — Plan with placeholders, execute, solve
- **PlanExecuteAgent** — Numbered plan, per-step ReAct, solve
- **ToTAgent** — Tree of Thoughts (DFS/BFS search)
- **LATSAgent** — Language Agent Tree Search (MCTS)
- **SelfDiscoveryAgent** — Select reasoning modules, adapt, operationalize, solve
- **LLMCompilerAgent** — Plan a DAG, parallel execute, join

## Graph

Build typed, directed graphs with algorithms and persistence:

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
- Read the [API docs](api.md) for detailed reference
- Check the [UV guide](uv-guide.md) for development setup
