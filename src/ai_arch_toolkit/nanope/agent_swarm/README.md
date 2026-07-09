# Agent Swarm

Experimental MVP for coordinating multiple `toolkit.agents.Agent` instances as
one swarm run. This nano project is isolated under `nanope/`: it composes the
existing core/toolkit APIs and does not require changes to either layer.

## What Exists

- `AgentNode`: wraps an existing `Agent` with identity, role, grid, metadata, and permissions.
- `Swarm`: runs multiple nodes with `parallel` or `sequential` execution.
- `SwarmBus`: in-memory message bus and shared notes store.
- `SharedNotes`: grid-scoped facade for shared notes.
- `Message` / `SharedNote` / `SwarmEvent` / `SwarmRunResult`: typed runtime records.
- `swarm_tool_group()`: optional communication tools for agents that should send messages,
  broadcast, read inbox, and write/search shared notes during their own ReAct loop.
- `Swarm.iter()` / `Swarm.iter_sync()`: event stream for start/end, agent errors, messages,
  and notes.

## Minimal Usage

```python
from ai_arch_toolkit.core import LLM, ToolGroup
from ai_arch_toolkit.nanope.agent_swarm import AgentNode, Swarm, SwarmPolicy
from ai_arch_toolkit.toolkit.agents import Agent, ReasoningSpec

llm = LLM("gpt-5-mini")

researcher = Agent(
    ReasoningSpec(strategy="react", system="Research the task and write concise findings."),
    llm,
    ToolGroup(),
)
reviewer = Agent(
    ReasoningSpec(strategy="react", system="Review the findings and identify gaps."),
    llm,
    ToolGroup(),
)
finalizer = Agent(
    ReasoningSpec(strategy="completion", system="Synthesize the swarm output."),
    llm,
    ToolGroup(),
)

swarm = Swarm(
    [
        AgentNode(id="researcher", agent=researcher, role="Collect useful facts."),
        AgentNode(id="reviewer", agent=reviewer, role="Find risks and omissions."),
        AgentNode(id="finalizer", agent=finalizer, role="Produce the final answer."),
    ],
    policy=SwarmPolicy(mode="parallel", finalizer_id="finalizer"),
)

result = swarm.run_sync("Compare SQLite and Postgres for a small SaaS product.")
print(result.final_text)
```

## Communication Tools

`swarm_tool_group()` closes over a `SwarmBus`, so agents can use the bus as normal tools.
Build those tools before constructing the `Agent`.

```python
from ai_arch_toolkit.nanope.agent_swarm import SwarmBus, swarm_tool_group

bus = SwarmBus()
tools = swarm_tool_group(bus, "researcher")

# Pass `tools` into Agent(...), then pass the same bus into Swarm(..., bus=bus).
```

Available tools:

- `send_message(to, content)`
- `broadcast_message(content)`
- `read_inbox(limit=10)`
- `write_shared_note(content, tags="")`
- `search_shared_notes(query="", limit=10)`

## Current Limits

- This is an in-memory MVP. There is no durable persistence yet.
- Direct/broadcast messages are available through `SwarmBus` and optional tools, but there is
  no continuous tick loop yet.
- Parallel mode is fan-out/fan-in: work agents run concurrently, then the optional finalizer runs.
- Non-text multimodal tasks are passed through to agents as-is, without prompt-level swarm context.
- Permissions are simple booleans, not a full RBAC system.
