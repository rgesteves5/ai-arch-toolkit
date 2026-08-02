# Advanced Multi-Purpose Configurable Agent

A reusable agent runtime that can be configured into many roles without rewriting the
agent logic.

It is not the swarm yet. It is the single-agent building block that the swarm project
can later compose.

## Core Composition

### Agent Identity

- `name`
- `description`

### Optional Agent Context

- `role`
- `goals`
- `tasks`
- `style`
- optional personality/behavior constraints

The system prompt always includes `name` and `description` as mandatory fields. It can
also include optional fields when they exist, such as `role`, `goals`, `tasks`, `style`,
and behavior constraints.

### Model Configuration

- provider/model name
- temperature
- max tokens
- fallback models
- structured output schema
- streaming on/off

`AgentConfig.to_dict()` omits `api_key` by default, and configuration fingerprints never
include it. Internal runtime resolution preserves explicitly configured keys without exposing
them through normal serialization. Runtime override reports use the toolkit's central redaction
policy, so accepted secret values are reported as `[REDACTED]`.

### Reasoning Strategy

The agent supports configurable reasoning modes using existing toolkit flows:

- `react`
- `plan_execute`
- `reflexion`
- `self_discovery`
- `generate_review`
- `rewoo`
- `llm_compiler`
- `tot`
- `lats`

### Tools

- configurable tool groups
- enable/disable tools per agent
- tool permissions
- timeout per tool
- allowed filesystem paths, shell commands, web tools, etc.
- custom tools added by the user

### Capability Profiles

Capability profiles are reusable bundles of abilities, permissions, tools, limits, and
behavior constraints.

They describe what the agent can do, not who the agent is.

Examples:

- `read_only`
- `web_researcher`
- `report_writer`
- `code_editor`
- `critic`
- `private_memory_user`

A capability profile can include:

- allowed tools
- blocked tools
- tool groups
- permissions
- memory behavior
- model preferences
- default reasoning strategy
- safety rules
- budget limits
- output expectations
- approval requirements
- observability level

Agents can compose multiple profiles:

```yaml
capability_profiles:
  - web_researcher
  - report_writer
  - private_memory_user
```

The distinction is:

```text
Agent identity says who it is.
Optional context says what it is trying to do.
Capability profiles say what it can do.
```

### Private Memory

- short-term task state
- long-term private memory
- graph-backed memory using existing `toolkit.memory`
- memory search
- memory write policy
- memory confidence/source metadata
- only this agent can access its own private memory

### Runtime State

- current task
- conversation history
- intermediate steps
- tool calls
- observations
- final answer
- errors/retries
- cost and token usage

### Policy & Limits

- max iterations
- max tool calls
- max cost
- max runtime
- retry config
- stop conditions
- safety rules
- approval requirements for dangerous tools

### Runtime Config Overrides

The agent has a base config, but callers can override selected fields at runtime for a
run, session, step, or environment.

Common runtime overrides:

- model
- temperature
- max tokens
- flow/reasoning strategy
- enabled tools
- tool permissions
- memory read/write behavior
- budget
- max iterations
- output schema
- system prompt additions
- approval policy
- logging/debug level

Override policies define what can and cannot be changed:

```yaml
override_policy:
  allow:
    - model
    - temperature
    - limits.budget
    - limits.max_iterations
    - output.schema
  deny:
    - identity.name
    - identity.description
    - memory.private
    - tools.shell.enabled
    - permissions.filesystem
```

Default rule:

```text
deny wins over allow
```

Useful precedence order:

```text
framework defaults
< capability profiles
< agent config
< environment config
< session overrides
< run overrides
< step overrides
```

Runtime overrides say what changes for this execution. Override policies say what must
not be changed.

### Input / Output

- plain text input
- multimodal input using existing `Content`
- structured output for `react` and the `generate_review` generator
- final `AgentRunResult`

Planned but not wired yet:

- streaming run events
- artifact output

### Observability

- event log
- trace of reasoning steps
- tool call history
- memory reads/writes
- cost tracking
- debug mode
- export run report

### Configuration Format

The agent should be definable from config, probably YAML/TOML plus Python:

```yaml
name: researcher
description: Researches a topic and produces grounded notes.
role: Research Agent
model: grok-4-1-fast-reasoning
flow: react
capability_profiles:
  - web_researcher
  - report_writer
  - private_memory_user
tools:
  - wikipedia
  - dictionary
  - web
memory:
  enabled: true
  private: true
limits:
  max_iterations: 8
  budget: 0.25
override_policy:
  allow:
    - model
    - temperature
    - limits.budget
    - limits.max_iterations
    - output.schema
  deny:
    - identity.name
    - identity.description
    - memory.private
    - tools.shell.enabled
    - permissions.filesystem
```

## Main Features

- Create an agent from a config file.
- Create an agent from Python.
- Swap reasoning strategy without changing agent code.
- Attach different tools depending on the role.
- Give each agent private memory.
- Run sync or async.
- Track cost and token usage.
- Save/load private memory and terminal chat sessions.
- Inspect the last run, trace, tool set, prompt, profiles, and cost in terminal chat.
- Support structured outputs for `react` and the `generate_review` generator.
- Support model fallback.
- Govern dangerous tools with explicit allow, dry-run behavior, and max tool-call limits.
- Support evaluator/reviewer loops for higher quality answers.

The clean boundary is:

```text
Advanced Configurable Agent = one powerful reusable agent
Agent Swarm = many configurable agents + shared notes + grids + communication
```

## Current Implementation

This nano project now includes a Python runtime for a single configurable agent.

Implemented:

- config dataclasses and mapping/TOML loaders
- optional YAML loading when PyYAML is installed
- capability profile resolution
- runtime overrides with allow/deny policy
- deterministic system prompt rendering
- optional toolkit prompt manifests with typed variables and Text/Markdown/XML/JSON layouts
- toolkit Knowledge injection into prompt manifests (including resource-backed entries)
- small tool registry and enabled/disabled tool resolution
- `ConfigurableAgent.run()` and `ConfigurableAgent.run_sync()`
- terminal chat for manual real-LLM testing
- reasoning execution using existing toolkit flows
- private memory tools backed by `GraphStore`
- private memory injection into the user task context
- explicit memory inspection and duplicate consolidation tools
- general web search tool
- structured output schema config for supported LLM calls
- tool governance for dangerous tools, dry runs, and max tool calls
- session history persistence for terminal chat
- profile/strategy/last-run inspection commands
- structured `AgentRunResult`

Reasoning strategies currently wired:

- `react`
- `plan_execute`
- `reflexion`
- `self_discovery`
- `generate_review`
- `rewoo`
- `llm_compiler`
- `tot`
- `lats`

Built-in capability profiles:

- `basic_chat`
- `web_researcher`
- `math_helper`
- `data_helper`
- `geo_weather`
- `local_reader`
- `local_operator`
- `private_memory_user`
- `deep_reasoner`
- `reviewer`
- `all_tools`

Not implemented yet:

- swarm/grid orchestration
- generic agent-state persistence beyond private memory and chat sessions
- exported run-history files beyond last-run terminal inspection
- streamed `ConfigurableAgent.run()` progress events
- config-driven middleware/hooks

Example:

```python
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import (
    ConfigurableAgent,
)
from ai_arch_toolkit.toolkit.tools import wikipedia_search

agent = ConfigurableAgent(
    {
        "identity": {
            "name": "researcher",
            "description": "Researches topics and produces grounded notes.",
        },
        "model": {"name": "gpt-5-mini"},
        "context": {
            "role": "Research Agent",
            "goals": ["collect reliable information", "produce concise synthesis"],
        },
        "reasoning": {
            "strategy": "react",
            "max_iterations": 8,
        },
        "tools": {
            "enabled": ["wikipedia_search"],
        },
        "override_policy": {
            "allow": ["model.temperature", "limits.max_cost", "tools.enabled"],
            "deny": ["identity.name", "identity.description", "memory.private_enabled"],
        },
    },
    tool_registry={"wikipedia_search": wikipedia_search},
)

result = agent.run_sync(
    "Research the history of graph memory for agents.",
    overrides={"model.temperature": 0.2},
)

print(result.final_text)
```

### Manual Chat

From the repository root:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent
```

The chat loads `.env` from the current directory by default. You can also load it
yourself:

```bash
set -a
source .env
set +a
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent
```

Useful options:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent \
  --model gpt-5-mini \
  --temperature 0.2 \
  --strategy react \
  --profiles web_researcher \
  --max-iterations 6 \
  --tools wikipedia_search,wikipedia_article,define_word,math_eval,datetime_now
```

Use `--tools none` for a pure chat without tools.

Use `--tools all` to expose all built-in chat tools.

Dangerous tools (`run_command`, `python_repl`) are blocked unless explicitly allowed:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent \
  --tools all \
  --allow-dangerous-tools
```

Dry-run tools without executing them:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent \
  --tools all \
  --dry-run-tools \
  --max-tool-calls 5
```

Use private memory:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent \
  --memory \
  --memory-path private-memory.json
```

Then try:

```text
Remember that I prefer concise answers.
What do you remember about my answer style?
```

Memory tools available when memory is enabled:

- `remember`
- `recall`
- `explore_memory`
- `forget_memory`
- `list_memories`
- `find_duplicate_memories`
- `consolidate_memories`

Example memory inspection prompts:

```text
List all memories.
Find duplicate memories and report them.
Consolidate duplicate memories.
```

Use every non-swarm capability currently available:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent \
  --model gpt-5-mini \
  --strategy react \
  --profiles all_tools,private_memory_user \
  --tools all \
  --memory \
  --memory-path private-memory.json \
  --session-path manual-session.json \
  --history-turns 12 \
  --max-iterations 10 \
  --temperature 1
```

Load from a config file and apply runtime overrides:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent \
  --config agent.toml \
  --override model.temperature=0.4 \
  --override reasoning.max_iterations=8
```

Use a reusable toolkit prompt manifest alongside or instead of built-in sections:

```toml
[prompt]
manifest = "prompts/researcher.prompt.yaml"
layout = "markdown"
mode = "append" # or "replace"

[prompt.variables]
domain = "distributed systems"
```

Relative manifest paths are resolved against the agent configuration file.

Knowledge is supplied at construction time so the agent does not own file parsing or registry
lifecycle:

```python
from ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent import ConfigurableAgent
from ai_arch_toolkit.toolkit.knowledge import KnowledgeRegistry

knowledge = KnowledgeRegistry.from_directory("knowledge/", recursive=True, prefix="kb.")
agent = ConfigurableAgent(config, knowledge=knowledge)
```

Manifest sections can then select keys with `knowledge:`. For command-line prompt workflows,
use `ai-arch prompt render ... --knowledge-dir knowledge/` or repeated
`--knowledge key=path` options.

Request structured output with a JSON Schema file:

```bash
uv run python -m ai_arch_toolkit.nanope.advanced_multi_purpose_configurable_agent \
  --output-schema answer.schema.json
```

Structured output is supported for `react` and the generator phase of `generate_review`.
Other reasoning strategies raise a clear error when `output.schema` is configured.

Chat commands:

- `/help`
- `/reset`
- `/tools`
- `/profiles`
- `/profile NAME`
- `/strategies`
- `/prompt`
- `/capabilities`
- `/last-tools`
- `/last-cost`
- `/last-result`
- `/last-trace`
- `/save-session`
- `/exit`
