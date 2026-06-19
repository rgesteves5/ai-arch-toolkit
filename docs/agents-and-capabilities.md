# Agents and Framework Capabilities

This document covers the agent system and all framework capabilities that support, enhance, and compose with agents.

---

## Agent Flows

The package exposes these built-in flow factories. See [Flow Architecture](flow-architecture.md) for the underlying primitives (State, Step, Result, Policy, Trace, Scope).

```python
from ai_arch_toolkit.toolkit.agents.flows import (
    react_flow, react_initial_state,
    reflexion_flow, reflexion_initial_state,
    rewoo_flow, rewoo_initial_state,
    plan_execute_flow, plan_execute_initial_state,
    tot_flow, tot_initial_state,
    lats_flow, lats_initial_state,
    self_discovery_flow, self_discovery_initial_state,
    llm_compiler_flow, llm_compiler_initial_state,
    generate_review_flow, generate_review_initial_state,
)
```

### Usage pattern

Every flow factory follows the same pattern:

```python
from ai_arch_toolkit.core import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state

llm = LLM("claude-sonnet-4-20250514")
tools = ToolGroup(my_tool_a, my_tool_b)

# Create the flow
flow = react_flow(llm, tools, system="You are a helpful assistant.")

# Create initial state
state = State(operational=react_initial_state("What's the weather in Paris?"))

# Run
result = await flow.run(state)

# response may be None if the flow halted early (check result.trace for errors)
response = state.get("response")
answer = response.text if response else result.trace.steps[-1].error
print(f"Cost: ${result.total_cost:.4f}")
```

### Streaming

Every flow supports event streaming via `flow.iter()`:

```python
async for event in flow.iter(state):
    match event.type:
        case "flow_start":   print(f"Starting {event.flow_name}")
        case "step_start":   print(f"  Running {event.step_name}")
        case "step_end":     print(f"  Done: {event.result.value}")
        case "step_skipped": print(f"  Skipped: {event.step_name}")
        case "flow_end":     print(f"Finished. Cost: ${event.trace.total_cost:.4f}")

# Or synchronously:
for event in flow.iter_sync(state):
    ...
```

### Inspecting results

Every `FlowResult` provides cost, usage, duration, and a full execution trace:

```python
result = await flow.run(state)

result.total_cost        # sum across all steps
result.total_duration    # wall clock seconds
result.trace.total_usage # summed Usage(input_tokens, output_tokens, ...)
result.trace             # Trace with per-step detail

# Per-step inspection
for st in result.trace.steps:
    print(f"  {st.name}: {st.duration:.1f}s, ${st.cost:.4f}")
    print(f"  Policy decisions: {st.policy_decisions}")
```

### Per-phase LLM/tools override

Most flow factories accept override parameters for different phases:

```python
flow = plan_execute_flow(
    llm,
    tools,
    planner_llm=LLM("claude-haiku-4-5-20251001"),   # cheap model for planning
    exec_llm=LLM("claude-sonnet-4-20250514"),         # mid-tier for execution
    solver_llm=LLM("claude-opus-4-0-20250514"),       # expensive model for final answer
)
```

---

## Built-in Agent Flows

### ReAct

Cyclic flow — LLM reasoning + tool execution loop.

```python
flow = react_flow(llm, tools, system="...", max_iterations=10)
state = State(operational=react_initial_state("Find the capital of France"))
```

Steps: `llm_call` (when: needs_llm_call) → `execute_tools` (when: has_tool_calls) → loop

The LLM reasons, calls tools, observes results, and repeats until it has a final answer or hits max iterations.

### Reflexion

Cyclic flow — inner ReAct with evaluate + reflect retry loop.

```python
def my_evaluator(task: str, answer: str) -> float:
    return 1.0 if "correct" in answer else 0.3

flow = reflexion_flow(llm, tools, evaluator=my_evaluator, threshold=0.7, max_retries=3)
state = State(operational=reflexion_initial_state("Solve this math problem"))
```

Steps: `attempt` → `evaluate` → `reflect` → loop (all gated by `when: not passed`)

Each retry includes accumulated reflections, so the agent learns from its mistakes.

### ReWOO

Sequential flow — plan with evidence placeholders, execute, solve.

```python
flow = rewoo_flow(llm, tools)
state = State(operational=rewoo_initial_state("Research topic X"))
```

Steps: `plan` → `execute` → `solve`

The planner generates `#E1 = ToolName[args]` steps. The executor runs tools sequentially, substituting `#E{n}` references. The solver synthesizes the final answer.

Key difference from ReAct: the planner never sees tool results. All reasoning happens upfront, reducing LLM calls.

### Plan-Execute

Sequential flow — numbered plan, per-step ReAct execution, solve.

```python
flow = plan_execute_flow(llm, tools, max_replans=1, max_iterations_per_step=3)
state = State(operational=plan_execute_initial_state("Build a report on climate change"))
```

Steps: `plan_and_execute` → `solve`

The plan_and_execute step internally: plans numbered steps, runs each via inner ReAct, replans on failure.

### Tree of Thoughts (ToT)

Cyclic flow — DFS/BFS search over reasoning paths.

```python
flow = tot_flow(llm, tools, strategy="dfs", n_candidates=3, max_depth=3, max_iterations=10)
state = State(operational=tot_initial_state("Solve this puzzle"))
```

Steps: `search_step` (when: search_not_done) → loop

Each iteration: select from frontier, generate candidates, evaluate, expand or solve. Terminates when a high-confidence answer is found (score >= 0.9), the frontier is empty, or max iterations/depth reached.

### LATS (Language Agent Tree Search)

Cyclic flow — Monte Carlo Tree Search with ReAct rollouts.

```python
flow = lats_flow(llm, tools, max_rollouts=10, exploration_weight=1.41)
state = State(operational=lats_initial_state("Complex reasoning task"))
```

Steps: `mcts_rollout` (when: search_not_done) → loop

Each rollout: UCT selection, ReAct expansion, evaluation, backpropagation, optional reflection. Low-scoring answers (< 0.5) trigger reflection — the feedback is stored on tree nodes and injected into future rollouts.

### Self-Discovery

Sequential flow — select reasoning modules, adapt, operationalize, solve via ReAct.

```python
flow = self_discovery_flow(llm, tools, max_react_iterations=10)
state = State(operational=self_discovery_initial_state("Analyze this problem"))
```

Steps: `select` → `adapt` → `operationalize` → `solve`

10 default reasoning modules (critical thinking, analogical reasoning, etc.) are selected and adapted to the task before solving.

### LLM Compiler

Sequential flow — plan DAG, parallel execute, join/replan.

```python
flow = llm_compiler_flow(llm, tools, max_replans=2, max_react_iterations=3)
state = State(operational=llm_compiler_initial_state("Multi-step research task"))
```

Steps: `compile` (internally: plan → parallel execute → join → optional replan)

The planner generates `$N. task [deps: $1, $2]` format. Independent tasks run concurrently via `asyncio.gather`.

### Generate-Review

Cyclic flow — generator and reviewer cooperate until the reviewer accepts the answer
or the retry budget is exhausted.

```python
flow = generate_review_flow(gen_llm, review_llm, max_cycles=3)
state = State(operational=generate_review_initial_state("Draft a release note"))
```

Steps: `generate` → `review` → loop while not accepted

This is useful when you want an explicit critique pass, optional tool use in both
phases, and accumulated reviewer feedback injected into later generation attempts.

---

## LLM Facade

The `LLM` class is the single interface to all providers. Model prefix auto-routes:

```python
from ai_arch_toolkit import LLM

llm = LLM("claude-sonnet-4-20250514")  # → Anthropic
llm = LLM("gpt-4o")                    # → OpenAI
llm = LLM("gemini-2.0-flash")          # → Gemini
llm = LLM("grok-2")                    # → xAI
```

### Core methods

```python
# Simple completion
response = await llm.complete("What is 2+2?")
response = await llm.complete(messages, system="You are helpful.")

# With tools
response = await llm.complete(messages, tools=my_tool_group)

# Structured output (Pydantic model or OutputSchema)
response = await llm.complete(messages, output_schema=MyModel)
response.parsed  # → MyModel instance

# Extended thinking (Anthropic)
response = await llm.complete(messages, thinking=True, thinking_budget=5000)
response.thinking  # → tuple of ThinkingBlock

# JSON mode
response = await llm.complete(messages, json_mode=True)

# Streaming (text chunks)
async for chunk in await llm.stream(messages):
    print(chunk, end="")

# Streaming (structured events)
async for event in await llm.stream_events(messages, tools=tools):
    match event.kind:
        case "text": print(event.text, end="")
        case "thinking": print(f"[thinking] {event.thinking.text}")
        case "tool_call": print(f"[tool] {event.tool_call.name}")

# Sync versions
response = llm.complete_sync("Hello")
for chunk in llm.stream_sync("Hello"):
    print(chunk, end="")
```

### Response

Every LLM call returns a `Response`:

```python
response.text           # answer text
response.tool_calls     # tuple of ToolCall(id, name, input)
response.thinking       # tuple of ThinkingBlock (extended thinking)
response.parsed         # structured output (if output_schema used)
response.usage          # Usage(input_tokens, output_tokens, cache_write_tokens, cache_read_tokens)
response.cost           # estimated USD (from pricing registry)
response.stop_reason    # "end_turn", "tool_use", "max_tokens", etc.
response.model          # actual model used
response.citations      # tuple of Citation (web search results)
response.attempts       # tuple of Attempt (retry/fallback history)
response.has_tool_calls # bool shorthand
response.to_message()   # convert to assistant message dict
```

### Fallback chains

```python
llm = LLM(
    "claude-opus-4-0-20250514",
    fallback=["claude-sonnet-4-20250514", "gpt-4o"],
    fallback_on=(APIError, TimeoutError),  # default
)

# If Opus fails → tries Sonnet → tries GPT-4o
# response.attempts records what happened at each step
```

### Retry

```python
from ai_arch_toolkit import RetryConfig

llm = LLM(
    "claude-sonnet-4-20250514",
    retry=RetryConfig(
        max_retries=3,
        base_delay=1.0,      # exponential backoff
        max_delay=60.0,
        retry_on_status=(429, 500, 502, 503, 504),
    ),
)
```

### Token counting

```python
token_count = await llm.count_tokens(messages, system="...", tools=tools)
# or sync:
token_count = llm.count_tokens_sync(messages)
```

---

## Tools

### Defining tools

The `@tool` decorator auto-generates JSON Schema from type hints and Google-style docstrings:

```python
from ai_arch_toolkit import tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: City name to look up.
        units: Temperature units — celsius or fahrenheit.
    """
    # ... implementation ...
    return f"Weather in {city}: 22°{units[0].upper()}"
```

The decorator attaches a `__tool_definition__` (a `ToolDefinition` holding a `.schema` and a runtime `.policy`). Only `.schema` is sent to the LLM; the policy stays server-side.

For custom schemas:

```python
@tool(name="custom_name", schema={"properties": {"x": {"type": "integer"}}})
def my_tool(x: int) -> str:
    ...
```

### ToolGroup

Bundle tools together for agents:

```python
from ai_arch_toolkit import ToolGroup

tools = ToolGroup(get_weather, search_wikipedia, run_command)

tools.definitions    # list of JSON Schema dicts (sent to LLM)
len(tools)           # 3
"get_weather" in tools  # True

# Manual execution — both return a structured ToolResult
result = tools.execute(tool_call)              # sync
result = await tools.async_execute(tool_call)  # async
result.ok                                      # True/False
text = result.to_model_text()                  # string for the LLM tool-result message
```

### Pre-built tools

All use stdlib only (zero pip dependencies). All return error strings instead of raising exceptions.

```python
from ai_arch_toolkit import (
    # Datetime
    datetime_now, timezone_convert,
    # Math
    math_eval, unit_convert,
    # Filesystem
    read_file, list_directory, search_files,
    # Text
    text_stats, regex_search, base64_encode, base64_decode,
    # Web
    http_get, scrape_text,
    # Weather (Open-Meteo, no API key needed)
    get_weather, get_forecast,
    # Knowledge
    wikipedia_search, wikipedia_article, define_word,
    # Geo
    geocode, ip_lookup, country_info,
    # JSON/CSV
    json_extract, csv_read,
    # News
    hacker_news,
    # Shell
    run_command,
)

# Combine into a ToolGroup for a flow
tools = ToolGroup(get_weather, wikipedia_search, math_eval, datetime_now)
flow = react_flow(llm, tools)
```

### Server tools

Provider-hosted tools (run on the provider's infrastructure):

```python
from ai_arch_toolkit import web_search, code_execution

response = await llm.complete(
    "Search for recent news about AI",
    tools=[web_search()],  # provider executes the search
)
response.citations  # web search results with URLs

response = await llm.complete(
    "Calculate fibonacci(20) using Python",
    tools=[code_execution()],  # provider runs the code
)
```

### run_tools helper

Execute all tool calls from a response in one call:

```python
from ai_arch_toolkit import run_tools

response = await llm.complete(messages, tools=tools)
if response.has_tool_calls:
    tool_results = await run_tools(response, tools)
    # tool_results is a list of tool_result() message dicts
    # ready to append to messages for the next LLM call
```

---

## Middleware

Middleware hooks into every LLM call — before the request and after the response:

```python
from ai_arch_toolkit import Middleware, Request, Response

class CostTracker:
    """Track cumulative cost across all LLM calls."""

    def __init__(self):
        self.total_cost = 0.0

    def before(self, request: Request) -> Request:
        # Modify the request (add context, filter messages, etc.)
        return request

    def after(self, request: Request, response: Response) -> Response:
        self.total_cost += response.cost or 0.0
        print(f"Call cost: ${response.cost:.4f} | Total: ${self.total_cost:.4f}")
        return response

tracker = CostTracker()
llm = LLM("claude-sonnet-4-20250514", middleware=[tracker])
```

### Request object

```python
request.messages    # list of message dicts
request.system      # system prompt (str | None)
request.tools       # tool definitions (list | None)
request.model       # model name
request.kwargs      # extra provider kwargs
```

### Async middleware

If your middleware needs async operations (database lookups, API calls), implement `abefore` / `aafter`:

```python
class AsyncMiddleware:
    async def abefore(self, request: Request) -> Request:
        context = await fetch_from_database(request.messages[-1])
        # ... modify request ...
        return request

    async def aafter(self, request: Request, response: Response) -> Response:
        await log_to_database(response)
        return response
```

The framework auto-detects async variants and falls back to sync if not present.

### Middleware execution order

`before` hooks run **in order** (first middleware first). `after` hooks run **in reverse** (last middleware first). This creates an onion-like wrapping:

```
Request  → MW1.before → MW2.before → MW3.before → Provider
Response ← MW1.after  ← MW2.after  ← MW3.after  ← Provider
```

**Example** — a logger wrapping a cost tracker:

```python
class Logger:
    def before(self, req: Request) -> Request:
        print(f"[log] Sending {len(req.messages)} messages")
        return req
    def after(self, req: Request, res: Response) -> Response:
        print(f"[log] Got {res.usage.output_tokens} tokens")
        return res

class CostGuard:
    def __init__(self, budget: float):
        self.spent = 0.0
        self.budget = budget
    def after(self, req: Request, res: Response) -> Response:
        self.spent += res.cost or 0.0
        if self.spent > self.budget:
            raise RuntimeError(f"Budget exceeded: ${self.spent:.2f}")
        return res

llm = LLM("claude-sonnet-4-20250514", middleware=[Logger(), CostGuard(1.00)])
# Request:  Logger.before → CostGuard.before (no-op) → Provider
# Response: CostGuard.after → Logger.after ← Provider
```

---

## Memory

Graph-backed memory for agents. Built on the `core/graph/` layer.

### GraphStore

```python
from ai_arch_toolkit import GraphStore, Node
from ai_arch_toolkit.core.graph import Graph

graph = Graph()       # in-memory graph (NetworkX backend)
store = GraphStore(graph.backend)

# Store a memory
node = await store.add(Node(
    type="fact",
    content={"text": "The capital of France is Paris"},
    source="user",
    confidence=0.95,
))

# Retrieve (auto-tracks access count and last_accessed)
node = await store.get(node.id)

# Search
results = await store.keyword_search("capital France", limit=5)
for result in results:
    print(f"{result.node.content['text']} (score: {result.score:.2f})")
```

### Memory Node fields

```python
Node(
    id="auto-generated",       # 16-char hex
    type="fact",               # node type (for filtering)
    content={"text": "..."},   # searchable key-value pairs
    metadata={},               # arbitrary metadata
    embedding=None,            # vector embedding (for similarity search)
    timestamp=datetime.now(),  # effective time
    created_at=datetime.now(), # insertion time
    access_count=0,            # bumped on every get()
    last_accessed=None,        # updated on every get()
    confidence=1.0,            # 0.0–1.0
    source="unknown",          # provenance
)
```

### Views

Views provide structured access patterns over the memory store:

```python
from ai_arch_toolkit import TemporalView, RelationalView, PropertyView, SimilarityView

# Time-based queries
temporal = TemporalView(store)
recent = await temporal.recent(limit=10)
since_yesterday = await temporal.since(yesterday)

# Graph-based queries
relational = RelationalView(store)
neighbors = await relational.neighbors(node_id)
path = await relational.path(from_id, to_id)

# Property-based queries
props = PropertyView(store)
trusted = await props.by_confidence(min_confidence=0.8)
from_user = await props.by_source("user")
most_used = await props.most_accessed(limit=5)

# Vector similarity (requires embed function + index)
similarity = SimilarityView(store)
similar = await similarity.search(query_embedding, limit=10)
```

### MemoryMiddleware

Auto-injects relevant memories into LLM context on every call:

```python
from ai_arch_toolkit import MemoryMiddleware

memory_mw = MemoryMiddleware(store, max_results=5, threshold=0.3)
llm = LLM("claude-sonnet-4-20250514", middleware=[memory_mw])

# Now every LLM call automatically gets relevant memories injected
```

### Presets

```python
from ai_arch_toolkit.toolkit.memory import conversational, cognitive

# Conversational: optimized for chat-style memory
store = conversational(backend)

# Cognitive: optimized for knowledge graph patterns
store = cognitive(backend)
```

### memory_tools

Generate `@tool`-decorated functions so agents can manage their own memory:

```python
from ai_arch_toolkit.toolkit.memory import memory_tools

recall, remember, forget = memory_tools(store)
tools = ToolGroup(recall, remember, forget, get_weather)

flow = react_flow(llm, tools)
# Agent can now: recall("what did we discuss?"), remember("user likes Python"), forget(node_id)
```

---

## Knowledge Registry

Sync in-memory registry for prompt-injectable reference data. Use it for domain knowledge, style guides, few-shot examples, or any structured context.

```python
from ai_arch_toolkit import KnowledgeRegistry

registry = KnowledgeRegistry()

# Register entries
registry.register(
    "company_style",
    content="Always use Oxford commas. Avoid passive voice.",
    category="style",
    tags=frozenset({"writing", "formatting"}),
)

registry.register(
    "api_reference",
    content='{"endpoints": ["/users", "/posts"]}',
    format="json",
    category="technical",
)

# Query
style_guides = registry.by_category("style")
writing_docs = registry.by_tags("writing", "formatting")

# Inject into prompts
context = registry.as_context("company_style", "api_reference", separator="\n---\n")
# → "Always use Oxford commas. Avoid passive voice.\n---\n{\"endpoints\": ...}"

response = await llm.complete(
    "Write API documentation",
    system=f"Follow these guidelines:\n{context}",
)
```

### Loaders

Load knowledge from files:

```python
from ai_arch_toolkit.toolkit.knowledge import (
    load_text, load_json, load_yaml, load_toml, load_markdown, load_directory,
)

# Single file
entries = load_text("style-guide.txt")

# Structured data (nested keys become separate entries)
entries = load_json("api-spec.json")
entries = load_yaml("config.yaml")
entries = load_toml("settings.toml")

# Markdown (sections become entries)
entries = load_markdown("docs/reference.md")

# Bulk load directory
entries = load_directory("knowledge/", recursive=True)
```

---

## Content and Messages

### Message constructors

```python
from ai_arch_toolkit import user, assistant, system, tool_result

messages = [
    system("You are a helpful assistant."),
    user("What's the weather?"),
    assistant("Let me check that for you."),
    tool_result("22°C and sunny", tool_use_id="call_123", name="get_weather"),
]
```

### Multimodal content

```python
from ai_arch_toolkit import user, image, document, cache

# Image (URL, base64, or bytes)
messages = [user(["Describe this image:", image("https://example.com/photo.jpg")])]
messages = [user(["Describe this:", image(raw_bytes, media_type="image/png")])]

# PDF document
messages = [user(["Summarize this:", document("report.pdf", media_type="application/pdf")])]

# Anthropic prompt caching
messages = [user([cache(long_context), "Now answer my question."])]
```

### Content type

```python
type ContentPart = str | ImagePart | DocumentPart | CachePart
type Content = str | list[ContentPart]
```

All agents accept `Content` as their task input — so you can pass images and documents to any agent flow.

---

## Pricing and Cost Tracking

### Automatic cost estimation

Every `Response` includes an estimated cost:

```python
response = await llm.complete("Hello")
print(f"${response.cost:.6f}")  # e.g. $0.000342
```

Costs are calculated from a built-in pricing registry (`_default_pricing.toml`) that covers all supported models.

### Pricing registry

```python
from ai_arch_toolkit import pricing

# Check if a model has pricing
pricing.has("claude-sonnet-4-20250514")  # True

# Get pricing details
p = pricing.get("claude-sonnet-4-20250514")
p.input   # USD per 1M input tokens
p.output  # USD per 1M output tokens

# Estimate cost
cost, known = pricing.estimate_cost(
    "claude-sonnet-4-20250514",
    input_tokens=1000,
    output_tokens=500,
)

# Register custom pricing
from ai_arch_toolkit.core._pricing import ModelPricing
pricing.register("my-model", ModelPricing(input=1.0, output=3.0))

# List all models
pricing.list_models()
```

### Cost tracking across flows

Flow results accumulate cost across all steps:

```python
result = await flow.run(state)
print(f"Total cost: ${result.total_cost:.4f}")

# Per-step breakdown
for st in result.trace.steps:
    print(f"  {st.name}: ${st.cost:.4f}, {st.duration:.1f}s")
```

---

## Structured Output

Force the LLM to return data matching a schema:

```python
from pydantic import BaseModel
from ai_arch_toolkit import LLM, OutputSchema

class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str

# With Pydantic model
response = await llm.complete("Weather in Paris", output_schema=WeatherReport)
report = response.parsed  # → WeatherReport(city="Paris", temperature=22.0, ...)

# With OutputSchema (manual JSON Schema)
schema = OutputSchema(
    name="weather",
    schema={"type": "object", "properties": {"city": {"type": "string"}}},
)
response = await llm.complete("Weather in Paris", output_schema=schema)
```

---

## Extended Thinking

Anthropic models support extended thinking — the model reasons through the problem before answering:

```python
response = await llm.complete(
    "Solve this step by step: what is 127 * 389?",
    thinking=True,
    thinking_budget=5000,  # max thinking tokens
)

# Access thinking trace
for block in response.thinking:
    print(f"[Thinking] {block.text}")

print(f"Answer: {response.text}")
```

---

## Putting It Together

### Research agent with memory and knowledge

```python
from ai_arch_toolkit import (
    LLM, ToolGroup, KnowledgeRegistry, GraphStore, MemoryMiddleware,
    get_weather, wikipedia_search, datetime_now,
)
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.memory import memory_tools
from ai_arch_toolkit.core.graph import Graph

# Knowledge base
knowledge = KnowledgeRegistry()
knowledge.register("research_guidelines", content="Always cite sources. Verify claims.")
context = knowledge.as_context("research_guidelines")

# Memory
graph = Graph()
store = GraphStore(graph.backend)
recall, remember, forget = memory_tools(store)
memory_mw = MemoryMiddleware(store, max_results=5)

# LLM with memory middleware
llm = LLM(
    "claude-sonnet-4-20250514",
    middleware=[memory_mw],
    fallback="gpt-4o",
)

# Agent flow with tools + memory tools
tools = ToolGroup(get_weather, wikipedia_search, datetime_now, recall, remember)
flow = react_flow(
    llm, tools,
    system=f"You are a research assistant.\n\n{context}",
    max_iterations=15,
)

state = State(operational=react_initial_state(
    "Research the history of the Eiffel Tower and remember key facts"
))
result = await flow.run(state)
```

### Multi-model pipeline with cost control

```python
from ai_arch_toolkit import LLM, ToolGroup
from ai_arch_toolkit.core import State
from ai_arch_toolkit.toolkit.agents.flows import plan_execute_flow, plan_execute_initial_state

fast = LLM("claude-haiku-4-5-20251001")
smart = LLM("claude-opus-4-0-20250514")

flow = plan_execute_flow(
    fast,  # default model
    tools,
    planner_llm=fast,     # cheap model plans
    exec_llm=fast,        # cheap model executes
    solver_llm=smart,     # smart model synthesizes
)

state = State(operational=plan_execute_initial_state("Complex research task"))
result = await flow.run(state)
print(f"Total cost: ${result.total_cost:.4f}")
```

### Parallel research with Flow composition

```python
from ai_arch_toolkit.core import LLM, State, Step, Result, StateSnapshot
from ai_arch_toolkit.toolkit.flow import Flow, FlowStep
from ai_arch_toolkit.toolkit.agents.flows import react_flow, react_initial_state

llm = LLM("claude-sonnet-4-20250514")

async def research_topic(topic_key: str):
    """Create a step that researches a topic via inner ReAct."""
    async def _research(snap: StateSnapshot) -> Result:
        task = snap.require("task")
        inner = react_flow(llm, tools, system=f"Research: {task}", max_iterations=5)
        state = State(operational=react_initial_state(task))
        result = await inner.run(state)
        response = state.get("response")
        return Result(
            value=response.text if response else "",
            artifacts={topic_key: response.text if response else ""},
            cost=result.total_cost,
        )
    return Step(name=topic_key, fn=_research)

flow = Flow(
    FlowStep(step=await research_topic("tech")),
    FlowStep(step=await research_topic("market")),
    FlowStep(
        step=Step(name="synthesize", fn=synthesize),
        after=("tech", "market"),  # runs after both complete
    ),
    name="parallel_research",
)

state = State(operational={"task": "Electric vehicle batteries"})
result = await flow.run(state)
```

---

## Quick Reference: What Enhances Agent Flows

| Capability | How it connects | Example |
|---|---|---|
| **Fallback chains** | LLM-level, transparent to flows | `LLM("opus", fallback="sonnet")` |
| **Retry** | LLM-level, exponential backoff | `LLM("opus", retry=RetryConfig(max_retries=3))` |
| **Middleware** | Hooks into every LLM call | Cost tracking, logging, memory injection |
| **Memory** | `MemoryMiddleware` + `memory_tools()` | Agents remember across conversations |
| **Knowledge** | Injected into system prompts | Domain context, style guides |
| **Pre-built tools** | 25 ready-to-use tools | Weather, Wikipedia, math, filesystem |
| **Server tools** | Provider-hosted web search, code execution | `tools=[web_search()]` |
| **Structured output** | `output_schema` on LLM call | Pydantic models as output |
| **Extended thinking** | `thinking=True` on LLM call | Anthropic reasoning traces |
| **Multimodal input** | `Content` accepts images, PDFs | Vision + tools agents |
| **Per-phase models** | Factory kwargs (`planner_llm=...`) | Cheap planner, smart solver |
| **Cost tracking** | Automatic on every Response + FlowResult | `result.total_cost` |
| **Token counting** | `llm.count_tokens()` | Budget estimation before running |
| **Pricing registry** | Built-in model pricing | `pricing.estimate_cost(...)` |
| **Flow streaming** | `flow.iter()` / `flow.iter_sync()` | Real-time progress updates |
| **Flow composition** | `flow.as_step()` / nested Flows | Agents inside agents |
| **Policy** | Per-step retry, timeout, confidence | `Step(policy=Policy(...))` |
| **Trace** | Full execution history | `result.trace.steps` |
