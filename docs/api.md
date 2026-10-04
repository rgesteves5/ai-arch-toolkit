# API Documentation

## Auto-generate API docs with pdoc

```bash
uv sync --extra dev --extra docs
uv run pdoc ai_arch_toolkit -o site/api
uv run pdoc ai_arch_toolkit --port 8080  # serve locally
```

## Public API Surface

This page summarizes the main public API. For exhaustive symbol-level reference,
generate the pdoc site above.

Most public types are re-exported from `ai_arch_toolkit` (top-level) or from
`ai_arch_toolkit.core` / `ai_arch_toolkit.toolkit`; the rest come from their own subpackage,
such as `ai_arch_toolkit.toolkit.resources` or `ai_arch_toolkit.toolkit.knowledge`. Among them
are the recommended entry point, `Agent` (see [Toolkit — Agents](#toolkit-agents)), and
`generate_review_flow`: import them from `ai_arch_toolkit.toolkit.agents`.

### Core — LLM & Providers

| Symbol | Module | Description |
|--------|--------|-------------|
| `LLM` | `_llm.py` | Unified facade. `complete()` / `stream()` / `stream_events()`, batch helpers, token counting, and `_sync` wrappers |
| `Response` | `_response.py` | LLM response with `text`, `tool_calls`, `usage`, `cost` |
| `Attempt`, `Usage`, `ToolCall`, `ThinkingBlock`, `Citation` | `_response.py` | Response components and attempt tracking |
| `OutputSchema` | `_response.py` | Structured output constraint |
| `StreamEvent`, `RichStreamResponse` | `_response.py` | Streaming types |
| `Content`, `ContentPart` | `_content.py` | `str | list[ContentPart]` message content |
| `user()`, `assistant()`, `system()`, `tool_result()` | `_content.py` | Message constructors |
| `ImagePart`, `DocumentPart`, `CachePart` | `_content.py` | Multimodal content parts |
| `Middleware`, `Request` | `_middleware.py` | Before/after hooks protocol |
| `RateLimitMiddleware`, `TracingMiddleware` | `_rate_limit.py`, `_telemetry.py` | Built-in middleware for rate limiting and tracing |
| `RetryConfig` | `_retry.py` | Exponential backoff configuration |
| `pricing` | `_pricing.py` | Per-model pricing registry singleton |
| `count_tokens_local()`, `count_tokens_local_batch()`, `chars_to_tokens()`, `tokens_to_chars()` | `_tokens.py` | Local token estimation helpers |
| `ServerTool`, `code_execution()`, `web_search()` | `_server_tools.py` | Provider-hosted tools |
| `BatchRequest`, `BatchResult` | `_batch.py` | Batch API types |
| `ProviderError`, `RequestError`, `UnpricedModelError`, `APIError`, `RateLimitError`, `TransportError`, `ProviderTimeout`, `ResponseError` | `_exceptions.py` | Provider failures. `ProviderError` is the base and carries `delivery` (`not_sent`/`unbilled`/`indeterminate`); `RequestError` (also a `ValueError`): nothing was sent, and its `UnpricedModelError`: a metered call to a model the scope cannot price; `APIError` (`status_code`, `body`) and its `RateLimitError` (`retry_after`); `TransportError` (also a `ConnectionError`); `ProviderTimeout` (also a `TimeoutError`); `ResponseError`: a response with no usable result. See [Provider errors](llm.md#provider-errors) |
| `Moderator`, `ModerationResult`, `ModerationError` | `_moderation.py` | Core moderation protocol and results |

### Core — Tools

| Symbol | Description |
|--------|-------------|
| `@tool` | Decorator: auto-generates JSON Schema from type hints + docstrings |
| `ToolGroup` | Collection with `execute()` / `async_execute()`, its own gates, approval handler, and `max_calls` |
| `ToolResult`, `ToolError` | Structured outcome of one tool call; `execute()` never raises on tool failure |
| `execute_tool()`, `async_execute_tool()` | Run one `ToolCall` against a list of callables through the governed executor |
| `run_tools()`, `run_tools_sync()` | Run every tool call in a `Response` and return `tool_result` messages (a toolkit helper: import it from `ai_arch_toolkit` or `ai_arch_toolkit.toolkit`) |
| `ToolGate`, `ExecutionContext`, `GateResult`, `GateBlock`, `GateModify`, `GateDryRun` | Protocol and results for custom pre-execution gates |
| `ApprovalGate`, `DangerousToolGate`, `ApprovalHandler`, `ApprovalRequest`, `ApprovalDecision` | Built-in gates and the human-approval contract |
| `ToolRuntimePolicy` | Risk metadata, capability, output cap (`max_output_chars`), and deadline (`timeout_s`) that `@tool(...)` attaches to a tool |
| `Range` | Inclusive bounds for a numeric parameter, `Annotated[int, Range(1, 25)]`: in the schema, and enforced before the call |
| `infer_schema()` | Manual schema inference from a callable |
| `prepare_tools()` | Normalize tools (`@tool` functions, dicts, `ToolGroup`s) into the provider-facing definition dicts the adapters take |

### Core — Graph

| Symbol | Description |
|--------|-------------|
| `Graph` | Primary facade — async-first with `_sync` wrappers |
| `Node[T]` (exported as `GraphNode`) | Generic typed node: `id`, `type`, `content: T`, `metadata` |
| `Edge` (exported as `GraphEdge`) | Directed edge: `source`, `target`, `relation`, `weight`, `metadata` |
| `NodeID`, `NodeType`, `Direction` | Type aliases (`Direction` from `ai_arch_toolkit.core.graph`) |
| `GraphBackend` | Protocol — storage interface |
| `GraphAlgorithms` | Protocol — optional algorithms (from `ai_arch_toolkit.core.graph`) |
| `NetworkXBackend` | Default in-memory backend (import from `core.graph._networkx`) |

**Graph facade methods** (all but `to_dict` have `_sync` counterparts):

- **Node ops**: `add`, `get`, `update`, `remove`, `list`, `count`, `has`, `degree`, `add_many`, `remove_many`, `clear`
- **Edge ops**: `connect`, `disconnect`, `edges`, `neighbors`, `get_edges_between`, `list_edges`
- **Queries**: `node_count`, `edge_count`, `is_empty`, `filter_nodes`, `filter_edges`, `get_orphan_nodes`, `get_stats`
- **Algorithms** (require `GraphAlgorithms` backend): `bfs`, `dfs`, `shortest_path`, `find_all_paths`, `get_ancestors`, `get_descendants`, `get_subgraph`, `get_ego_graph`, `pagerank`, `centrality`, `connected_components`
- **Persistence**: `save`, `load`, `to_dict`, `from_dict`, `copy`

### Core — Flow Primitives

| Symbol | Description |
|--------|-------------|
| `State`, `StateSnapshot`, `MergeStrategy` | 4-layer mutable state container |
| `Step`, `StepFn`, `Result` | Named async functions with structured output |
| `Policy` | Retry, timeout, confidence thresholds, cost limits |
| `Trace`, `StepTrace`, `PolicyDecision`, `TraceCapture` | Execution records; `TraceCapture` sets what each step's record keeps |
| `execute_step()` | Single-step execution with policy enforcement |

### Core — Metering

The neutral meter under every budget. See [Cumulative budgets](safety.md#cumulative-budgets).

| Symbol | Description |
|--------|-------------|
| `MeterScope` | Context manager (`with` / `async with`) that meters the LLM calls (`complete`, `stream`, `stream_events`) and tool calls run inside it: `snapshot()`, and `events()` (empty unless `retain_meter_events` is set) |
| `RunConfig` | A scope's settings: `controller`, `sinks`, `pricer`, `redactor`, `retain_meter_events`, … |
| `MeterSnapshot`, `UsageEvent`, `UsageSink` | Cumulative counters, one metered operation, and the audit-sink protocol |
| `AdmissionController`, `AdmissionDecision`, `AdmissionDenied` | The enforcement hook a scope's `controller` implements, and the terminal error of a denied call |
| `Cost`, `CostKind`, `Money`, `Pricer` | Typed cost and the pricing protocol a scope uses |

### Toolkit — Flow Orchestration

| Symbol | Description |
|--------|-------------|
| `Flow` | Composes Steps into sequential, cyclic, or DAG execution graphs |
| `FlowStep` | Wraps a Step with optional `when` conditions and `after` dependencies |
| `FlowResult` | Total cost, duration, usage, full Trace, and the run's `meter` report |
| `FlowEvent` | Streaming events: `flow_start`, `flow_end`, `step_start`, `step_end` (with the step's `step_trace`), `step_skipped`, `retry`, `timeout`, `fallback`, `policy_decision`, and `llm_event` (an LLM call's stream event, in an iterated run) |
| `Scope` | Controls what keys a Step can see (include/exclude/transform/enrich) |
| `execute_flow()`, `iter_flow()` | Execution and streaming entry points |
| `FlowExecution`, `SyncFlowExecution` | What `Flow.iter()` / `iter_sync()` return: iterate the events, then read `.result`; close them (`async with` / `with`) to stop a run early |

### Toolkit — Agents

The recommended entry point. Import these from `ai_arch_toolkit.toolkit.agents`; they are not
re-exported from `ai_arch_toolkit` or `ai_arch_toolkit.toolkit`. See
[Agent & ReasoningSpec](agents.md).

| Symbol | Description |
|--------|-------------|
| `Agent` | Binds a `ReasoningSpec` to an `LLM`, a `ToolGroup`, and per-phase `deps`, and compiles the `Flow` once: `run()` / `run_sync()`, `iter()`, `as_step()`, `Agent.from_flow()` |
| `ReasoningSpec` | Frozen, declarative description of how an agent reasons (`from_mapping()` builds one from plain config data): `strategy`, `system`, `max_iterations`, `knobs`, `policy`, `timeout`, `trace_capture`, `llm_kwargs`, `output_schema` |
| `AgentResult` | Outcome of a run: `text`, `response`, `flow_result`, `usage`, `cost`, `report`, `errors` |
| `load_agent_manifest()`, `agent_from_manifest()` | Load a file-backed agent manifest, then assemble it into an `Agent` |
| `register_strategy()`, `get_strategy()`, `strategy_names()` | The strategy registry: the ten built-in strategies plus your own |

### Toolkit — Agent Flows

Import the nine factories from `ai_arch_toolkit.toolkit.agents`. All but `generate_review_flow`
(and `generate_review_initial_state`) are also re-exported from `ai_arch_toolkit` and
`ai_arch_toolkit.toolkit`.

| Flow Factory | Architecture |
|-------------|-------------|
| `react_flow()` | Thought → Action → Observation loop |
| `reflexion_flow()` | ReAct + self-critique retry |
| `rewoo_flow()` | Plan → Execute → Solve |
| `plan_execute_flow()` | Plan → per-step ReAct → Solve |
| `tot_flow()` | Tree of Thoughts — DFS/BFS |
| `lats_flow()` | MCTS + ReAct rollouts |
| `self_discovery_flow()` | Reasoning module selection → Solve |
| `llm_compiler_flow()` | DAG plan → parallel execute → join |
| `generate_review_flow()` | Generator → reviewer loop with retry feedback |

Each factory has a companion `*_initial_state(task)` helper that creates the initial operational dict for `State(operational=...)`.

### Toolkit — Budgets

The opinion layer over the core meter, from `ai_arch_toolkit` or `ai_arch_toolkit.toolkit.budget`.
See [Cumulative budgets](safety.md#cumulative-budgets).

| Symbol | Description |
|--------|-------------|
| `BudgetPolicy` | Run-wide caps: `max_llm_calls`, `max_tool_calls`, `max_input_tokens`, `max_output_tokens`, `max_total_tokens`, `max_cost`, `max_wall_s` (plus `reserve` and `unpriced` rules). Pass it to `Flow(budget_policy=...)` or to a run (`run(..., budget_policy=...)`) |
| `budget_scope()` | A `MeterScope` enforcing a `BudgetPolicy` around any code, raw `LLM` calls included (measure-only without a policy) |
| `BudgetReport` | What a run consumed and whether it went over budget; `FlowResult.meter` and `AgentResult.report` return one, and `BudgetReport.from_snapshot()` builds one from a scope's `snapshot()` |
| `BudgetController` | The `AdmissionController` that enforces a `BudgetPolicy` inside a `MeterScope` |
| `BudgetExceeded` | The `AdmissionDenied` subclass a capped call raises (`dimension`, `limit`, `current`, `attempted`; the plain `AdmissionDenied` when the meter's re-check catches a race). A flow that owns its meter reports it in the `FlowResult` instead of raising |
| `Estimator`, `HeuristicEstimator` | Estimate a call's worst case: the hold a `reserve="strict"` policy takes before each call, and the cost bound of a failed call |
| `Reserve`, `Unpriced` | The types of `BudgetPolicy.reserve` (`"none"` / `"strict"`) and `BudgetPolicy.unpriced` (`"fail_closed"` / `"allow"`) |

### Toolkit — Memory

| Symbol | Description |
|--------|-------------|
| `GraphStore` | Graph-backed memory store with search and access tracking; it and the views are async-only (no `_sync` wrappers) |
| `Node` (memory) | Memory node: adds `timestamp`, `source`, `confidence`, `embedding`, `access_count` |
| `TemporalView` | Query by recency (`recent`, `since`) |
| `RelationalView` | Graph traversal (`neighbors`, `path`) |
| `PropertyView` | Filter by `confidence`, `source`, `access_count` |
| `SimilarityView` | Vector similarity search |
| `MemoryMiddleware` | Auto-injects memories into LLM context |
| `MemoryPreset` | Preset configurations |
| `conversational()`, `cognitive()` | Built-in presets |
| `memory_tools()` | Build a `ToolGroup` of memory tools (`remember`, `recall`, `explore_memory`, `forget_memory`) for agent use |

### Toolkit — Knowledge

| Symbol | Description |
|--------|-------------|
| `KnowledgeRegistry` | In-memory store for reference data |
| `KnowledgeEntry` | Entry with `key`, `content`, `format`, `category`, `tags` |
| `KnowledgeAlreadyExistsError` | Duplicate key without explicit `overwrite=True` |
| `KnowledgeRegistry.load()` / `.from_directory()` | Resource-backed loading conveniences |
| `KnowledgeRegistry.search()` | Deterministic lexical search with explainable scores |
| `load_text()`, `load_json()`, `load_toml()`, `load_yaml()`, `load_markdown()` | File loaders |
| `load_directory()` | Bulk loader (flat or recursive) |

### Toolkit — Structured Prompts

| Symbol | Description |
|--------|-------------|
| `PromptSection` | Named content with deterministic order and stability metadata |
| `Prompt` | Immutable section collection with a configurable separator |
| `PromptTemplate`, `PromptTemplateSection` | Reusable sources, variables, and explicit templates |
| `PromptVariable` | Required/default/type/JSON-Schema variable declaration |
| `RenderedPrompt` | Exact text, ordered sections, SHA-256 fingerprint, and stable-prefix diagnostics |
| `PromptConversation`, `PromptMessage` | Ordered system/user/assistant prompts over text or multimodal `Content` |
| `RenderedPromptConversation`, `RenderedPromptMessage` | Rendered messages and plain LLM request conversion |
| `render_prompt()` | Validate and render a structured prompt |
| `load_prompt()` | Load a `.prompt.yaml`, `.prompt.json`, or `.prompt.toml` manifest |
| `TextLayout`, `MarkdownLayout`, `XmlLayout`, `JsonLayout` | Built-in section layouts |
| `SeparatorPolicy`, `SectionSpan` | Boundary separators and rendered offsets |
| `validate_cache_layout()` | Opt-in validation of a cache-optimized stability layout |
| `prompt_from_sections()` | Freeze a sequence of sections into a `Prompt` |

### Toolkit — Resources

| Symbol | Description |
|--------|-------------|
| `Resource`, `ResourceRef`, `ResourceProvenance` | Raw, decoded, parsed, and origin data |
| `ResourceResolver` | Loader/codec registry and resolution facade |
| `ResourcePolicy` | Allowed roots, size, symlink, and remote rules |
| `load_resource()`, `load_resources()` | Load a file or deterministic directory snapshot |
| `JsonPointer`, `MarkdownHeading`, `LineRange`, `NamedBlock` | Built-in selectors |
| `serialize_resource_value()` | Text/JSON/YAML/Markdown serialization |
| `SerializerRegistry` | Resolver-scoped custom serializer registration |
| `Resource.from_text()` / `.from_bytes()` | Immutable in-memory resource snapshots |

### Toolkit — Moderation

| Symbol | Description |
|--------|-------------|
| `LLMModerator` | Moderation via a regular LLM using a classification prompt |
| `ModerationMiddleware` | Middleware that moderates the latest user input and/or the response; a flag raises `ModerationError` (`on_flagged="raise"`) or logs a warning (`"warn"`) |
| `OpenAIModerator` | OpenAI Moderation API adapter, available from `ai_arch_toolkit.toolkit.moderation` |
