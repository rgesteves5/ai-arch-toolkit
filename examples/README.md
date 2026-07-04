# Examples

Thirty-six runnable scripts that walk through every public surface in the
toolkit, from a one-line completion to a Tree-of-Thoughts agent with custom
middleware. Each file is self-contained — pick a number, copy-paste, run.

## Run an example

```bash
uv sync --extra dev                              # installs every provider
set -a && source .env && set +a                  # load API keys (see .env.example)
uv run python examples/01_hello_world.py
```

Examples default to inexpensive models (`gpt-4.1-nano`, `claude-haiku-4-5`,
etc.) — total cost to run the entire suite is well under $1 with current
pricing.

API keys: each example states which provider it expects. Set the matching env
var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or
`XAI_API_KEY`). Pure-Python examples (no LLM call) work without any key.

## By topic

### 🟢 Start here — core LLM

The shortest path from zero to a working call, plus the basic building blocks.

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 01 | [`01_hello_world.py`](01_hello_world.py) | One completion, inspect text and usage | OpenAI |
| 02 | [`02_multi_turn_conversation.py`](02_multi_turn_conversation.py) | Conversation history with `user()` / `assistant()` | OpenAI |
| 03 | [`03_streaming.py`](03_streaming.py) | `stream_sync()` and accessing `.response` after consumption | OpenAI |
| 04 | [`04_structured_output.py`](04_structured_output.py) | Pydantic-typed responses via `output_config` | OpenAI |
| 07 | [`07_thinking.py`](07_thinking.py) | Extended reasoning (Claude / Gemini thinking blocks) | Anthropic |
| 08 | [`08_async.py`](08_async.py) | `await llm.complete(...)` and concurrent calls | OpenAI |
| 11 | [`11_multimodal.py`](11_multimodal.py) | Image + text input | OpenAI |

### 📡 Streaming deep-dive

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 20 | [`20_rich_streaming_events.py`](20_rich_streaming_events.py) | `stream_events()` — typed events for text, thinking, tool calls | OpenAI |
| 21 | [`21_stream_fallback.py`](21_stream_fallback.py) | Stream fallback across providers when the primary fails | OpenAI |

### 🔧 Tools

From hand-written tool dicts to the `@tool` decorator and the 25 pre-built
toolkit tools.

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 05 | [`05_tool_calling.py`](05_tool_calling.py) | Manual tool definition + `tool_result()` round-trip | Anthropic |
| 06 | [`06_tool_loop.py`](06_tool_loop.py) | `@tool` + `ToolGroup` + `run_tools_sync` loop | OpenAI |
| 24 | [`24_toolkit_tools_showcase.py`](24_toolkit_tools_showcase.py) | Tour of pre-built tools (weather, geo, wiki, …) | OpenAI |
| 25 | [`25_server_tools.py`](25_server_tools.py) | Provider-hosted tools: `web_search()`, `code_execution()` | OpenAI |

### 🛡️ Reliability

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 22 | [`22_retry_config.py`](22_retry_config.py) | `RetryConfig` exponential backoff on transient errors | OpenAI |
| 23 | [`23_prompt_caching.py`](23_prompt_caching.py) | Anthropic prompt cache breakpoints + cost savings | Anthropic |
| 36 | [`36_fallback_chains_and_attempts.py`](36_fallback_chains_and_attempts.py) | Multi-provider fallback with attempt tracking | OpenAI |

### 🤖 Agent flows

All nine built-in agent architectures, plus tool / multimodal / structured
output / middleware variants of ReAct.

| # | File | Pattern | Key |
|---|------|---------|-----|
| 09 | [`09_react_agent.py`](09_react_agent.py) | ReAct: thought → tool → observation loop | OpenAI |
| 10 | [`10_react_agent_streaming.py`](10_react_agent_streaming.py) | ReAct with rich streaming events | OpenAI |
| 12 | [`12_react_agent_multimodal.py`](12_react_agent_multimodal.py) | ReAct over image + text input | OpenAI |
| 13 | [`13_structured_output_agent.py`](13_structured_output_agent.py) | ReAct producing a typed Pydantic answer | OpenAI |
| 14 | [`14_middleware_agent.py`](14_middleware_agent.py) | Tracing + custom middleware inside a flow | OpenAI |
| 15 | [`15_reflexion_agent.py`](15_reflexion_agent.py) | Reflexion: inner ReAct + evaluator + reflect retry | OpenAI |
| 16 | [`16_rewoo_agent.py`](16_rewoo_agent.py) | ReWOO: plan → execute → solve, three phases | OpenAI |
| 17 | [`17_plan_execute_agent.py`](17_plan_execute_agent.py) | Plan-Execute: numbered plan, per-step ReAct, solve | OpenAI |
| 18 | [`18_tot_agent.py`](18_tot_agent.py) | Tree of Thoughts (DFS / BFS search) | OpenAI |
| 19 | [`19_lats_agent.py`](19_lats_agent.py) | LATS — Language Agent Tree Search (MCTS) | OpenAI |
| 26 | [`26_self_discovery_agent.py`](26_self_discovery_agent.py) | Self-Discovery: select reasoning modules → adapt → solve | OpenAI |
| 27 | [`27_llm_compiler_agent.py`](27_llm_compiler_agent.py) | LLMCompiler: plan DAG → parallel execute → join | OpenAI |

> **Generate-Review flow** (`generate_review_flow`) isn't in the numbered
> examples yet — see `tests/agents/flows/test_generate_review.py` for usage.

### 🧠 Memory

`GraphStore`-backed agent memory, plus convenience tools/middleware.

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 28 | [`28_memory_graph_basics.py`](28_memory_graph_basics.py) | `GraphStore`: nodes, edges, search, views, persistence | None |
| 29 | [`29_memory_middleware.py`](29_memory_middleware.py) | `MemoryMiddleware` — auto-retrieve + record around LLM calls | OpenAI |
| 30 | [`30_memory_agent_tools.py`](30_memory_agent_tools.py) | `memory_tools()` exposed to a ReAct flow | OpenAI |

### 🔁 Flow orchestration

The `Flow` primitives that the agent architectures are built on.

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 31 | [`31_flow_basics.py`](31_flow_basics.py) | `Step` / `FlowStep` / `Flow` / `Result` + state artifacts | None |
| 32 | [`32_flow_streaming.py`](32_flow_streaming.py) | `flow.iter()` / `iter_sync()` and `FlowEvent` inspection | None |
| 33 | [`33_flow_with_llm.py`](33_flow_with_llm.py) | Flow steps calling an `LLM` with cost accumulation | OpenAI |

### 📚 Knowledge registry

Prompt-injectable reference data.

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 34 | [`34_knowledge_registry.py`](34_knowledge_registry.py) | `KnowledgeRegistry`, categories, `as_context()` | None |
| 35 | [`35_knowledge_loaders.py`](35_knowledge_loaders.py) | File and directory loaders (text, json, toml, yaml, markdown) | None |

### 💰 Budgets & metering

Measure what a run costs, and cap it.

| # | File | What it shows | Key |
|---|------|---------------|-----|
| 37 | [`37_budgets_and_metering.py`](37_budgets_and_metering.py) | `result.meter`, `BudgetPolicy` (construction + per-run), `budget_scope`, event audit | OpenAI |

## Suggested reading order

For a guided tour of the framework, walk the examples in this order rather
than strict numerical order:

1. **Get a call working** — 01, 02, 03 (then 11 if you care about images).
2. **Add structure** — 04 (structured output), 07 (thinking), 08 (async).
3. **Bring in tools** — 06, then 24 and 25.
4. **Step up to agents** — 09, 10, 13, 14, then any of 15–27 depending on the
   architecture you need.
5. **Memory** — 28 → 29 → 30.
6. **Compose your own flows** — 31, 32, 33.
7. **Knowledge + reliability** — 34, 35, 22, 23, 36.
8. **Budgets & cost** — 37 (measure a run, then cap it).

For deep notes on examples 28–36, see [`EXAMPLES_REPORT.md`](EXAMPLES_REPORT.md).
