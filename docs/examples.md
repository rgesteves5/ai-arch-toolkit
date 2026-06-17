# Examples

All examples live in the `examples/` directory. Each is a self-contained script. Run with `uv run python examples/NN_name.py`. Some examples require API keys — see [UV guide](uv-guide.md).

---

## Foundations

| # | File | What it demonstrates |
|---|------|---------------------|
| 01 | `01_hello_world.py` | Create an LLM, send a single prompt, inspect response text and token usage |
| 02 | `02_multi_turn_conversation.py` | Build a conversation with plain dict messages, system prompt, and appending replies |
| 03 | `03_streaming.py` | Stream text chunks from the model, access full Response via `stream.response` |
| 04 | `04_structured_output.py` | Use `OutputSchema` to enforce strict JSON response format with `response.parsed` |
| 05 | `05_tool_calling.py` | Define a tool as a plain dict, detect model calls, execute locally, send `tool_result` back |
| 06 | `06_tool_loop.py` | `@tool` decorator for auto-generated schemas, `ToolGroup`, and `run_tools_sync` |
| 07 | `07_thinking.py` | Extended thinking with `thinking_budget` and `thinking_effort` parameters |
| 08 | `08_async.py` | Async-first LLM with `complete()`, `stream()`, and parallel requests via `asyncio.gather()` |

## Agent Flows

| # | File | What it demonstrates |
|---|------|---------------------|
| 09 | `09_react_agent.py` | `react_flow` automates the Thought → Action → Observation tool loop |
| 10 | `10_react_agent_streaming.py` | `flow.iter_sync()` for real-time flow events, debugging multi-step reasoning |
| 15 | `15_reflexion_agent.py` | `reflexion_flow` wraps ReAct in a retry loop with self-critique and reflection |
| 16 | `16_rewoo_agent.py` | `rewoo_flow` separates planning from execution using `#E{n}` evidence placeholders |
| 17 | `17_plan_execute_agent.py` | `plan_execute_flow` — Plan → Execute → Solve with optional replanning |
| 18 | `18_tot_agent.py` | `tot_flow` — Tree of Thoughts DFS/BFS search over reasoning paths |
| 19 | `19_lats_agent.py` | `lats_flow` — Monte Carlo Tree Search with ReAct rollouts and backpropagation |
| 26 | `26_self_discovery_agent.py` | `self_discovery_flow` — select reasoning modules, adapt, operationalize, solve |
| 27 | `27_llm_compiler_agent.py` | `llm_compiler_flow` — plan a DAG of tasks, execute in parallel, join results |

## Multimodal

| # | File | What it demonstrates |
|---|------|---------------------|
| 11 | `11_multimodal.py` | Send an image alongside text using the `image()` helper |
| 12 | `12_react_agent_multimodal.py` | `react_flow` with image + text input, using Wikipedia to complement visual analysis |

## Features

| # | File | What it demonstrates |
|---|------|---------------------|
| 13 | `13_structured_output_agent.py` | `react_flow` with tools returning typed JSON via `OutputSchema` |
| 14 | `14_middleware_agent.py` | LLM middleware firing on every call inside `react_flow` loop (cost logging) |
| 20 | `20_rich_streaming_events.py` | `stream_events()` with structured `StreamEvent` objects (text, thinking, tool_call) |
| 21 | `21_stream_fallback.py` | Automatic provider fallback during streaming when primary fails |
| 22 | `22_retry_config.py` | Automatic retries with exponential backoff for transient API failures |
| 23 | `23_prompt_caching.py` | Anthropic prompt caching with `cache()` for reduced latency and cost |
| 24 | `24_toolkit_tools_showcase.py` | Pre-built safe tools and explicit opt-in dangerous tools |
| 25 | `25_server_tools.py` | Provider-hosted server tools (web search, code execution) |
| 36 | `36_fallback_chains_and_attempts.py` | Fallback chains, attempt tracking across retries/fallbacks, flow-level traces |

## Memory

| # | File | What it demonstrates |
|---|------|---------------------|
| 28 | `28_memory_graph_basics.py` | GraphStore basics — store, search, views, keyword search, persistence |
| 29 | `29_memory_middleware.py` | MemoryMiddleware auto-injects memories into system prompt, auto-records interactions |
| 30 | `30_memory_agent_tools.py` | `memory_tools()` — remember, recall, explore, forget as agent-usable tools with `react_flow` |

## Flow and Knowledge

| # | File | What it demonstrates |
|---|------|---------------------|
| 31 | `31_flow_basics.py` | Define Steps, compose into a Flow, inspect FlowResult trace and artifacts |
| 32 | `32_flow_streaming.py` | `flow.iter()` / `iter_sync()` for step-by-step streaming |
| 33 | `33_flow_with_llm.py` | Flow calling LLM in each step, accumulating cost and tokens |
| 34 | `34_knowledge_registry.py` | Register reference data, filter by category/tags, build prompt context |
| 35 | `35_knowledge_loaders.py` | Load knowledge from files (text, JSON, TOML, Markdown) and directories |
