# ai-arch-toolkit Examples

Runnable examples showcasing core toolkit workflows.

## Prerequisites

```bash
# Install all dependencies
uv sync --dev
```

## Environment Variables

Set API keys for the providers you want to use:

| Variable            | Provider  | Required by                          |
|---------------------|-----------|--------------------------------------|
| `OPENAI_API_KEY`    | OpenAI    | 01, 04, 10, 14, 15, 16, 17, 19, 20, 21-31 |
| `ANTHROPIC_API_KEY` | Anthropic | 02, 05, 07, 13, 16                   |
| `XAI_API_KEY`       | xAI       | 06, 09, 12                           |
| `GEMINI_API_KEY`    | Gemini    | 03, 08, 11                           |

## Example Index

| #  | File                            | Provider  | Feature                              |
|----|---------------------------------|-----------|--------------------------------------|
| 01 | `01_hello_world.py`             | OpenAI    | Basic Client chat                    |
| 02 | `02_multi_turn_conversation.py` | Anthropic | Message history, system prompt       |
| 03 | `03_streaming.py`               | Gemini    | stream() + stream_events()           |
| 04 | `04_structured_output.py`       | OpenAI    | JSON schema enforcement              |
| 05 | `05_tool_calling.py`            | Anthropic | Manual Tool / ToolCall / ToolResult  |
| 06 | `06_tool_registry.py`           | xAI       | @tool decorator + ToolRegistry       |
| 07 | `07_thinking.py`                | Anthropic | ThinkingConfig (budget + adaptive)   |
| 08 | `08_multimodal.py`              | Gemini    | ImagePart (inline base64 image)      |
| 09 | `09_server_tools.py`            | xAI       | xai-responses provider, ServerTool   |
| 10 | `10_react_agent.py`             | OpenAI    | ReActAgent + events + tools          |
| 11 | `11_plan_execute_agent.py`      | Gemini    | PlanExecuteAgent                     |
| 12 | `12_tree_of_thoughts_agent.py`  | xAI       | TreeOfThoughts (reasoning-only)      |
| 13 | `13_self_discovery_agent.py`    | Anthropic | SelfDiscovery 4-phase reasoning      |
| 14 | `14_async_client.py`            | OpenAI    | AsyncClient + parallel requests      |
| 15 | `15_middleware_stack.py`        | OpenAI    | Cache + Cost + Guardrails + Tracing  |
| 16 | `16_fallback_client.py`         | OpenAI+Anthropic | Sync/async provider fallback  |
| 17 | `17_templates_and_parsing.py`   | OpenAI    | PromptTemplate + ChatTemplate + parsing |
| 18 | `18_tokens_and_memory.py`       | None      | Token estimation + memory windowing  |
| 19 | `19_guardrails.py`              | OpenAI    | GuardrailMiddleware blocking          |
| 20 | `20_batch_api.py`               | OpenAI    | Batch submit/status/results           |
| 21 | `21_stream_events_deep_dive.py` | OpenAI    | Detailed event handling (`text/tool_call/usage/done`) |
| 22 | `22_retry_and_timeout_controls.py` | OpenAI | RetryConfig + per-request timeouts + error handling |
| 23 | `23_async_batch_api.py`         | OpenAI    | AsyncBatchClient submit/status/results |
| 24 | `24_custom_middleware.py`       | OpenAI    | Custom middleware + short-circuit     |
| 25 | `25_custom_cache_backend.py`    | OpenAI    | Custom CacheBackend + ResponseCache   |
| 26 | `26_cost_reporting.py`          | OpenAI    | CostTracker snapshot and per-model cost |
| 27 | `27_tracing_opentelemetry.py`   | OpenAI    | TracingMiddleware with OpenTelemetry-style tracer |
| 28 | `28_lats_agent.py`              | OpenAI    | LATSAgent (tree search)               |
| 29 | `29_reflexion_agent.py`         | OpenAI    | ReflexionAgent (retry + reflection)   |
| 30 | `30_rewoo_agent.py`             | OpenAI    | ReWOOAgent (plan/worker/solver)       |
| 31 | `31_llm_compiler_agent.py`      | OpenAI    | LLMCompilerAgent (DAG planning + execution) |

## Running

```bash
uv run python examples/01_hello_world.py
```

> **Cost note:** These examples make real API calls and will incur charges
> on your provider accounts. All examples use cost-effective models to
> minimise spend.
