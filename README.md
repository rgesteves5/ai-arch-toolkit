# ai-arch-toolkit

[![CI](https://github.com/rgesteves5/ai-arch-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rgesteves5/ai-arch-toolkit/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A lightweight, unified LLM client for Anthropic, OpenAI, Gemini, and xAI — plus
Flow orchestration, nine built-in agent architectures, a typed graph layer with
agent memory, knowledge loading, and moderation. Zero core dependencies; bring
your own provider SDK.

## Why

- **One client, every provider.** `LLM("claude-…")`, `LLM("gpt-…")`, `LLM("gemini-…")`, `LLM("grok-…")` — same call surface, automatic routing.
- **Local models.** Point at Ollama, LM Studio, or vLLM with `base_url=` — arbitrary model tags, no API key needed on localhost, real-time reasoning events.
- **Async-first, sync everywhere.** Every coroutine has a `_sync` wrapper, so you never have to choose.
- **Agent architectures as building blocks.** ReAct, Reflexion, ReWOO, Plan-Execute, Tree of Thoughts, LATS, Self-Discovery, LLM Compiler, and Generate-Review — all as `Flow` factories.
- **No mandatory deps.** Install only the provider SDKs you actually use.

## Install

The package is not on PyPI yet. Install directly from this repository:

```bash
uv add "git+https://github.com/rgesteves5/ai-arch-toolkit.git#egg=ai-arch-toolkit[openai]"
# or substitute another extra: [anthropic], [gemini], [xai], or [all]
```

Pip works the same way. Extras:

| Extra      | Pulls in                                              |
| ---------- | ----------------------------------------------------- |
| `anthropic`| `anthropic>=0.40`                                     |
| `openai`   | `openai>=1.50`                                        |
| `gemini`   | `google-genai>=1.0`                                   |
| `xai`      | `xai-sdk>=1.7.0`                                      |
| `graph`    | `networkx>=3.0` (required by graph + memory)          |
| `tokens`   | `tiktoken>=0.7` (local token counting)                |
| `yaml`     | `pyyaml>=6.0` (yaml knowledge loader)                 |
| `all`      | Every provider plus every optional feature            |

API keys are read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
(or `GEMINI_API_KEY`), or `XAI_API_KEY` — copy `.env.example` to `.env` and
`set -a && source .env && set +a` (or use a tool like `direnv`). If both Gemini
keys are set, `GOOGLE_API_KEY` wins.

## Quick start

### One completion

```python
from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-nano")
response = llm.complete_sync("What is the capital of France? Reply in one sentence.")

print(response.text)
print(f"Tokens — in: {response.usage.input_tokens}, out: {response.usage.output_tokens}")
print(f"Cost: ${response.cost:.6f}")
```

### Streaming

```python
from ai_arch_toolkit import LLM

llm = LLM("gpt-4.1-nano")
stream = llm.stream_sync("Explain photosynthesis in three sentences.")

for chunk in stream:
    print(chunk, end="", flush=True)

# After the stream is consumed, the full Response is available:
print(f"\nCost: ${stream.response.cost:.6f}")
```

### Tools with `@tool` and `ToolGroup`

Tools should expose explicit, typed operations. Avoid executing arbitrary code
from user or model input; prefer narrow functions with clear parameters.

```python
from ai_arch_toolkit import LLM, run_tools_sync
from ai_arch_toolkit.core import ToolGroup, tool


@tool
def multiply(a: float, b: float) -> str:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a * b)


tools = ToolGroup(multiply)
llm = LLM("gpt-4.1-nano")
messages = [{"role": "user", "content": "What is 42 * 17?"}]

response = llm.complete_sync(messages, tools=tools)
while response.has_tool_calls:
    messages.append(response.to_message())
    messages.extend(run_tools_sync(response, tools))
    response = llm.complete_sync(messages, tools=tools)

print(response.text)
```

### A ReAct agent in five lines

```python
from ai_arch_toolkit import LLM, State, ToolGroup
from ai_arch_toolkit.toolkit.agents import react_flow, react_initial_state
from ai_arch_toolkit.toolkit.tools import geocode, get_weather

llm = LLM("gpt-4.1-nano")
flow = react_flow(llm, ToolGroup(get_weather, geocode), max_iterations=5)
state = State(operational=react_initial_state("Weather and coordinates of Tokyo?"))
result = flow.run_sync(state)

print(state["response"].text)
print(f"Steps: {len(result.trace.steps)} · Cost: ${result.total_cost:.4f}")
```

## Provider × feature matrix

|                       | Anthropic | OpenAI | Gemini | xAI |
| --------------------- | :-------: | :----: | :----: | :-: |
| Sync + async          | ✅        | ✅     | ✅     | ✅  |
| Streaming             | ✅        | ✅     | ✅     | ✅  |
| Rich stream events    | ✅        | ✅     | ✅     | ✅  |
| Tool / function call  | ✅        | ✅     | ✅     | ✅  |
| Structured output     | ✅ native | ✅     | ✅     | ✅  |
| Multimodal (image)    | ✅        | ✅     | ✅     | ✅  |
| Documents (PDF, etc.) | ✅        | ✅     | ✅     | —   |
| Prompt caching        | ✅        | ✅     | —      | —   |
| Extended thinking     | ✅        | —      | ✅     | ✅  |
| Server-hosted tools   | ✅ code+web | ✅ code+web | —      | —   |
| Batch API             | ✅        | ✅     | —      | —   |

## Agent architectures

Each factory returns a `Flow` and has a matching `*_initial_state(task)` helper.

| Factory                 | Pattern                                                  |
| ----------------------- | -------------------------------------------------------- |
| `react_flow()`          | Cyclic LLM → tool execution loop                         |
| `reflexion_flow()`      | Inner ReAct with evaluate + reflect retry                |
| `rewoo_flow()`          | Plan with `#E{n}` placeholders → execute → solve         |
| `plan_execute_flow()`   | Numbered plan → per-step ReAct → solve                   |
| `tot_flow()`            | Tree of Thoughts with DFS/BFS search                     |
| `lats_flow()`           | Language Agent Tree Search (MCTS)                        |
| `self_discovery_flow()` | Select reasoning modules → adapt → operationalize → solve|
| `llm_compiler_flow()`   | Plan DAG → parallel execute → join                       |
| `generate_review_flow()`| Generator → reviewer retry loop, with optional tools     |

## Tour of the package

```
ai_arch_toolkit/
├── core/        Stateless async-first foundation — LLM, providers, tools,
│                graph, retry, middleware, pricing
└── toolkit/     Convenience layer — Flow orchestration, 9 agent factories,
                 pre-built tools, graph-backed memory, knowledge registry,
                 moderation
```

For a deeper read, see [`docs/framework-overview.md`](docs/framework-overview.md)
and the 36 runnable scripts under [`examples/`](examples/).

## Documentation

```bash
uv sync --extra dev --extra docs
uv run mkdocs serve            # http://localhost:8000
uv run pdoc ai_arch_toolkit -o site/api
```

Browse `docs/` for guides, or jump straight to:

- [`docs/getting-started.md`](docs/getting-started.md)
- [`docs/framework-overview.md`](docs/framework-overview.md)
- [`docs/flow-architecture.md`](docs/flow-architecture.md)
- [`docs/graph.md`](docs/graph.md)
- [`docs/model-compatibility.md`](docs/model-compatibility.md)

## Development

```bash
git clone https://github.com/rgesteves5/ai-arch-toolkit.git
cd ai-arch-toolkit
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a provider, ship a new
tool, or contribute an agent flow.

## License

MIT — see [LICENSE](LICENSE).
