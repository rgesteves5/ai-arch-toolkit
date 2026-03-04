# Context — ai-arch-toolkit

## Current Architecture

Two layers under `src/ai_arch_toolkit/`:

```
ai_arch_toolkit/
├── core/              # Stateless async-first foundation
│   ├── _providers/    # SDK adapters: Anthropic, OpenAI, Gemini, xAI
│   ├── _tools/        # @tool decorator, ToolGroup, execute helpers
│   ├── _content.py    # Message builders, multimodal types
│   ├── _llm.py        # LLM class — complete/stream (async) + sync wrappers
│   ├── _response.py   # Response, ToolCall, Usage, OutputSchema, ThinkingBlock
│   ├── _pricing.py    # PricingRegistry + default_pricing.toml
│   ├── _middleware.py  # Middleware Protocol with before/after hooks
│   ├── _retry.py      # RetryConfig + exponential backoff
│   └── _server_tools.py  # ServerTool, code_execution, web_search
├── toolkit/           # Convenience utilities built on core/
│   ├── agents/        # Agent architectures (ReAct, Reflexion, ReWOO)
│   ├── tools/         # 25 pre-built tools (weather, geo, news, etc.)
│   └── _runner.py     # run_tools / run_tools_sync
└── __init__.py        # Re-exports from core/ + toolkit/
```

## Architecture Rules

- **core/** never imports toolkit/
- **toolkit/** imports core/ only
- All dataclasses: `frozen=True, slots=True`; add `kw_only=True` for 3+ fields
- Python 3.13+, `from __future__ import annotations` in every file
- PEP 695 `type` aliases
- `__all__` in every `__init__.py`
- Internal modules prefixed with `_`
- Ruff line length: 99
- SDKs are optional dependencies — `pip install ai-arch-toolkit[anthropic]`
- Toolkit tools return error strings (never raise)

## Agent Architecture

All agents under `toolkit/agents/`, built on core/ primitives:

- **BaseAgent** (`_base.py`): ABC with `_run_loop()` async generator pattern.
  `AgentConfig` for shared settings. `AgentEvent` / `AgentStep` / `AgentResult`.
- **ReActAgent** (`_react.py`): Iterative tool loop.
- **ReflexionAgent** (`_reflexion.py`): ReAct + self-critique retry loop.
  `ReflexionConfig` for evaluator, threshold, max_retries.
- **ReWOOAgent** (`_rewoo.py`): Plan → Execute → Solve.
  `ReWOOConfig` for planner/solver system prompts.

## Key Design Decisions

1. **Thinking**: `thinking: bool`, `thinking_effort: str | None`, `thinking_budget: int | None`
2. **Structured output**: `output_schema` param accepts `OutputSchema` or Pydantic model
3. **Agents yield events**: `_run_loop()` is a pure async generator. Callbacks fire in `_consume()`.
4. **Agent configs are standalone**: `ReflexionConfig`/`ReWOOConfig` don't inherit from
   `AgentConfig` (avoids frozen+slots inheritance issues). Passed separately to constructor.
