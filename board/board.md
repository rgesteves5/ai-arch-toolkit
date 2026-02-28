# Board — ai-arch-toolkit

## Completed

**SDK Provider Rewrite** (Phases 1–6) — see `tasks/phase-{1..6}-*.md` for details.
Rewrote all four providers (Anthropic, OpenAI, Gemini, xAI) as thin SDK adapters.
Thinking, structured output, streaming, error mapping all working. 819 tests.

**Project Restructuring** — deleted `_legacy/`, moved agents into `toolkit/agents/`,
rewrote top-level `__init__.py`. Two-layer architecture: `core/` + `toolkit/`.

**ReActAgent** — Thought → Action → Observation loop on core/ primitives.
Supports output_schema, llm_kwargs, parallel tool calls, budget/timeout.

**ReflexionAgent** — ReAct + self-critique retry loop. Evaluator callback,
configurable threshold and max retries.

**ReWOOAgent** — Plan → Execute → Solve. Placeholder substitution, tool execution,
solver with output_schema forwarding.

**PlanExecuteAgent** — Plan numbered steps → Execute via inner ReAct → Solve.
Configurable replanning on failure.

**ToTAgent** — Tree of Thoughts with DFS/BFS search. Generate-evaluate-expand
loop with configurable candidates, depth, and evaluation.

**LATSAgent** — Language Agent Tree Search (MCTS). UCT selection, ReAct rollouts,
LLM/external evaluation, backpropagation, reflection on failures.

**Rich Streaming Events** — `StreamEvent`, `RichStreamResponse`,
`SyncRichStreamResponse` at core/ layer. Text, thinking, tool_call event kinds.

**Stream Fallback + Middleware** — `stream()` and `stream_events()` now support
provider fallback on `APIError` and middleware before/after hooks.

**Anthropic Native Structured Output** — Verified: uses `output_config` with
`json_schema` natively (not tool trick). Verification tests added.

**Documentation Site** — mkdocs with getting-started guide, updated landing page,
nav with Getting Started, UV Guide, API Docs.

## In Progress

_(none)_

## Next

## Future

- Memory / conversation management layer
- Token estimation utilities
- Template / prompt management
