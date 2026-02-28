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

## In Progress

_(none)_

## Next

- More agent architectures (PlanExecute, ToT, LATS)
- Anthropic native structured output (replace tool trick)
- Rich streaming events (StreamEvent type, real-time thinking)
- Batch + Fallback on core/ layer
- Documentation site (mkdocs)

## Future

- Memory / conversation management layer
- Token estimation utilities
- Template / prompt management
