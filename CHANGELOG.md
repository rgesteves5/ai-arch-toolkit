# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- nanope research_center, agent_swarm scaffold, and advanced configurable agent (work in progress, not part of the public toolkit API).
- AGENTS.md project guidance file for Codex.
- `app` optional-dependency extra (`reflex>=0.7` + graph + yaml).

### Changed
- Pricing registry refreshed (2026-05-19): Claude 4.7 added; Claude 4.6/4.7 now ship with 1M context at standard rates (long-context tier removed).
- Examples 31–33 updated to the variadic `Flow(*steps)` API and async `flow.run(state)`.
- Docs and example index aligned with the Flow-based architecture; legacy "pipelines" and "8 agent architectures" wording removed.
- Sync timeouts in `core/_sync.py` no longer expose the dead `SYNC_TIMEOUT` / `STREAM_JOIN_TIMEOUT` aliases; use `configure_sync_timeouts()` instead.
- `RateLimitMiddleware` docstring documents the streaming-bypass limitation explicitly (previously a TODO).

### Fixed
- Structured output: parsed JSON is now validated against the Pydantic model before being returned.

## Historical log

Chronological worklog of features and major changes prior to adopting Keep a Changelog.

| Date | Change |
|------------|--------|
| 2026-03-12 | Add nanope BBEH benchmark suite, expand toolkit tools (python eval, dictionary), and add generate-review flow |
| 2026-03-08 | Add content moderation system with Moderator protocol, OpenAI and LLM implementations |
| 2026-03-08 | Replace legacy agents and pipeline with Flow-based architecture |
| 2026-03-05 | Update pricing registry with latest model prices from all providers |
| 2026-03-05 | Add local token counting and extend pricing registry |
| 2026-03-05 | Extract general-purpose graph layer from memory system with new methods |
| 2026-03-04 | Add production readiness: timeouts, validation, rate limiting, observability |
| 2026-03-03 | Add pipeline system, knowledge registry, and LLM fallback chains with attempt tracking |
| 2026-03-02 | Add graph-backed memory system with views, middleware, and agent tools |
| 2026-02-28 | Add SelfDiscovery and LLMCompiler agents, per-phase customization (PhaseConfig) |
| 2026-02-28 | Add PlanExecute, ToT, and LATS agents, rich streaming events, stream fallback |
| 2026-02-28 | Add Reflexion and ReWOO agents, delete legacy layer |
| 2026-02-27 | Add agent architecture implementation with ReAct and multimodal capabilities |
| 2026-02-25 | Rewrite core/ layer, add Gemini and xAI providers, restructure project |
| 2026-02-24 | New standard paradigm, research knowledge, board system, file structure reorganization |
| 2026-02-23 | Add tools layer — schema inference, @tool decorator, execution, and ToolGroup |
| 2026-02-23 | Add client reuse, stream metadata, and OpenAI provider |
| 2026-02-22 | Rewrite LLM layer from first principles |
| 2026-02-21 | Add MkDocs documentation, build commands, API docs, and CI configuration |
| 2026-02-09 | Migrate from Poetry to uv, initial project structure with examples |
| 2026-02-08 | Add LLMs documentation: agents architecture and API guide |
| 2026-02-07 | Initial commit |
