# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

ai-arch-toolkit is a documentation-only research repository containing practical reference guides for building AI-powered applications and Python development. There is no runnable code, build system, or test suite — the repository consists entirely of Markdown documents with embedded code examples.

## Repository Structure

All content lives in `research/`. The guides fall into two topic areas:

**AI / LLM:**
- `llm_agent_architectures.md` — Patterns for building LLM agents (ReAct, ReWOO, LLMCompiler, Reflexion, LATS, Tree of Thoughts, Plan-then-Execute, Self-Discovery) with pseudocode, tradeoff analysis, and a decision tree
- `llm_api_complete_guide_2026.md` — Comprehensive LLM API reference covering OpenAI, Anthropic, xAI, Google Gemini, Mistral, and Groq with canonical request/response schemas, runnable Python examples (using raw `requests`), pricing tables, and model lifecycle info

**Python / Graphs:**
- `python_best_practices.md` — Python conventions and patterns guide targeting Python 3.12+ (project structure, naming, toolchain, type hints, design patterns, testing, async, CI/CD)
- `modern_python_2015_16.md` — What changed in the Python ecosystem in 2025-2026 (uv, ruff, ty, Python 3.12-3.14 features, free-threading, deferred annotations)
- `graph_algorithms_overview.md` — Graph algorithm taxonomy (traversal, shortest path, MST, flow, matching, etc.)
- `networkx_guide.md` — Practical NetworkX 3.6+ reference (graph types, algorithms, visualization, pandas/numpy conversion, performance)

## Conventions

- All code examples use raw `requests` for LLM API calls (not provider SDKs) so they are self-contained and portable
- Python helper functions (`post_json`, `extract_text`, `extract_usage`) are defined in the API guide and referenced throughout — keep them consistent when editing
- Provider-specific differences (auth headers, schema shapes, role names) are called out explicitly; avoid generalizing across providers without noting divergences
- The API guide includes a changelog section at the bottom — append entries when making updates
- Python guides target 3.12+ and recommend the modern toolchain: uv + ruff + pytest + pyright

## Key Considerations When Editing

- Model IDs, pricing, and deprecation dates change frequently — verify against official docs before updating
- Anthropic uses `input_schema` for tools (not `parameters` like OpenAI)
- Anthropic's `system` prompt is a top-level field, not a message role
- Gemini uses `contents`/`parts` structure (not `messages`/`content`)
- Claude 4.5 models use Extended Thinking (`thinking.type = "enabled"` + `budget_tokens`); Claude 4.6 uses Adaptive Thinking (`thinking.type = "adaptive"` + `effort` levels)
- OpenAI has two API surfaces: Chat Completions (`/v1/chat/completions`) and Responses API (`/v1/responses`) — keep both documented
