# Examples 28–36 Summary

This file tracks the current purpose of examples `28` through `36`.
It is a guide to the actual files in `examples/`, not a historical execution log.

## Overview

| # | File | Area | API key | Summary |
|---|------|------|---------|---------|
| 28 | `28_memory_graph_basics.py` | Memory | No | `GraphStore` basics: nodes, edges, search, views, and persistence |
| 29 | `29_memory_middleware.py` | Memory + LLM | Yes | `MemoryMiddleware` retrieval + recording around normal LLM calls |
| 30 | `30_memory_agent_tools.py` | Memory + Agent Flow | Yes | `memory_tools()` combined with `react_flow` |
| 31 | `31_flow_basics.py` | Flow | No | `Step`, `Result`, `Flow`, `FlowStep`, `FlowResult`, and state artifacts |
| 32 | `32_flow_streaming.py` | Flow | No | `flow.iter()` / `iter_sync()` and flow event inspection |
| 33 | `33_flow_with_llm.py` | Flow + LLM | Yes | Flow steps that call an `LLM` and accumulate usage/cost |
| 34 | `34_knowledge_registry.py` | Knowledge | No | `KnowledgeRegistry` basics and prompt-context assembly |
| 35 | `35_knowledge_loaders.py` | Knowledge | No | File and directory loaders for knowledge entries |
| 36 | `36_fallback_chains_and_attempts.py` | LLM + Agent Flow | Yes | Fallback chains, attempt tracking, streaming fallback, and flow traces |

## 28. Memory Graph Basics

`28_memory_graph_basics.py` is the lowest-level memory example. It creates a
`GraphStore`, seeds typed nodes, connects edges, performs keyword search, and
demonstrates `TemporalView`, `RelationalView`, `PropertyView`, and JSON
persistence.

## 29. Memory Middleware

`29_memory_middleware.py` shows how `MemoryMiddleware` wraps ordinary LLM calls.
Relevant memories are injected before a request, and new interaction memories
are recorded after the response.

## 30. Memory Agent Tools

`30_memory_agent_tools.py` demonstrates `memory_tools()` generating a `ToolGroup`
that a `react_flow` can use to remember, recall, explore, and forget items in
the memory graph.

## 31. Flow Basics

`31_flow_basics.py` is the introductory Flow example. It defines a few simple
steps, runs them sequentially in a `Flow`, and inspects the resulting trace,
duration, cost, and merged state artifacts.

## 32. Flow Streaming

`32_flow_streaming.py` focuses on execution events. It uses both `flow.iter()`
and `flow.iter_sync()` to stream `FlowEvent` values while a flow runs and then
compares that with a normal `run()` call.

## 33. Flow With LLM

`33_flow_with_llm.py` shows a realistic content pipeline built with `Flow`
primitives rather than a separate pipeline API. Each step calls the shared
`LLM`, returns `Result(usage=..., cost=...)`, and contributes to the flow trace.

## 34. Knowledge Registry

`34_knowledge_registry.py` demonstrates `KnowledgeRegistry`, category/tag
filtering, and prompt context construction from registered knowledge entries.

## 35. Knowledge Loaders

`35_knowledge_loaders.py` demonstrates `load_text()`, `load_json()`,
`load_toml()`, `load_markdown()`, and directory loading helpers for building a
registry from files on disk.

## 36. Fallback Chains and Attempts

`36_fallback_chains_and_attempts.py` shows LLM fallback chains in increasing
depth: string shorthand, multiple configured fallback clients, retry plus
fallback, response attempt history, stream fallback finalization, and the same
ideas inside a `react_flow`.

## Notes

- There is no standalone `Pipeline` API in the current codebase. Examples `31`,
  `32`, and `33` are Flow examples.
- There is no `ReActAgent` class in the public API. The current examples use
  `react_flow`.
- Some examples require provider SDK extras in addition to API keys. Install the
  extra for the model family you plan to use, for example
  `ai-arch-toolkit[openai]` or `ai-arch-toolkit[anthropic]`.
