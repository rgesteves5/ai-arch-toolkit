# Pipeline & Knowledge Examples — Execution Report

**Date:** 2026-03-03
**Branch:** `refactor/clean_project`
**Runtime:** Python 3.13, macOS Darwin 25.3.0

---

## Overview

| # | Example | Type | API Key | Status | Duration |
|---|---------|------|---------|--------|----------|
| 31 | Pipeline Basics | Pipeline | No | OK | <1ms |
| 32 | Pipeline Streaming & Resume | Pipeline | No | OK | <1ms |
| 33 | Pipeline with LLM | Pipeline + LLM | Yes (OpenAI) | OK | 7.47s |
| 34 | Knowledge Registry | Knowledge | No | OK | <1ms |
| 35 | Knowledge Loaders | Knowledge | No | OK | <1ms |

---

## Example 31: Pipeline Basics

**File:** `examples/31_pipeline_basics.py`
**Features demonstrated:** Phase functions, `Pipeline.run()`, `PipelineContext`, provenance tracking, metadata, warnings aggregation

### Pipeline Configuration

| Property | Value |
|----------|-------|
| Pipeline name | `essay` |
| Phases | `gather_requirements` → `create_outline` → `draft_content` |
| Phase count | 3 |
| Agent type | None (pure pipeline) |
| Tools used | None |
| LLM calls | 0 |
| Initial context | `{"topic": "async programming"}` |
| Metadata | `{"run_id": "demo-001"}` |

### Phase Results

| Phase | Status | Duration | Artifacts Produced | Warnings |
|-------|--------|----------|--------------------|----------|
| `gather_requirements` | ok | <0.0001s | `requirements`, `audience` | — |
| `create_outline` | ok | <0.0001s | `outline` | `["outline is simplified"]` |
| `draft_content` | ok | <0.0001s | `draft` | — |

### Provenance Map

| Artifact | Produced By |
|----------|-------------|
| `requirements` | `gather_requirements` |
| `audience` | `gather_requirements` |
| `outline` | `create_outline` |
| `draft` | `draft_content` |

### Aggregated Result

| Metric | Value |
|--------|-------|
| Overall status | `ok` |
| Total duration | <0.001s |
| Token usage | None (no LLM calls) |
| Total warnings | `["outline is simplified"]` |

### Output

```
Final draft: Draft for general: Introduction -> Main argument -> Conclusion
```

---

## Example 32: Pipeline Streaming & Resume

**File:** `examples/32_pipeline_streaming_and_resume.py`
**Features demonstrated:** `Pipeline.iter()` streaming, early break, `stop_on_failure`, `stop_on_partial`, `Pipeline.run_from()` resume, `PhaseResult.partial()`, token accumulation

### Scenario A: Streaming with `iter()`

| Phase | Status | Token Usage |
|-------|--------|-------------|
| `phase_fetch` | ok | — |
| `phase_parse` | ok | input: 50, output: 30 |
| `phase_validate` | ok | — |
| `phase_store` | ok | input: 10, output: 5 |

**Accumulated tokens:** `{"input": 60, "output": 35}`

### Scenario B: Early Break

Broke after `phase_fetch`. Only `raw_data` artifact present in context; `parsed` was not produced.

### Scenario C: `stop_on_failure=True`

Pipeline: `phase_fetch` → `phase_boom` → `phase_store`

| Phase | Status | Note |
|-------|--------|------|
| `phase_fetch` | ok | Ran normally |
| `phase_boom` | failed | `ConnectionError: network error` |
| `phase_store` | skipped | `reason="pipeline stopped"` |

**Overall status:** `failed`
**Failed phases:** `["phase_boom"]`
**Skipped phases:** `["phase_store"]`

### Scenario D: `run_from()` Resume

Pipeline: `phase_fetch` → `phase_parse` → `phase_validate` → `phase_store`
Resumed from: `phase_validate` (pre-populated `raw_data` and `parsed`)

| Phase | Status | Note |
|-------|--------|------|
| `phase_fetch` | skipped | `reason="resumed past"` |
| `phase_parse` | skipped | `reason="resumed past"` |
| `phase_validate` | ok | Ran from checkpoint |
| `phase_store` | ok | Ran normally |

**Overall status:** `ok`
**Phase ordering preserved:** `[phase_fetch, phase_parse, phase_validate, phase_store]`

### Scenario E: `stop_on_partial=True`

| Phase | Status | Artifacts Merged |
|-------|--------|-----------------|
| `phase_partial_work` | partial | `items=["a", "b", "c"]` (merged despite partial) |
| `phase_store` | skipped | Pipeline stopped on partial |

**Overall status:** `partial`

---

## Example 33: Pipeline with LLM

**File:** `examples/33_pipeline_with_llm.py`
**Features demonstrated:** LLM integration in pipeline phases, per-phase token tracking, token accumulation, real API calls

### Configuration

| Property | Value |
|----------|-------|
| Pipeline name | `content-pipeline` |
| Model | `gpt-4.1-nano` (OpenAI) |
| Provider | `OpenAIProvider` |
| Agent type | None (pipeline with direct LLM calls) |
| Tools used | None |
| Knowledge used | None |
| Topic | "why async programming matters" |

### Phase Results

| Phase | Status | Input Tokens | Output Tokens | Duration |
|-------|--------|-------------|---------------|----------|
| `research` | ok | 26 | 55 | ~2.5s |
| `draft` | ok | 70 | 76 | ~2.5s |
| `review` | ok | 97 | 102 | ~2.5s |

### Token Tracking

| Metric | Value |
|--------|-------|
| Total input tokens | 193 |
| Total output tokens | 233 |
| Total tokens | 426 |
| Total duration | 7.47s |

### Cost Estimate

Using OpenAI gpt-4.1-nano pricing ($0.10/1M input, $0.40/1M output):

| Phase | Input Cost | Output Cost | Total |
|-------|-----------|-------------|-------|
| `research` | $0.0000026 | $0.0000220 | $0.0000246 |
| `draft` | $0.0000070 | $0.0000304 | $0.0000374 |
| `review` | $0.0000097 | $0.0000408 | $0.0000505 |
| **Total** | **$0.0000193** | **$0.0000932** | **$0.0001125** |

**Total cost: ~$0.00011 (< 1/100th of a cent)**

### LLM Requests/Responses

**Phase 1: `research`**
- **Prompt:** `"List 3 key points about: why async programming matters. Be concise, one sentence each."`
- **Response:**
  1. Async programming improves application responsiveness by allowing tasks to run concurrently without blocking the main thread.
  2. It enhances scalability by efficiently managing multiple I/O-bound operations, reducing resource consumption.
  3. Async code enables better user experiences through faster load times and smoother interactions.

**Phase 2: `draft`**
- **Prompt:** `"Write a concise paragraph incorporating these points:\n{key_points}"`
- **Response:** Async programming enhances application responsiveness by enabling tasks to run concurrently without blocking the main thread, resulting in faster load times and smoother user interactions. It also improves scalability by efficiently managing multiple I/O-bound operations, which reduces resource consumption and allows applications to handle more simultaneous tasks effectively. Overall, adopting asynchronous techniques leads to more efficient, responsive, and scalable applications that deliver a better user experience.

**Phase 3: `review`**
- **Prompt:** `"Rate this text 1-10 for clarity and suggest one improvement:\n\n{draft_text}"`
- **Response:** Rating 8/10 for clarity. Suggestion: include a brief example or analogy to make the concept more relatable for less technical readers.

### Provenance Map

| Artifact | Produced By |
|----------|-------------|
| `key_points` | `research` |
| `draft_text` | `draft` |
| `review` | `review` |

---

## Example 34: Knowledge Registry

**File:** `examples/34_knowledge_registry.py`
**Features demonstrated:** `KnowledgeRegistry.register()`, `by_category()`, `by_tags()`, `as_context()` with separator/transform, prompt template building

### Registry Contents

| Key | Format | Category | Tags |
|-----|--------|----------|------|
| `tone_guide` | text | constraints | writing, style |
| `audience` | text | constraints | writing, audience |
| `api_schema` | json | schemas | api, reference |
| `error_codes` | text | reference | api, errors |
| `project_context` | text | context | project |

**Total entries:** 5
**Categories:** constraints, context, reference, schemas

### Filtering Results

| Filter | Method | Matches |
|--------|--------|---------|
| Category: "constraints" | `by_category("constraints")` | 2 (tone_guide, audience) |
| Tags: "api" (match_any) | `by_tags("api", match_all=False)` | 2 (api_schema, error_codes) |
| Tags: "writing"+"style" (match_all) | `by_tags("writing", "style")` | 1 (tone_guide) |

### Context Building

**Basic `as_context()`:** Joins content with `\n\n---\n\n` separator (default).

**Custom transform (XML tags):**
```xml
<project_context>
We are building a REST API for a task management application.
</project_context>

---

<tone_guide>
Write in a friendly, professional tone. Avoid jargon. Use short sentences.
</tone_guide>
```

**Prompt template output:**
```
You are a technical writer.

[PROJECT CONTEXT]
We are building a REST API for a task management application.

[TONE GUIDE]
Write in a friendly, professional tone. Avoid jargon. Use short sentences.

[AUDIENCE]
Target audience: software developers with 2-5 years of experience.

Use the above guidelines for all responses.
```

---

## Example 35: Knowledge Loaders

**File:** `examples/35_knowledge_loaders.py`
**Features demonstrated:** `load_text()`, `load_json()`, `load_toml()`, `load_markdown()`, `load_directory()` flat + recursive, prefix keys, category/tags pass-through

### Individual Loaders

| Loader | Key | Format | Source File | Notes |
|--------|-----|--------|-------------|-------|
| `load_text` | `style` | text | style_guide.txt | Raw content preserved |
| `load_json` | `api_config` | json | api_config.json | Validated + pretty-printed (2-space indent) |
| `load_toml` | `model_settings` | toml | settings.toml | Validated via `tomllib`, raw TOML stored |
| `load_markdown` | `overview` | markdown | readme.md | Raw content preserved |

### Directory Loading (flat)

`load_directory(registry, tmp, prefix="project.", category="project-files", tags=("auto-loaded",))`

| Key | Format | Source |
|-----|--------|--------|
| `project.api_config` | json | api_config.json |
| `project.readme` | markdown | readme.md |
| `project.settings` | toml | settings.toml |
| `project.style_guide` | text | style_guide.txt |

**Files loaded:** 4
**Key derivation:** `prefix + stem` → `project.api_config`
**Ordering:** Sorted by filename (deterministic)

### Directory Loading (recursive)

`load_directory(registry, tmp, recursive=True, prefix="kb.")`

| Key | Format | Source Path |
|-----|--------|-------------|
| `kb.api_config` | json | api_config.json |
| `kb.readme` | markdown | readme.md |
| `kb.prompts.review` | text | prompts/review.txt |
| `kb.prompts.system` | text | prompts/system.txt |
| `kb.settings` | toml | settings.toml |
| `kb.style_guide` | text | style_guide.txt |
| `kb.schemas.user` | json | schemas/user.json |

**Files loaded:** 7
**Key derivation:** `prefix + relative.path.with.dots` → `kb.prompts.review`

---

## Cross-Example Summary

### Features Covered

| Feature | Ex. 31 | Ex. 32 | Ex. 33 | Ex. 34 | Ex. 35 |
|---------|--------|--------|--------|--------|--------|
| `Pipeline()` constructor | x | x | x | | |
| `Pipeline.run()` | x | x | x | | |
| `Pipeline.iter()` | | x | x | | |
| `Pipeline.run_from()` | | x | | | |
| `PipelineContext` data/metadata | x | x | x | | |
| `PipelineContext.require()` | x | | x | | |
| `PipelineContext.provenance` | x | | x | | |
| `PhaseResult.ok()` | x | x | x | | |
| `PhaseResult.failed()` | | x | | | |
| `PhaseResult.partial()` | | x | | | |
| `PhaseResult.skipped()` | | x | | | |
| `stop_on_failure` | | x | | | |
| `stop_on_partial` | | x | | | |
| Early break from `iter()` | | x | | | |
| Token tracking / accumulation | | x | x | | |
| Warnings aggregation | x | x | | | |
| LLM integration | | | x | | |
| `KnowledgeRegistry.register()` | | | | x | x |
| `KnowledgeRegistry.require()` | | | | x | x |
| `by_category()` | | | | x | |
| `by_tags()` (match_all/any) | | | | x | |
| `as_context()` | | | | x | x |
| `as_context()` + transform | | | | x | x |
| `load_text()` | | | | | x |
| `load_json()` | | | | | x |
| `load_toml()` | | | | | x |
| `load_markdown()` | | | | | x |
| `load_directory()` flat | | | | | x |
| `load_directory()` recursive | | | | | x |

### API Usage Summary

| Metric | Value |
|--------|-------|
| Total LLM calls | 3 (example 33 only) |
| Provider | OpenAI |
| Model | gpt-4.1-nano |
| Total input tokens | 193 |
| Total output tokens | 233 |
| Total tokens | 426 |
| Estimated cost | ~$0.00011 |
| Agent architectures used | None (raw pipeline) |
| Tools called | None |
| Knowledge entries created | 5 (ex. 34) + 4+4+7 (ex. 35) = 20 |
| Files loaded | 11 (across ex. 35 scenarios) |
