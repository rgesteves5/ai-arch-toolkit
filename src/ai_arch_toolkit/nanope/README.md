# Nano Project Examples

This directory contains examples of nano projects using the ai-arch-toolkit (core/ and toolkit/).

## BBEH Mini Benchmark (`bbeh/`)

Evaluates ai-arch-toolkit reasoning flows against BBEH mini (460 questions, 23 tasks) using Inspect AI.

### Setup

```bash
uv sync --extra bench
```

### Strategies

| Strategy | Description |
|---|---|
| `baseline` | Raw LLM call, no tools or orchestration |
| `self_discovery` | Select → Adapt → Operationalize → Solve reasoning flow |
| `react_tools` | ReAct loop with `think` (scratchpad) + `math_eval` tools |

### Running

```bash
# Via our wrapper
uv run python -c "
from ai_arch_toolkit.nanope.bbeh import bbeh_task
from inspect_ai import eval
results = eval(bbeh_task(strategy='react_tools', model='gpt-5-nano'))
"

# Quick smoke test (2 samples)
uv run python -c "
from ai_arch_toolkit.nanope.bbeh import bbeh_task
from inspect_ai import eval
results = eval(bbeh_task(strategy='baseline', model='gpt-5-nano'), limit=2)
"
```

### Cost Tracking

Each solver stores `cost` in `TaskState.metadata["cost"]` from the framework's cost tracking. After eval, aggregate from the log:

```python
total_cost = sum(s.metadata.get("cost", 0) for s in log.samples)
```

## Research Center (`research_center/`)

Multi-agent research pipeline with shared wiki memory. Four agents — all built as `generate_review_flow` configurations — collaborate through a shared `GraphStore` wiki.

### Architecture

The **Owner** (the user) dictates what to research, report format, focus areas, audience, etc. The **Manager** translates the owner's brief into plans for each agent.

```
Owner (brief) → Manager → [Researcher → Linker → Manager]* → Writer
                   ↕            ↕           ↕                    ↕
                [=============== Shared Wiki Memory ===============]
```

| Agent | Model | Role | Tools |
|---|---|---|---|
| Researcher | grok-4-1-fast-reasoning | Gather knowledge from wikipedia/dictionary | wikipedia, dictionary, wiki write+read |
| Linker | grok-4-1-fast-reasoning | Discover and create connections between nodes | wiki read+write |
| Manager | grok-4-1-fast-reasoning | Translate owner's brief into plans for each agent | wiki read, web, reasoning strategies |
| Writer | gemini-3-flash | Synthesize wiki into a structured report | wiki read, reasoning strategies |

### Running

```python
from ai_arch_toolkit.nanope.research_center import (
    create_wiki_sync, save_wiki_sync, load_wiki_sync, run_pipeline_sync,
)

# --- New project ---
wiki = create_wiki_sync()
result = run_pipeline_sync(
    "The history of the Internet",
    wiki,
    owner_brief=(
        "I want a technical report aimed at CS students. "
        "Focus on protocol evolution (TCP/IP, HTTP, DNS) and key people. "
        "Keep it under 2000 words. Use a chronological structure."
    ),
    budget=0.50,
    max_cycles=2,
)
print(result.report)

# Save the wiki for later reuse
save_wiki_sync(wiki, "internet_history.json")

# --- Resume a previous project ---
wiki = load_wiki_sync("internet_history.json")
result = run_pipeline_sync(
    "The history of the Internet",
    wiki,
    owner_brief=(
        "Now add a section on internet governance (ICANN, IETF). "
        "Produce an executive summary for non-technical readers."
    ),
    budget=0.30,
    max_cycles=1,
)
```

### Budget Control

The pipeline checks remaining budget between each agent phase. Set `budget` (USD) and `max_cycles` to control cost and depth. The Manager agent can also decide to stop early if coverage is sufficient.
