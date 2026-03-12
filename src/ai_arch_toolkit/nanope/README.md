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
