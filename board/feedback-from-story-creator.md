# Feedback from downstream project: `story-creator`

This file collects learnings, rough edges, and improvement ideas accumulated while
building a 3-step Story Writing Knowledge Graph (SWKG) on top of `ai-arch-toolkit`
in the `story-creator` project (`~/Documents/dev/pessoal/story-creator/`). The SWKG
uses `react_flow` + `ToolGroup` + `LLM` heavily — a planner agent, a writer agent
with 17 tools, and cumulative state tracking across many sessions. Everything below
is grounded in a real integration, not hypothetical.

Organized from highest impact to lowest. Items marked **[BLOCKER]** required
non-obvious workarounds; **[ERGONOMICS]** are annoyances; **[DOC]** needs better
documentation; **[NICE-TO-HAVE]** are future polish.

---

## What worked well (keep/promote)

These need no change — flag them as wins so they stay stable:

1. **Pydantic model as `@tool` parameter** — `infer_schema` correctly calls
   `model_json_schema()` for Pydantic types. This was the key that unlocked our
   `submit_writing_output(output: WritingSessionOutput)` pattern. Writers pass a
   deeply nested Pydantic structure as a single tool argument and we get a
   provider-valid JSON schema for free. **Document this prominently.**

2. **`react_flow` control flow** — `needs_llm_call` / `has_tool_calls` predicates
   with `FlowStep`-based composition are clean and easy to reason about. The
   `final_answer_hint` parameter (inject last-turn "no tools, answer in text"
   message) was exactly the right affordance for our terminal submit pattern.

3. **`Usage` accumulation in `state`** — `state.get("total_usage")` returning a
   cumulative `Usage` across ReAct turns is exactly what we needed for token-level
   cost accounting. No manual summing required.

4. **Provider detection from model name prefix** — `claude-*` / `gpt-*` / `grok-*`
   / `gemini-*` → correct provider. Removed a whole config concern from our side.

5. **`strict="ignore"` / optional zip strictness** — We didn't hit bugs around
   tool_calls + results length mismatch. Whatever ReAct does there, it's
   resilient in practice.

---

## Missing features (unblock or would simplify)

### 1. `react_flow(output_schema=...)` for structured terminal output **[BLOCKER]**

We wanted a structured final answer from the writer's ReAct loop. `react_flow`
has no `output_schema` parameter (we checked `toolkit/agents/flows/_react.py`
line 18). Workaround: built a terminal `submit_writing_output(output:
WritingSessionOutput)` tool whose closure captures the model's submission, then
read the capture after `flow.run(state)` completes.

This works but:
- Requires a dummy "submit" tool in every ReAct-based agent that wants
  structured output
- The writer can "forget" to call submit — we have a 1-LLM-call "burned iteration"
  on the final free-text answer that gets discarded
- Final-answer-hint helps but doesn't guarantee behavior
- Each consumer project re-implements the collector pattern

**Proposed API:**

```python
@dataclass
class ReactResult[T]:
    parsed: T | None          # the structured output (None on failure)
    messages: list[dict]      # full conversation history
    usage: Usage              # cumulative usage
    turns: int

result: ReactResult[WritingSessionOutput] = await react_flow(
    llm, tools,
    system=...,
    output_schema=WritingSessionOutput,  # NEW
    max_iterations=10,
).run(task)
```

Internally: inject a synthetic `submit(output: T)` tool, capture its arg,
expose on the result. No external-tool ceremony.

### 2. `ToolGroup` has no public read API **[BLOCKER for tests]**

From `core/_tools/_group.py`:
- `__contains__(name)` ✓
- `__len__()` ✓
- `execute(tool_call) / async_execute(tool_call)` ✓
- iteration ✗
- `get(name) -> Callable` ✗
- `names() -> list[str]` ✗
- `definitions` (property) ✓

Our tests needed to invoke raw tools (without building a synthetic `ToolCall`
just to exercise a tool's body). We had to reach into `ToolGroup._fns` with a
comment explaining why. Every downstream project writing tests for tools will
hit this.

**Proposed API:**

```python
class ToolGroup:
    def get(self, name: str) -> Callable[..., Any]: ...
    def names(self) -> list[str]: ...
    def __iter__(self) -> Iterator[Callable[..., Any]]: ...  # iterates functions
    def __getitem__(self, name: str) -> Callable[..., Any]: ...
```

### 3. Auto-validate Pydantic tool args **[ERGONOMICS, high value]**

`core/_tools/_executor.py` currently does `result = fn(**tool_call.input)`.
If `fn` is annotated `def submit(output: WritingSessionOutput)`, the caller
gets `output=<dict>`, not `output=WritingSessionOutput(...)`. Every tool body
has to do:

```python
try:
    validated = WritingSessionOutput.model_validate(output)
except ValidationError as exc:
    return f"Error: {exc}"
```

**Proposed behavior:** executor detects Pydantic type hints, auto-validates,
converts `ValidationError` into a tool-result string prefixed with `"Error:"`.
Successful validation means the tool body receives a typed instance.

Could be opt-in via `@tool(auto_validate=True)`.

### 4. Per-tool parallelism policy **[ERGONOMICS]**

`react_flow(parallel_tool_calls=True)` is a single global toggle. In our writer
loop:
- Read tools are idempotent → fine in parallel
- Mutation tools (`upsert_character`, `plant_chekovs_gun`) do last-write-wins
  on the same entity name → NOT safe in parallel

We set `parallel_tool_calls=False` globally, losing parallelism on reads. A
tool-level flag:

```python
@tool(parallel_safe=False)
def upsert_character(...): ...
```

Then `react_flow` groups calls per turn: parallel-safe tools run concurrently,
unsafe ones serialize.

### 5. Cost estimation API **[MISSING utility]**

`Response.cost` exists and we can aggregate, but there's no "estimate before
running" helper. Every downstream project rolls its own per-model pricing
tables. story-creator has `shared/cost_tracking.py` with hard-coded prices for
Claude / GPT / Grok / Gemini models.

**Proposed:**

```python
from ai_arch_toolkit import estimate_cost, get_pricing

pricing = get_pricing("claude-sonnet-4-5")
# → Pricing(input_per_million=3.0, output_per_million=15.0, ...)

cost: Decimal = estimate_cost(
    model="claude-sonnet-4-5",
    input_tokens=40000,
    output_tokens=12000,
)  # → 0.30 USD
```

Backed by a JSON file in the package so it ships with the toolkit. Downstream
projects add their own margin/conversion (we do credits = EUR * 10 * 5x).

### 6. Final-response extraction helper **[ERGONOMICS]**

After `flow.run(state)`, getting the final LLM message requires:

```python
response = react_state.get("response")
text = response.text if response else ""
```

**Proposed:** `state.final_response_text() -> str | None` or a dedicated
method on `Flow`.

---

## Ergonomic gaps (noticeable but non-blocking)

### 7. Named constants for state artifact keys **[ERGONOMICS]**

Throughout `_react.py`, artifacts use string keys: `"messages"`, `"response"`,
`"has_tool_calls"`, `"needs_llm_call"`, `"total_usage"`, `"turn"`. Downstream
code needs to know these to introspect. A module-level constants block or an
enum:

```python
class ReactStateKey(StrEnum):
    MESSAGES = "messages"
    RESPONSE = "response"
    TOTAL_USAGE = "total_usage"
    TURN = "turn"
```

Prevents typo-shaped bugs.

### 8. Tool-result error protocol **[ERGONOMICS]**

Currently tool errors become strings prefixed `"Error: ..."` (in `react_flow`'s
`_safe_execute`). No structured way for a tool to signal:
- "permanent failure, abort the loop"
- "retry same tool with different args"
- "this result is non-fatal but should count as a negative outcome"

Every consumer invents a string convention. A small `ToolResult` class or
typed exceptions (`FatalToolError`, `RetriableToolError`) would help.

### 9. Iteration budget = LLM calls OR tool calls? **[DOC]**

`max_iterations=10` — is this 10 LLM calls or 10 tool calls? Looking at
`_react.py`: the `llm_call` step increments `turn`, and `is_final = turn >=
max_iterations`. So it's the LLM-call count.

A writer that calls 5 tools in one turn (parallel_tool_calls=True) uses 1
iteration. A writer that calls 1 tool per turn for 5 turns uses 5. Same tool
work; very different costs against the cap. Worth documenting; maybe provide
separate `max_llm_calls` and `max_tool_calls` limits.

### 10. `final_answer_hint` vs `strip_tools_on_final` **[DOC]**

We picked `final_answer_hint=True, strip_tools_on_final=False` by guess. The
difference isn't obvious from the docstrings. A decision tree / when-to-use
would help:

> Use `final_answer_hint` when the model should normally comply with a
> text-only instruction.
>
> Use `strip_tools_on_final` when the model is sticky about calling tools
> even after being told not to (some providers / prompt styles).

### 11. `State` vs `StateSnapshot` **[DOC]**

Both are imported / referenced. We only ever used `State`. Unclear when a
consumer would construct a `StateSnapshot`. Quick note in the module docstring
would save reading `_state.py`.

### 12. `react_initial_state(task)` accepts `Content` — what is `Content`? **[DOC]**

Signature: `task: Content`. `Content` is a type alias in `_content.py` (union
of string, list of message dicts, etc.). A one-line "you can pass a plain
string here" note in the docstring beats inferring from tests.

### 13. Structured output failure mode **[DOC + maybe API]**

`llm.complete(..., output_schema=X)` returns `Response` with `parsed: X | None`.
When the provider returns malformed JSON, `parsed is None` and `text` has the
raw reply. We added runtime guards everywhere. Options:

- Document the None case prominently
- Add a `strict_schema=True` flag that raises instead of returning None
- Add `response.require_parsed() -> X` (raises if None, with diagnostic)

---

## Nice-to-haves

### 14. Structured turn log

`state["messages"]` is `list[dict]` — OpenAI-wire-format. Post-mortem / replay
/ debugging would benefit from a typed turn log:

```python
@dataclass
class Turn:
    kind: Literal["llm_call", "tool_result"]
    timestamp: datetime
    usage: Usage | None
    content: ...  # typed
```

Accessible as `state.turn_log: list[Turn]`.

### 15. Recording / replay for tests

We can't test the writer's full ReAct loop without a live LLM. A recording
feature (capture LLM responses during a live run → replay in tests) would let
downstream projects regression-test integrations without API calls.

This is a substantial feature (conversation-tape format, matching heuristics,
stale-recording detection). Low priority; high value if ever built.

### 16. Per-session conversation store primitive

story-creator built its own SWKG (character state, chekov's guns, etc.) on top
of `StoryStateStore`. A generic "append-only session store" + "cumulative
snapshot derivable from session stream" primitive might factor out. Unclear
it's reusable enough to justify being in core.

### 17. Streaming ReAct tool-call events

ReAct loops currently run to completion before returning. For UX, streaming
the LLM's in-progress reasoning + tool calls would let UIs show "the writer is
querying Alice's state" in real time. Not needed today; would be nice later.

---

## Documentation additions requested

Shortlist of docs that would have saved us significant time:

1. **Tutorial: "Build a ReAct agent with structured final output"** — walks
   through the submit-tool pattern we reinvented. Show `WriterOutputCollector`
   and pydantic validation in the tool body.
2. **Tutorial: "Test your ReAct agent"** — the `_fns` private-access issue.
   Recommend patterns.
3. **Cheatsheet: `state` artifact keys** — `"messages"`, `"total_usage"`,
   `"turn"`, `"response"`, `"has_tool_calls"`, `"needs_llm_call"`.
4. **Patterns page: "When to use `llm.complete(output_schema=...)` vs
   `react_flow` vs `ReActAgent`"** — we weren't sure; picked by trial.
5. **Provider quirks page** — which providers require `strip_tools_on_final`,
   which obey `final_answer_hint` reliably, known structured-output
   limitations per provider.

---

## Prioritized list (if someone's planning work)

| Priority | Item | Reason |
|---|---|---|
| P0 | Auto-validate Pydantic tool args (#3) | Biggest ergonomic win, minimal API change |
| P0 | `ToolGroup` public read API (#2) | Every test file currently reaches into `_fns` |
| P1 | `react_flow(output_schema=...)` (#1) | Eliminates the submit-tool boilerplate |
| P1 | Cost estimation API (#5) | Every downstream project rolls this |
| P2 | Per-tool parallelism policy (#4) | Blocks safe parallel tool calls with mutation |
| P2 | Strict schema + require_parsed (#13) | Defensive code everywhere would go away |
| P3 | Named state artifact keys (#7) | Typo prevention |
| P3 | Final-response text helper (#6) | Syntactic |
| P4 | Tutorials (docs shortlist) | Reduces onboarding cost for next consumer |

---

## Context / version pins

Written against ai-arch-toolkit as of 2026-04-23 (story-creator's `uv.lock`
pulls `main` branch). Files referenced:

- `toolkit/agents/flows/_react.py`
- `core/_tools/_executor.py`
- `core/_tools/_decorator.py`
- `core/_tools/_schema.py`
- `core/_tools/_group.py`

If any of the above have changed since writing, re-verify before acting on
specific code pointers.
