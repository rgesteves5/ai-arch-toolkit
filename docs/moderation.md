# Moderation

Content moderation that plugs into the LLM middleware chain to screen **input** (before the call) and/or **output** (after the response). Two moderator backends ship in the box, both built on a common `Moderator` protocol.

## Moderators

### OpenAIModerator

Uses OpenAI's free `omni-moderation-latest` endpoint. Returns a flagged verdict, the triggered category names, and per-category scores.

```python
from ai_arch_toolkit.toolkit.moderation import OpenAIModerator

mod = OpenAIModerator()                       # api_key from env; model overridable
result = await mod.moderate("some text")
result.flagged      # bool
result.categories   # list[str]
result.scores       # dict[str, float]
```

`OpenAIModerator(*, api_key=None, model="omni-moderation-latest")`. It's an async context manager (`async with OpenAIModerator() as mod: ...`) and exposes `moderate()` / `moderate_sync()`.

### LLMModerator

Uses any `LLM` as a classifier against your own category list — handy for custom policies or providers without a moderation endpoint.

```python
from ai_arch_toolkit import LLM
from ai_arch_toolkit.toolkit.moderation import LLMModerator

classifier = LLM("claude-haiku-4-5-20251001")
mod = LLMModerator(classifier, ["Violence", "Harassment", "PII"], fail_behavior="closed")
result = await mod.moderate("some text")
```

`LLMModerator(llm, categories, *, fail_behavior="closed")`. With `fail_behavior="closed"` (default) a classification failure flags the content (fail safe); `"open"` lets it through. Don't attach `ModerationMiddleware` to the classifier `LLM` itself — that would recurse.

Both return a **`ModerationResult`**: `flagged`, `categories`, `scores`, `explanation`, `raw`.

---

## ModerationMiddleware

Wires a moderator into an `LLM` so checks run automatically.

```python
from ai_arch_toolkit import LLM
from ai_arch_toolkit.toolkit.moderation import ModerationMiddleware, OpenAIModerator

mw = ModerationMiddleware(
    input=OpenAIModerator(),     # screen the user prompt before the call
    output=None,                 # optionally screen the model's reply too
    on_flagged="raise",          # "raise" -> ModerationError, "warn" -> log only
)
llm = LLM("claude-sonnet-4-20250514", middleware=[mw])

response = llm.complete_sync("User prompt here")  # raises ModerationError if flagged
```

`ModerationMiddleware(*, input=None, output=None, on_flagged="raise")` — supply at least one of `input` / `output`. When flagged, `"raise"` throws `ModerationError` (carrying `categories` and `explanation`); `"warn"` only logs.

> Output moderation runs after stream finalization, so streamed text may reach the user before the output check completes — prefer `input` screening (or non-streaming calls) when you must block before display.

---

See also: [Middleware](middleware.md) for the hook model and execution order.
