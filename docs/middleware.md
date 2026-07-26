# Middleware

Middleware hooks into every LLM call — before the request and after the response. Use it for cost tracking, logging, context injection, moderation, or memory.

```python
from ai_arch_toolkit import LLM, Middleware, Request, Response

class CostTracker:
    """Track cumulative cost across all LLM calls."""

    def __init__(self):
        self.total_cost = 0.0

    def before(self, request: Request) -> Request:
        # Modify the request (add context, filter messages, etc.)
        return request

    def after(self, request: Request, response: Response) -> Response:
        self.total_cost += response.cost or 0.0
        print(f"Call cost: ${response.cost:.4f} | Total: ${self.total_cost:.4f}")
        return response

tracker = CostTracker()
llm = LLM("claude-sonnet-5", middleware=[tracker])
```

---

## Request object

```python
request.messages    # list of message dicts
request.system      # system prompt (str | None)
request.tools       # tool definitions (list | None)
request.model       # model name
request.kwargs      # extra provider kwargs
```

A middleware's `before` returns a (possibly new) `Request`; `after` returns a (possibly new) `Response`.

---

## Async middleware

If your middleware needs async work (database lookups, API calls), implement `abefore` / `aafter`:

```python
class AsyncMiddleware:
    async def abefore(self, request: Request) -> Request:
        context = await fetch_from_database(request.messages[-1])
        # ... modify request ...
        return request

    async def aafter(self, request: Request, response: Response) -> Response:
        await log_to_database(response)
        return response
```

The framework auto-detects async variants and falls back to the sync hooks if they're absent.

---

## Execution order

`before` hooks run **in order** (first middleware first). `after` hooks run **in reverse** (last middleware first). This creates an onion-like wrapping:

```
Request  → MW1.before → MW2.before → MW3.before → Provider
Response ← MW1.after  ← MW2.after  ← MW3.after  ← Provider
```

**Example** — a logger wrapping a cost guard:

```python
class Logger:
    def before(self, req: Request) -> Request:
        print(f"[log] Sending {len(req.messages)} messages")
        return req
    def after(self, req: Request, res: Response) -> Response:
        print(f"[log] Got {res.usage.output_tokens} tokens")
        return res

class CostGuard:
    def __init__(self, budget: float):
        self.spent = 0.0
        self.budget = budget
    def after(self, req: Request, res: Response) -> Response:
        self.spent += res.cost or 0.0
        if self.spent > self.budget:
            raise RuntimeError(f"Budget exceeded: ${self.spent:.2f}")
        return res

llm = LLM("claude-sonnet-5", middleware=[Logger(), CostGuard(1.00)])
# Request:  Logger.before → CostGuard.before (no-op) → Provider
# Response: CostGuard.after → Logger.after ← Provider
```

---

## Built-in middleware

Several ready-made middlewares ship in the box — drop them into `middleware=[...]`.

### RateLimitMiddleware

Proactive client-side rate limiting via a token bucket. Smooths bursts so you stay under a provider's requests-per-minute ceiling.

```python
from ai_arch_toolkit import LLM, RateLimitMiddleware

llm = LLM("claude-sonnet-5", middleware=[RateLimitMiddleware(requests_per_minute=60)])
# burst defaults to int(requests_per_minute); override with burst=...
```

> The limiter only acts on the **async** path (`abefore`), which `LLM.complete()` uses. `stream()` / `stream_events()` run middleware through the sync hook and **bypass** the limiter — use `complete()` when you need rate limiting.

### TracingMiddleware

Emits an OpenTelemetry span per LLM call, tagged with input/output tokens, cost, and duration. It's a **no-op if OpenTelemetry isn't installed**, so it's safe to leave in.

```python
from ai_arch_toolkit import LLM, TracingMiddleware

llm = LLM("claude-sonnet-5", middleware=[TracingMiddleware(tracer_name="my-app")])
```

### MemoryMiddleware & ModerationMiddleware

- **`MemoryMiddleware`** — auto-injects relevant memories into the system prompt and records each turn. See [Memory](memory.md#memorymiddleware).
- **`ModerationMiddleware`** — screens input and/or output through a content moderator. See [Moderation](moderation.md).

---

See also: [LLM Facade](llm.md) · [Pricing & Cost Tracking](pricing.md) for run-wide budgets via `BudgetPolicy`.
