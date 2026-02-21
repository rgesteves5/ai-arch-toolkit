# Production Readiness TODO

Completed items are removed. This file tracks only open issues.

## Open Issues (Current)

- [ ] Regenerate `scripts/output/*.output.txt` using the new full-text tracing format so each call includes complete generated text between `*_BEGIN`/`*_END` markers.
- [ ] Add a `scripts/run_examples.py` option to run a single example file directly (faster output refresh for one example).
- [ ] Add validation that output files contain call-boundary trace markers (`[CALL ...] START/END`) and no `preview=` snippets.

## Example Coverage Gaps

- [ ] Add `examples/21_stream_events_deep_dive.py` to demonstrate handling `text`, `tool_call`, `usage`, and `done` events.
- [ ] Add `examples/22_retry_and_timeout_controls.py` to show `RetryConfig`, per-request `timeout`, and error handling (`APIError`, `RateLimitError`).
- [ ] Add `examples/23_async_batch_api.py` showing `AsyncBatchClient` submit/status/results flow.
- [ ] Add `examples/24_custom_middleware.py` with a user-defined `Middleware` implementation and request short-circuit behavior.
- [ ] Add `examples/25_custom_cache_backend.py` implementing `CacheBackend` and wiring it to `ResponseCache`.
- [ ] Add `examples/26_cost_reporting.py` focused on `CostTracker` + `CostSnapshot` reporting over multiple calls.
- [ ] Add `examples/27_tracing_opentelemetry.py` showing end-to-end OpenTelemetry setup with `TracingMiddleware`.
- [ ] Add agent examples for currently uncovered architectures: `LATSAgent`, `ReflexionAgent`, `ReWOOAgent`, `LLMCompilerAgent`.
