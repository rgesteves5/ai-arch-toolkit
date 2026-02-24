# Phase 9c: Batch API

**Status**: Queued (after Phase 6, independent)
**Why**: Unlocks offline/bulk workloads.

## `core/_batch.py`

- `BatchRequest`, `BatchResult`, `BatchJob` — frozen dataclasses
- `BatchClient(provider, model, api_key)`: submit/status/results
- Anthropic: direct payload → poll → NDJSON results
- OpenAI: JSONL upload → batch create → poll → download
- `AsyncBatchClient` — async variant

## Tests: `tests/test_batch.py`

- Mock HTTP for both provider flows
- NDJSON result parsing
- Error handling in batch results
