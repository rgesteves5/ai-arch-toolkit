"""The offline logic of ``scripts/probe_openai_responses.py`` (O01); the live calls run by hand."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
from scripts.probe_openai_responses import (
    Budget,
    Check,
    Prober,
    Sample,
    filler,
    latency_rows,
    percentile,
    reasoning_outcome,
    redact,
    render,
    replayable,
)


def test_percentile_interpolates_and_handles_small_samples() -> None:
    assert percentile([], 0.5) is None
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_a_request_over_the_cap_is_skipped_without_reaching_the_client() -> None:
    prober = Prober(client=None, budget=Budget(cap=1e-9), repeats=1)
    sample, response = prober.chat(
        "plain", "gpt-6-luna", [{"role": "user", "content": "Hi"}], effort="none", max_tokens=10
    )
    assert response is None
    assert sample.skipped
    assert not sample.ok
    assert prober.budget.skipped == 1
    assert prober.budget.spent == 0.0


def test_redact_removes_organization_ids_and_key_fragments() -> None:
    text = "Rate limit reached in organization org-AbC123 for key sk-proj-****abcd.\n Retry."
    redacted = redact(text)
    assert "AbC123" not in redacted
    assert "abcd" not in redacted
    assert "\n" not in redacted
    assert redact("x" * 400, limit=50).endswith("...")


def test_replayable_items_use_the_wire_names() -> None:
    call = ResponseFunctionToolCall.model_validate(
        {
            "arguments": "{}",
            "call_id": "c1",
            "name": "get_weather",
            "type": "function_call",
            "id": "fc_1",
            "status": "completed",
            "async": False,
        }
    )
    item = replayable(call)
    assert item["async"] is False
    assert "async_" not in item


def test_reasoning_outcome_counts_encrypted_content_and_summaries() -> None:
    encrypted = ResponseReasoningItem(
        id="rs_1", type="reasoning", summary=[], encrypted_content="gAAAA"
    )
    plain = ResponseReasoningItem(id="rs_2", type="reasoning", summary=[])
    outcome = reasoning_outcome(SimpleNamespace(output=[encrypted, plain]))
    assert outcome == "2 reasoning item(s): encrypted_content on 1, summary text on 0"
    assert reasoning_outcome(SimpleNamespace(output=[])) == "no reasoning item"


def test_latency_rows_group_requests_and_leave_the_checks_out() -> None:
    samples = [
        Sample(scenario="plain", endpoint="chat", model="m", effort="none", ok=True, total_s=1.0),
        Sample(scenario="plain", endpoint="chat", model="m", effort="none", ok=True, total_s=3.0),
        Sample(scenario="plain", endpoint="chat", model="m", effort="none", error="429: slow"),
        Sample(scenario="check", endpoint="chat", model="m", effort="none", ok=True),
    ]
    rows = latency_rows(samples)
    assert len(rows) == 1
    cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
    assert cells[:6] == ["plain", "m", "none", "chat", "no", "2/3"]
    assert cells[6] == "2.00"  # p50 of the successful requests


def test_render_escapes_table_cells() -> None:
    prober = Prober(client=None, budget=Budget(cap=1.0), repeats=1)
    prober.checks.append(Check("6a", "stop sent in the body", "400: a | b"))
    report = render(prober, started="2026-10-02T00:00:00Z", models={"light": "gpt-6-luna"})
    assert "| 6a | stop sent in the body | 400: a \\| b |" in report
    assert "## Latency, tokens and cost" in report


def test_filler_is_long_enough_and_fixed() -> None:
    text = filler(9000)
    assert len(text) >= 9000
    assert text == filler(9000)
