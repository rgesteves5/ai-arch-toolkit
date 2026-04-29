from __future__ import annotations

import json
from pathlib import Path

from scripts.probe_models import (
    ModelProbeConfig,
    ProbeAssertionError,
    ProbeResult,
    _sanitize_error_message,
    classify_exception,
    load_model_configs,
    render_markdown_report,
    result_to_json_row,
    select_models,
    select_scenarios,
)

from ai_arch_toolkit.core._exceptions import APIError, RateLimitError


def test_load_model_configs(tmp_path: Path) -> None:
    config = tmp_path / "models.toml"
    config.write_text(
        """
[[models]]
id = "gpt-test"
provider = "openai"
scenarios = ["plain", "tools_loop"]
allow_transient = true
require_thinking = false
kwargs = { temperature = 0.0, max_tokens = 64 }
""".strip(),
        encoding="utf-8",
    )

    models = load_model_configs(config)

    assert models == [
        ModelProbeConfig(
            id="gpt-test",
            provider="openai",
            scenarios=("plain", "tools_loop"),
            allow_transient=True,
            kwargs={"temperature": 0.0, "max_tokens": 64},
        )
    ]


def test_select_models_and_scenarios() -> None:
    models = [
        ModelProbeConfig(id="a", scenarios=("plain", "structured")),
        ModelProbeConfig(id="b", scenarios=("tools_loop",)),
    ]

    assert select_models(models, ["b"]) == [models[1]]
    assert select_scenarios(models[0], ("plain", "tools_loop", "structured")) == (
        "plain",
        "structured",
    )


def test_classify_exception() -> None:
    assert classify_exception(ValueError("No API key provided")) == "auth_error"
    assert (
        classify_exception(
            APIError(403, "Content violates usage guidelines: SAFETY_CHECK_TYPE_BIO")
        )
        == "content_policy"
    )
    assert classify_exception(RateLimitError(429, "rate limit")) == "rate_limit"
    assert classify_exception(APIError(503, "high demand")) == "transient_provider_error"
    assert classify_exception(APIError(404, "model not found")) == "unsupported_model"
    assert (
        classify_exception(APIError(400, "Unsupported parameter: 'max_tokens'")) == "framework_bug"
    )
    assert (
        classify_exception(APIError(400, "tools are not supported by this model"))
        == "unsupported_capability"
    )
    assert classify_exception(TimeoutError(), allow_transient=True) == "transient_provider_error"
    assert classify_exception(TimeoutError()) == "timeout"
    assert classify_exception(ProbeAssertionError("bad answer")) == "unexpected_response"


def test_result_to_json_row() -> None:
    result = ProbeResult(
        model="gpt-test",
        provider="openai",
        scenario="tools_loop",
        ok=True,
        classification="ok",
        latency_s=1.23,
        input_tokens=10,
        output_tokens=5,
        cost=0.01,
        tool_calls=({"name": "add_numbers", "input": {"a": 2, "b": 3}},),
    )

    row = json.loads(result_to_json_row(result))

    assert row["model"] == "gpt-test"
    assert row["classification"] == "ok"
    assert row["tool_calls"] == [{"name": "add_numbers", "input": {"a": 2, "b": 3}}]


def test_render_markdown_report() -> None:
    results = [
        ProbeResult(
            model="ok-model",
            provider="openai",
            scenario="plain",
            ok=True,
            classification="ok",
            latency_s=0.1,
        ),
        ProbeResult(
            model="bad-model",
            provider="gemini",
            scenario="structured",
            ok=False,
            classification="transient_provider_error",
            latency_s=0.2,
            status_code=503,
            error_type="APIError",
            message="high demand",
        ),
    ]

    report = render_markdown_report(results, started_at="2026-04-28T00:00:00Z")

    assert "# Model Probe Report" in report
    assert "| ok-model | openai | plain | PASS | ok | 0.10s |  |" in report
    assert "## Failure Details" in report
    assert "bad-model / structured" in report


def test_sanitize_error_message_redacts_provider_ids() -> None:
    message = (
        "API 403: Content violates usage guidelines. "
        "Team: a64e110b-ab51-404b-8477-39fe42c5ee4d, "
        "API key ID: d63e1574-a3d4-43de-9608-dc25103dd4d6"
    )

    sanitized = _sanitize_error_message(message, 600)

    assert "Team: <redacted>" in sanitized
    assert "API key ID: <redacted>" in sanitized
    assert "a64e110b" not in sanitized
    assert "d63e1574" not in sanitized
