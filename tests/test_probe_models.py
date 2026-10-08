from __future__ import annotations

import json
import struct
import zlib
from datetime import date
from pathlib import Path

import pytest
from scripts.probe_models import (
    Classification,
    ModelProbeConfig,
    ProbeAssertionError,
    ProbeResult,
    _probe_vision,
    _refused_by_adapter,
    _sanitize_error_message,
    add_numbers,
    catalog_fragment,
    classify_exception,
    load_model_configs,
    red_square_png,
    render_markdown_report,
    result_to_json_row,
    select_models,
    select_scenarios,
)

from ai_arch_toolkit.core import ModelCatalog, Provenance, RequestError, prepare_tools, user
from ai_arch_toolkit.core._content import ImagePart
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError, UnpricedModelError
from ai_arch_toolkit.core._providers import create_provider
from ai_arch_toolkit.core._providers._base import refused_or_unread
from ai_arch_toolkit.core._response import Response
from tests.fake_provider import fake_llm
from tests.provider_calls import prepare


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

[[models]]
id = "muse-spark-test"
provider = "meta"
tool_choice = "auto"
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
        ),
        ModelProbeConfig(
            id="muse-spark-test",
            provider="meta",
            scenarios=("plain", "tools_loop", "structured"),
            tool_choice="auto",
        ),
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
    # An account without credits is billing, not a framework bug or a rate limit.
    low_balance = "Your credit balance is too low to access the Anthropic API."
    assert classify_exception(APIError(400, low_balance)) == "billing"
    no_credits = "Your team has either used all available credits or reached its monthly..."
    assert classify_exception(APIError(403, no_credits)) == "billing"
    quota = "You exceeded your current quota (insufficient_quota)."
    assert classify_exception(RateLimitError(429, quota)) == "billing"
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


def test_sanitize_error_message_redacts_meta_api_keys() -> None:
    sanitized = _sanitize_error_message("API 401: invalid key LLM|111111111111111|fakeKEY.", 600)

    assert "111111111111111" not in sanitized
    assert "fakeKEY" not in sanitized
    assert "invalid key <redacted>" in sanitized


def test_the_vision_image_is_a_valid_red_png() -> None:
    data = red_square_png(4)

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height, depth, colour = struct.unpack(">IIBB", data[16:26])
    assert (width, height, depth, colour) == (4, 4, 8, 2)
    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8 : start - 4])[0]
    rows = zlib.decompress(data[start : start + length])
    assert rows == (b"\x00" + b"\xff\x00\x00" * 4) * 4


async def test_the_vision_probe_sends_the_image_and_reads_the_colour() -> None:
    llm, provider = fake_llm(Response(text="Red."))

    data = await _probe_vision(llm)

    (message,) = provider.last.messages
    text, picture = message["content"]
    assert "colour" in text and isinstance(picture, ImagePart)
    assert picture.source == red_square_png() and picture.media_type == "image/png"
    assert data["text_preview"] == "Red."


async def test_the_vision_probe_fails_when_the_model_does_not_see_red() -> None:
    llm, _ = fake_llm(Response(text="I cannot see images."))

    with pytest.raises(ProbeAssertionError, match="Expected red"):
        await _probe_vision(llm)


def _result(
    model: str, provider: str, scenario: str, classification: Classification
) -> ProbeResult:
    return ProbeResult(
        model=model,
        provider=provider,
        scenario=scenario,
        ok=classification == "ok",
        classification=classification,
        latency_s=0.1,
    )


def _refused(model: str, provider: str, scenario: str) -> ProbeResult:
    """The adapter refused the call before sending it, as its own tables say."""
    return ProbeResult(
        model=model,
        provider=provider,
        scenario=scenario,
        ok=False,
        classification="unexpected_response",
        latency_s=0.0,
        error_type="RequestError",
        message=f"{model} takes no output_schema",
        refused_by_adapter=True,
    )


def test_the_catalog_fragment_states_only_what_a_run_proved(tmp_path: Path) -> None:
    """C06d: a pass states the fact true, the adapter's refusal false; a rate limit, a timeout
    or a wrong answer states nothing. The fragment loads into a catalog as probe facts."""
    results = [
        _result("gpt-test", "openai", "tools_loop", "ok"),
        _refused("gpt-test", "openai", "structured"),
        _result("gpt-test", "openai", "json_mode", "rate_limit"),
        _result("gpt-test", "openai", "stream", "timeout"),
        _result("gpt-test", "openai", "plain", "ok"),  # no catalog fact
        _result("grok-test", "xai", "tools_loop", "unexpected_response"),
    ]
    path = tmp_path / "run.catalog.toml"
    path.write_text(
        catalog_fragment(results, verified_at=date(2026, 10, 8), ref="probe run 20261008T0000Z"),
        encoding="utf-8",
    )
    catalog = ModelCatalog(defaults=False)

    catalog.load(path)

    found = catalog.get("gpt-test", provider="openai")
    assert found is not None
    assert (found.tools, found.structured_output) == (True, False)
    assert found.json_mode is None and found.streaming is None
    assert found.provenance("tools") == Provenance(
        kind="probe", ref="probe run 20261008T0000Z", verified_at=date(2026, 10, 8)
    )
    assert catalog.get("grok-test", provider="xai") is None  # it proved nothing


def test_an_empty_run_gives_a_fragment_that_loads_empty(tmp_path: Path) -> None:
    path = tmp_path / "run.catalog.toml"
    path.write_text(catalog_fragment([], verified_at=date(2026, 10, 8), ref="run"), "utf-8")
    catalog = ModelCatalog(defaults=False)

    catalog.load(path)

    assert catalog.entries() == []


def test_a_provider_error_that_reads_as_unsupported_is_left_for_review(tmp_path: Path) -> None:
    """C06d: only the adapter's refusal states a fact false. A provider's error that the
    heuristic reads as unsupported may be a framework bug ("Invalid schema for
    response_format"), so the fragment lists it as a comment, and states nothing."""
    results = [
        ProbeResult(
            model="gpt-test",
            provider="openai",
            scenario="structured",
            ok=False,
            classification="unsupported_capability",
            latency_s=0.2,
            status_code=400,
            error_type="APIError",
            message="API 400: Invalid schema for response_format 'answer'",
        ),
        _refused("gpt-test", "openai", "tools_loop"),
    ]
    text = catalog_fragment(results, verified_at=date(2026, 10, 8), ref="run")
    path = tmp_path / "run.catalog.toml"
    path.write_text(text, encoding="utf-8")
    catalog = ModelCatalog(defaults=False)

    catalog.load(path)

    found = catalog.get("gpt-test", provider="openai")
    assert found is not None
    assert found.tools is False and found.structured_output is None
    flagged = [line for line in text.splitlines() if "structured_output" in line]
    assert flagged and all(line.startswith("#") for line in flagged)


def test_only_the_adapters_own_preparation_is_its_refusal() -> None:
    """The adapter's ``prepare`` refusing what the model does not take is its refusal; the SDK's
    validation, an unpriced model or the caller's arguments are not."""
    agents = create_provider("grok-4.20-multi-agent", provider="xai", api_key="test-key")
    with pytest.raises(RequestError) as refused:  # its client tools are not supported
        prepare(agents, [user("hi")], tools=prepare_tools([add_numbers]))

    assert _refused_by_adapter(refused.value)
    for other in (
        refused_or_unread(ValueError("bad field"), sent=False),
        UnpricedModelError("gpt-test has no price"),
        RequestError("json_mode and output_schema are mutually exclusive"),
    ):
        try:
            raise other
        except RequestError as raised:
            assert not _refused_by_adapter(raised)


def test_a_fragment_keyed_by_an_alias_loads_over_the_default_catalog(tmp_path: Path) -> None:
    """The inventory names xAI's models by their aliases: the run's facts land on the entry of
    the model, over the seed."""
    results = [
        _result("grok-4.20-reasoning", "xai", "tools_loop", "ok"),
        _result("claude-haiku-4-5-20251001", "anthropic", "stream", "ok"),
    ]
    path = tmp_path / "run.catalog.toml"
    path.write_text(
        catalog_fragment(results, verified_at=date(2026, 10, 8), ref="probe run X"), "utf-8"
    )
    catalog = ModelCatalog()

    catalog.load(path)

    grok = catalog.get("grok-4.20-reasoning")
    assert grok is not None and grok.model == "grok-4.20-0309-reasoning"
    tools = grok.provenance("tools")
    assert tools is not None and tools.kind == "probe"
    assert grok.context_window == 1_000_000
    haiku = catalog.get("claude-haiku-4-5")
    streaming = haiku.provenance("streaming") if haiku is not None else None
    assert streaming is not None and streaming.kind == "probe"
