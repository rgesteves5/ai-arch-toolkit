#!/usr/bin/env python3
"""Run ad hoc live provider probes across configured model IDs.

This script intentionally stays outside normal pytest/CI. It calls live model
APIs, can cost money, and provider availability can be transient.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ai_arch_toolkit import LLM, OutputSchema, RetryConfig, ToolGroup, run_tools, tool, user
from ai_arch_toolkit.core._exceptions import RateLimitError

type ScenarioName = Literal["plain", "tools_loop", "structured", "json_mode", "stream", "thinking"]
type Classification = Literal[
    "ok",
    "auth_error",
    "content_policy",
    "rate_limit",
    "transient_provider_error",
    "unsupported_model",
    "unsupported_capability",
    "framework_bug",
    "unexpected_response",
    "timeout",
]

SCENARIOS: tuple[ScenarioName, ...] = (
    "plain",
    "tools_loop",
    "structured",
    "json_mode",
    "stream",
    "thinking",
)
SUITES: dict[str, tuple[ScenarioName, ...]] = {
    "smoke": ("plain", "tools_loop", "structured"),
    "full": SCENARIOS,
}
DEFAULT_MODELS_PATH = Path("scripts/model_probe_models.toml")
DEFAULT_OUTPUT_DIR = Path("scripts/output/model-probes")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProbeConfig:
    """Configuration for one live model probe target."""

    id: str
    provider: str = ""
    scenarios: tuple[ScenarioName, ...] = ("plain", "tools_loop", "structured")
    allow_transient: bool = False
    require_thinking: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeResult:
    """Serializable result row for one model/scenario probe."""

    model: str
    provider: str
    scenario: str
    ok: bool
    classification: Classification
    latency_s: float
    status_code: int | None = None
    error_type: str | None = None
    message: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    text_preview: str = ""
    final_text_preview: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    thinking_blocks: int = 0


class ProbeAssertionError(AssertionError):
    """Assertion failure with a probe classification."""

    def __init__(
        self, message: str, classification: Classification = "unexpected_response"
    ) -> None:
        self.classification = classification
        super().__init__(message)


@tool
def add_numbers(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.
    """
    return str(a + b)


ADD_TOOLS = ToolGroup(add_numbers)


def load_model_configs(path: Path) -> list[ModelProbeConfig]:
    """Load model probe configuration from TOML."""
    with path.open("rb") as f:
        data = tomllib.load(f)

    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("Expected `models` to be a TOML array.")

    models: list[ModelProbeConfig] = []
    for index, raw in enumerate(raw_models, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"models[{index}] must be a table.")
        model_id = raw.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"models[{index}].id must be a non-empty string.")
        kwargs = raw.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(f"models[{index}].kwargs must be a table or inline table.")
        models.append(
            ModelProbeConfig(
                id=model_id,
                provider=str(raw.get("provider", "")),
                scenarios=_parse_scenarios(raw.get("scenarios", SUITES["smoke"])),
                allow_transient=bool(raw.get("allow_transient", False)),
                require_thinking=bool(raw.get("require_thinking", False)),
                kwargs=dict(kwargs),
            )
        )
    return models


def select_models(
    models: Sequence[ModelProbeConfig],
    requested_ids: Sequence[str] | None,
) -> list[ModelProbeConfig]:
    """Filter models by requested ID while preserving config order."""
    if not requested_ids:
        return list(models)
    requested = set(requested_ids)
    return [model for model in models if model.id in requested]


def select_scenarios(
    model: ModelProbeConfig,
    requested: Sequence[ScenarioName],
) -> tuple[ScenarioName, ...]:
    """Return requested scenarios enabled for this model."""
    enabled = set(model.scenarios)
    return tuple(scenario for scenario in requested if scenario in enabled)


def classify_exception(exc: BaseException, *, allow_transient: bool = False) -> Classification:
    """Classify live provider errors into actionable buckets."""
    if isinstance(exc, ProbeAssertionError):
        return exc.classification

    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()

    if isinstance(exc, TimeoutError):
        return "transient_provider_error" if allow_transient else "timeout"
    if isinstance(exc, RateLimitError) or status_code == 429 or "rate limit" in message:
        return "rate_limit"
    if _looks_like_content_policy(message):
        return "content_policy"
    if _has_any(message, ("api key", "authentication", "unauthorized", "forbidden")):
        return "auth_error"
    if status_code in (401, 403):
        return "auth_error"
    if status_code == 404 or _has_any(message, ("model_not_found", "model not found")):
        return "unsupported_model"
    if _has_any(message, ("does not exist", "not found")) and "model" in message:
        return "unsupported_model"
    if status_code in (500, 502, 503, 504):
        return "transient_provider_error"
    if _has_any(message, ("high demand", "temporarily unavailable", "try again later")):
        return "transient_provider_error"
    if _looks_like_unsupported_capability(message):
        return "unsupported_capability"
    if _looks_like_framework_bug(message, status_code):
        return "framework_bug"
    return "unexpected_response"


def result_to_json_row(result: ProbeResult) -> str:
    """Serialize a probe result as one JSONL row."""
    return json.dumps(asdict(result), sort_keys=True)


def render_markdown_report(results: Sequence[ProbeResult], *, started_at: str) -> str:
    """Render a human-readable Markdown probe report."""
    total = len(results)
    passed = sum(1 for result in results if result.ok)
    failed = total - passed
    lines = [
        "# Model Probe Report",
        "",
        f"- started_at: `{started_at}`",
        f"- total: `{total}`",
        f"- passed: `{passed}`",
        f"- failed: `{failed}`",
        "",
        "| Model | Provider | Scenario | Status | Classification | Latency | Cost |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        cost = "" if result.cost is None else f"${result.cost:.6f}"
        lines.append(
            "| "
            f"{result.model} | {result.provider} | {result.scenario} | {status} | "
            f"{result.classification} | {result.latency_s:.2f}s | {cost} |"
        )

    failures = [result for result in results if not result.ok]
    if failures:
        lines.extend(["", "## Failure Details", ""])
        for result in failures:
            lines.extend(
                [
                    f"### {result.model} / {result.scenario}",
                    "",
                    f"- classification: `{result.classification}`",
                    f"- status_code: `{result.status_code}`",
                    f"- error_type: `{result.error_type}`",
                    f"- message: {result.message or '<empty>'}",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


async def run_probe(
    model: ModelProbeConfig,
    scenario: ScenarioName,
    *,
    timeout_seconds: float,
    max_retries: int,
) -> ProbeResult:
    """Run one live probe scenario and return a structured result."""
    started = time.monotonic()
    try:
        response_data = await _run_probe_success_path(
            model,
            scenario,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return ProbeResult(
            model=model.id,
            provider=model.provider,
            scenario=scenario,
            ok=True,
            classification="ok",
            latency_s=round(time.monotonic() - started, 3),
            **response_data,
        )
    except BaseException as exc:
        return ProbeResult(
            model=model.id,
            provider=model.provider,
            scenario=scenario,
            ok=False,
            classification=classify_exception(exc, allow_transient=model.allow_transient),
            latency_s=round(time.monotonic() - started, 3),
            status_code=getattr(exc, "status_code", None),
            error_type=type(exc).__name__,
            message=_sanitize_error_message(str(exc), 600),
        )


async def _run_probe_success_path(
    model: ModelProbeConfig,
    scenario: ScenarioName,
    *,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    llm_kwargs = dict(model.kwargs)
    llm_timeout = float(llm_kwargs.pop("timeout", timeout_seconds))
    retry = RetryConfig(max_retries=max_retries, base_delay=1.0, max_delay=5.0)
    async with LLM(model.id, timeout=llm_timeout, retry=retry, **llm_kwargs) as llm:
        if scenario == "plain":
            return await _probe_plain(llm)
        if scenario == "tools_loop":
            return await _probe_tools_loop(llm)
        if scenario == "structured":
            return await _probe_structured(llm)
        if scenario == "json_mode":
            return await _probe_json_mode(llm)
        if scenario == "stream":
            return await _probe_stream(llm)
        if scenario == "thinking":
            return await _probe_thinking(llm, require_thinking=model.require_thinking)
    raise ProbeAssertionError(f"Unknown scenario: {scenario}", "framework_bug")


async def _probe_plain(llm: LLM) -> dict[str, Any]:
    response = await llm.complete("Reply with exactly: PLAIN_OK")
    if "PLAIN_OK" not in _normalize_text(response.text):
        raise ProbeAssertionError(f"Expected PLAIN_OK, got {response.text!r}")
    return _response_data(response, text_preview=response.text)


async def _probe_tools_loop(llm: LLM) -> dict[str, Any]:
    messages = [
        user(
            "Call add_numbers exactly once with arguments a=2 and b=3. "
            "After the tool result, answer with only the returned sum."
        )
    ]
    first = await llm.complete(messages, tools=ADD_TOOLS, tool_choice="required")
    if not first.tool_calls:
        raise ProbeAssertionError("Expected at least one tool call.")
    tool_calls = tuple({"name": call.name, "input": dict(call.input)} for call in first.tool_calls)

    messages.append(first.to_message())
    try:
        messages.extend(await run_tools(first, ADD_TOOLS))
    except (KeyError, TypeError) as exc:
        raise ProbeAssertionError(f"Tool execution failed for {tool_calls!r}: {exc}") from exc
    final = await llm.complete(messages, tools=ADD_TOOLS)
    if "5" not in _normalize_text(final.text):
        raise ProbeAssertionError(f"Expected final answer to contain 5, got {final.text!r}")

    return {
        **_response_data(
            final,
            text_preview=first.text,
            final_text_preview=final.text,
            extra_input_tokens=first.usage.input_tokens,
            extra_output_tokens=first.usage.output_tokens,
            extra_cost=first.cost,
        ),
        "tool_calls": tool_calls,
    }


async def _probe_structured(llm: LLM) -> dict[str, Any]:
    response = await llm.complete(
        "Return JSON with exactly this semantic value: answer is 5.",
        output_schema=_answer_schema(),
    )
    if not isinstance(response.parsed, dict) or response.parsed.get("answer") != 5:
        raise ProbeAssertionError(
            f"Expected parsed answer=5, got {response.parsed!r}",
            "framework_bug",
        )
    return _response_data(response, text_preview=response.text)


async def _probe_json_mode(llm: LLM) -> dict[str, Any]:
    response = await llm.complete(
        'Return only a JSON object with this shape: {"answer": 5}',
        json_mode=True,
    )
    parsed = _parse_json_object(response.text)
    if parsed.get("answer") != 5:
        raise ProbeAssertionError(f"Expected JSON answer=5, got {response.text!r}")
    return _response_data(response, text_preview=response.text)


async def _probe_stream(llm: LLM) -> dict[str, Any]:
    stream = llm.stream("What is 2 + 3? Reply with only 5.")
    chunks: list[str] = []
    async for chunk in stream:
        chunks.append(chunk)
    final = stream.response
    combined = "".join(chunks)
    if not chunks:
        raise ProbeAssertionError("Expected at least one stream chunk.")
    if "5" not in _normalize_text(final.text or combined):
        raise ProbeAssertionError(f"Expected 5, got {final.text or combined!r}")
    return _response_data(final, text_preview=combined)


async def _probe_thinking(llm: LLM, *, require_thinking: bool) -> dict[str, Any]:
    response = await llm.complete("Reply with exactly: THINKING_OK", thinking=True)
    if response.text and "THINKING_OK" not in _normalize_text(response.text):
        raise ProbeAssertionError(f"Expected THINKING_OK, got {response.text!r}")
    if require_thinking and not response.thinking:
        raise ProbeAssertionError("Expected thinking blocks but none were returned.")
    return _response_data(
        response,
        text_preview=response.text,
        thinking_blocks=len(response.thinking),
    )


def _response_data(
    response: Any,
    *,
    text_preview: str = "",
    final_text_preview: str = "",
    tool_calls: tuple[dict[str, Any], ...] = (),
    thinking_blocks: int = 0,
    extra_input_tokens: int = 0,
    extra_output_tokens: int = 0,
    extra_cost: float | None = None,
) -> dict[str, Any]:
    cost = response.cost
    if extra_cost is not None:
        cost = (cost or 0.0) + extra_cost
    return {
        "input_tokens": response.usage.input_tokens + extra_input_tokens,
        "output_tokens": response.usage.output_tokens + extra_output_tokens,
        "cost": cost,
        "text_preview": _truncate(text_preview, 200),
        "final_text_preview": _truncate(final_text_preview, 200),
        "tool_calls": tool_calls,
        "thinking_blocks": thinking_blocks,
    }


def _answer_schema() -> OutputSchema:
    return OutputSchema(
        name="probe_answer",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live model probes from a TOML inventory.")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS_PATH)
    parser.add_argument("--model", action="append", default=None, help="Only run this model ID.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=SCENARIOS,
        default=None,
        help="Run only this scenario. Can be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print final summary as JSON.")
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> int:
    started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    models = select_models(load_model_configs(args.models), args.model)
    requested_scenarios = tuple(args.scenario) if args.scenario else SUITES[args.suite]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{run_id}.jsonl"
    markdown_path = args.output_dir / f"{run_id}.md"

    results: list[ProbeResult] = []
    progress = sys.stderr if args.json else sys.stdout
    for model in models:
        scenarios = select_scenarios(model, requested_scenarios)
        if not scenarios:
            print(f"[skip] {model.id}: no enabled scenarios selected", file=progress)
            continue
        for scenario in scenarios:
            print(f"[run] {model.id} / {scenario}", file=progress, flush=True)
            result = await run_probe(
                model,
                scenario,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            results.append(result)
            print(
                f"  -> {'ok' if result.ok else result.classification} ({result.latency_s:.2f}s)",
                file=progress,
                flush=True,
            )
            if args.fail_fast and not result.ok:
                break
        if args.fail_fast and any(not result.ok for result in results):
            break

    jsonl_path.write_text(
        "".join(result_to_json_row(result) + "\n" for result in results),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown_report(results, started_at=started_at),
        encoding="utf-8",
    )

    failed = sum(1 for result in results if not result.ok)
    summary = {
        "total": len(results),
        "failed": failed,
        "jsonl_path": str(jsonl_path),
        "markdown_path": str(markdown_path),
    }
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print("\nSummary")
        print(f"  total: {summary['total']}")
        print(f"  failed: {summary['failed']}")
        print(f"  jsonl: {jsonl_path}")
        print(f"  markdown: {markdown_path}")
    return 1 if failed else 0


def _parse_scenarios(value: Any) -> tuple[ScenarioName, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("scenarios must be a list.")
    scenarios: list[ScenarioName] = []
    for raw in value:
        if raw not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {raw!r}. Valid scenarios: {', '.join(SCENARIOS)}")
        scenarios.append(raw)
    return tuple(scenarios)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ProbeAssertionError(f"Response is not JSON: {text!r}") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ProbeAssertionError(f"Expected JSON object, got {type(parsed).__name__}.")
    return parsed


def _normalize_text(text: str) -> str:
    return text.strip().strip("\"'`")


def _truncate(text: str, limit: int) -> str:
    normalized = text.replace("\n", " ").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _sanitize_error_message(text: str, limit: int) -> str:
    redacted = re.sub(r"(Team:\s*)[0-9a-fA-F-]+", r"\1<redacted>", text)
    redacted = re.sub(r"(API key ID:\s*)[0-9a-fA-F-]+", r"\1<redacted>", redacted)
    return _truncate(redacted, limit)


def _has_any(message: str, needles: Iterable[str]) -> bool:
    return any(needle in message for needle in needles)


def _looks_like_unsupported_capability(message: str) -> bool:
    capability_words = ("tool", "json", "schema", "response_format", "thinking", "reasoning")
    rejection_words = ("unsupported", "not supported", "does not support", "invalid")
    if not _has_any(message, capability_words):
        return False
    if "unsupported parameter" in message and "max_tokens" in message:
        return False
    return _has_any(message, rejection_words)


def _looks_like_content_policy(message: str) -> bool:
    return _has_any(
        message,
        (
            "content violates usage guidelines",
            "safety_check",
            "safety check",
            "policy violation",
            "content policy",
        ),
    )


def _looks_like_framework_bug(message: str, status_code: int | None) -> bool:
    if "unsupported parameter" in message:
        return True
    if _has_any(message, ("malformed", "invalid request", "bad request")):
        return True
    return status_code == 400


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
