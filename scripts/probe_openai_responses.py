#!/usr/bin/env python3
"""Front O, task O01: OpenAI's Responses API against Chat Completions, live.

Measures latency (total time, time to the first text token, the server's processing time), cached
tokens and cost on both endpoints, and records how the Responses API answers what D43 depends on
(``blackboard/tasks/O01-openai-responses-probe.md``):

2. does a stateless response (``store: false``) carry the encrypted reasoning by itself, or only
   with ``include``;
3. a reasoning item replayed without its encrypted content, or without the item that follows it,
   and a tool turn replayed without its reasoning;
4. reasoning replayed to a model of another family;
5. a function tool with ``strict`` left out;
6. ``stop``, ``seed``, ``frequency_penalty`` and ``presence_penalty``, which Responses lacks;
7. sampling parameters and logprobs on the GPT-6 models;
8. tool calls while reasoning, reasoning summaries, and Chat Completions' refusals.

It drives the ``openai`` SDK directly, since the adapter is what O03 builds, and it costs money:
before each request it checks that the worst case still fits ``--max-cost``, and skips the request
when it does not. Run it from the repository root:

    set -a; source .env; set +a
    uv run python scripts/probe_openai_responses.py --repeats 6 --max-cost 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import openai

from ai_arch_toolkit.core import pricing

type Endpoint = Literal["chat", "responses"]

DEFAULT_OUTPUT_DIR = Path("scripts/output/model-probes")
ENCRYPTED = ("reasoning.encrypted_content",)
# The keys every function_call item has; anything else set on one is recorded.
USUAL_CALL_FIELDS = frozenset({"arguments", "call_id", "name", "type", "id", "status"})

WEATHER: dict[str, Any] = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name."}},
        "required": ["city"],
        "additionalProperties": False,
    },
}
# Outside strict mode's subset: "unit" is optional and other properties are allowed.
LOOSE_WEATHER: dict[str, Any] = {
    **WEATHER,
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}
PLAIN = "In two sentences, explain what a hash table is."
TWO_CITIES = (
    "What is the weather in Lisbon and in Porto? Call the tool once per city, then answer in one "
    "sentence."
)
ONE_CITY = "What is the weather in Lisbon? Use the tool."
REASONED_CALL = (
    "Three friends live in the capitals of Portugal, Spain and France. Work out which of the "
    "three cities lies furthest north, then call the tool for that city only and tell me its "
    "weather."
)
PUZZLE = (
    "A train leaves at 14:47 and arrives at 17:15 the same day. How many minutes does the trip "
    "take? Answer with the number only."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """One request: what was asked and what came back."""

    scenario: str
    endpoint: Endpoint
    model: str
    effort: str | None
    stream: bool = False
    ok: bool = False
    skipped: bool = False
    error: str = ""
    total_s: float = 0.0
    ttft_s: float | None = None
    server_ms: float | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class Check:
    """The answer to one of the task's questions."""

    number: str
    question: str
    outcome: str


@dataclass(slots=True)
class Budget:
    """Spending so far against a cap: a request runs only when its worst case still fits."""

    cap: float
    spent: float = 0.0
    skipped: int = 0

    def admits(self, model: str, payload: Any, max_output: int) -> bool:
        # Two characters per token over-counts both prose and JSON; the output limit is the worst.
        input_tokens = len(json.dumps(payload, default=str)) // 2
        worst = cost(model, input_tokens=input_tokens, output_tokens=max_output)
        if self.spent + worst > self.cap:
            self.skipped += 1
            return False
        return True


def cost(model: str, *, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """USD at the package's prices; ``input_tokens`` includes the cached ones, as OpenAI counts."""
    value = pricing.estimate_cost(
        model,
        input_tokens=max(input_tokens - cached_tokens, 0),
        output_tokens=output_tokens,
        cache_read_tokens=cached_tokens,
    )
    if value is None:
        raise SystemExit(f"{model} has no price in the package's table")
    return value


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q`` quantile by linear interpolation, or ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def redact(text: str, limit: int = 300) -> str:
    """An error message on one line, without organization ids or key fragments."""
    text = re.sub(r"\borg-[A-Za-z0-9]+", "org-<redacted>", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_*\-]+", "sk-<redacted>", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe(exc: openai.APIError) -> str:
    status = getattr(exc, "status_code", None)
    return redact(f"{status}: {exc.message}" if status else f"{type(exc).__name__}: {exc.message}")


def server_ms(headers: Any) -> float | None:
    value = headers.get("openai-processing-ms")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def replayable(item: Any) -> dict[str, Any]:
    """An output item as input to the next request, under its wire names (``async``)."""
    return item.model_dump(mode="json", exclude_none=True, by_alias=True)


def chat_tool(spec: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": spec}


def responses_tool(spec: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    return {"type": "function", **spec, "strict": strict}


def weather(arguments: str) -> str:
    """The fixed answer of the fake weather tool."""
    try:
        city = json.loads(arguments).get("city", "?")
    except (json.JSONDecodeError, AttributeError):
        city = "?"
    return json.dumps({"city": city, "temperature_c": 21, "sky": "clear"})


def filler(chars: int) -> str:
    """Fixed text of at least ``chars`` characters: a prefix long enough for prompt caching."""
    colors = ("red", "green", "blue", "amber", "white", "violet", "grey")
    lines: list[str] = []
    size = 0
    n = 0
    while size < chars:
        n += 1
        blinks, form = n % 5 + 2, n * 37 % 997
        line = f"Rule {n}: when the {colors[n % 7]} lamp blinks {blinks} times, file F-{form}."
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def reasoning_outcome(response: Any) -> str:
    items = [item for item in response.output if item.type == "reasoning"]
    if not items:
        return "no reasoning item"
    encrypted = sum(1 for item in items if item.encrypted_content)
    summarized = sum(1 for item in items if any(part.text for part in item.summary))
    return (
        f"{len(items)} reasoning item(s): encrypted_content on {encrypted}, "
        f"summary text on {summarized}"
    )


def answer_of(response: Any) -> str:
    text = (response.output_text or "").strip()
    calls = sum(1 for item in response.output if item.type == "function_call")
    return redact(text, 100) if text else f"{calls} tool call(s)"


def logprobs_outcome(response: Any) -> str:
    tokens = sum(
        len(part.logprobs or [])
        for item in response.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
    )
    return f"accepted; logprobs on {tokens} token(s)"


def _chat_tokens(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    prompt = usage.prompt_tokens_details
    completion = usage.completion_tokens_details
    return {
        "input_tokens": usage.prompt_tokens,
        "cached_tokens": (prompt.cached_tokens or 0) if prompt else 0,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": (completion.reasoning_tokens or 0) if completion else 0,
    }


def _responses_tokens(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "input_tokens": usage.input_tokens,
        "cached_tokens": usage.input_tokens_details.cached_tokens or 0,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens or 0,
    }


class Prober:
    """Runs the requests and keeps the samples, the checks, and the spending cap."""

    def __init__(self, client: Any, budget: Budget, *, repeats: int) -> None:
        self.client = client
        self.budget = budget
        self.repeats = repeats
        self.run_id = uuid.uuid4().hex[:8]
        self.samples: list[Sample] = []
        self.checks: list[Check] = []
        self.phases: set[str] = set()
        self.call_fields: set[str] = set()

    # -- requests ---------------------------------------------------------------------------

    def chat(
        self,
        scenario: str,
        model: str,
        messages: list[dict[str, Any]],
        *,
        effort: str | None,
        max_tokens: int,
        tools: Sequence[dict[str, Any]] = (),
        stream: bool = False,
        **extra: Any,
    ) -> tuple[Sample, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            **extra,
        }
        if effort is not None:
            params["reasoning_effort"] = effort
        if tools:
            params["tools"] = list(tools)
        asked = {"scenario": scenario, "endpoint": "chat", "model": model, "effort": effort}
        asked["stream"] = stream
        if not self.budget.admits(model, params, max_tokens):
            return self._keep(
                Sample(**asked, skipped=True, error="skipped: over --max-cost")
            ), None
        start = time.perf_counter()
        ttft: float | None = None
        final: Any = None
        try:
            if stream:
                raw = self.client.chat.completions.with_raw_response.create(
                    **params, stream=True, stream_options={"include_usage": True}
                )
                usage = None
                for chunk in raw.parse():
                    if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                        ttft = time.perf_counter() - start
                    usage = chunk.usage or usage
            else:
                raw = self.client.chat.completions.with_raw_response.create(**params)
                final = raw.parse()
                usage = final.usage
        except openai.APIError as exc:
            elapsed = time.perf_counter() - start
            return self._keep(Sample(**asked, error=describe(exc), total_s=elapsed)), None
        total = time.perf_counter() - start
        tokens = _chat_tokens(usage)
        sample = Sample(
            **asked,
            ok=True,
            total_s=total,
            ttft_s=ttft,
            server_ms=server_ms(raw.headers),
            cost=self._charge(model, tokens),
            **tokens,
        )
        return self._keep(sample), final

    def responses(
        self,
        scenario: str,
        model: str,
        items: list[dict[str, Any]],
        *,
        effort: str | None,
        max_tokens: int,
        tools: Sequence[dict[str, Any]] = (),
        stream: bool = False,
        include: Sequence[str] | None = ENCRYPTED,
        summary: str | None = None,
        **extra: Any,
    ) -> tuple[Sample, Any]:
        params: dict[str, Any] = {
            "model": model,
            "input": items,
            "max_output_tokens": max_tokens,
            "store": False,
            **extra,
        }
        reasoning = {"effort": effort, "summary": summary}
        if reasoning := {key: value for key, value in reasoning.items() if value is not None}:
            params["reasoning"] = reasoning
        if include:
            params["include"] = list(include)
        if tools:
            params["tools"] = list(tools)
        asked = {"scenario": scenario, "endpoint": "responses", "model": model, "effort": effort}
        asked["stream"] = stream
        if not self.budget.admits(model, params, max_tokens):
            return self._keep(
                Sample(**asked, skipped=True, error="skipped: over --max-cost")
            ), None
        start = time.perf_counter()
        ttft: float | None = None
        final: Any = None
        try:
            if stream:
                raw = self.client.responses.with_raw_response.create(**params, stream=True)
                for event in raw.parse():
                    if ttft is None and event.type == "response.output_text.delta":
                        ttft = time.perf_counter() - start
                    elif event.type in (
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    ):
                        final = event.response
            else:
                raw = self.client.responses.with_raw_response.create(**params)
                final = raw.parse()
        except openai.APIError as exc:
            elapsed = time.perf_counter() - start
            return self._keep(Sample(**asked, error=describe(exc), total_s=elapsed)), None
        total = time.perf_counter() - start
        if final is None:
            return self._keep(
                Sample(**asked, error="stream ended without a response", total_s=total)
            ), None
        self._observe(final)
        tokens = _responses_tokens(final.usage)
        error = ""
        if final.status != "completed":
            error = redact(f"status {final.status}: {final.incomplete_details or final.error}")
        sample = Sample(
            **asked,
            ok=not error,
            error=error,
            total_s=total,
            ttft_s=ttft,
            server_ms=server_ms(raw.headers),
            cost=self._charge(model, tokens),
            **tokens,
        )
        return self._keep(sample), final

    def _charge(self, model: str, tokens: dict[str, int]) -> float:
        if not tokens:
            return 0.0
        amount = cost(
            model,
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            cached_tokens=tokens["cached_tokens"],
        )
        self.budget.spent += amount
        return amount

    def _observe(self, response: Any) -> None:
        for item in response.output:
            if item.type == "message" and getattr(item, "phase", None):
                self.phases.add(item.phase)
            elif item.type == "function_call":
                self.call_fields.update(set(replayable(item)) - USUAL_CALL_FIELDS)

    def _keep(self, sample: Sample) -> Sample:
        self.samples.append(sample)
        state = "skip" if sample.skipped else "ok" if sample.ok else "ERR"
        line = f"{sample.scenario:<26} {sample.endpoint:<9} {sample.model:<13} {state:<4} "
        line += f"{sample.total_s:6.2f}s ${self.budget.spent:.4f}"
        print(line + (f"  {sample.error}" if sample.error else ""), flush=True)
        return sample

    def check(self, number: str, question: str, outcome: str) -> None:
        self.checks.append(Check(number, question, outcome))

    # -- checks -----------------------------------------------------------------------------

    def run_checks(self, light: str, thinker: str, other: str) -> None:
        self._check_stateless(light, thinker)
        self._check_replay(light, thinker, other)
        self._check_strict(light)
        self._check_missing_parameters(light)
        self._check_sampling(light, thinker)
        self._check_capabilities(light, thinker)

    def _check_stateless(self, light: str, thinker: str) -> None:
        prompt = [{"role": "user", "content": PUZZLE}]
        for model in (light, thinker):
            for label, include in (("by default", None), ("with include", ENCRYPTED)):
                sample, response = self.responses(
                    "check", model, prompt, effort="low", max_tokens=2000, include=include
                )
                question = f"{model}, store false, {label}: is the reasoning returned encrypted?"
                self.check("2", question, sample.error or reasoning_outcome(response))

    def _reasoned_call(self, models: Sequence[str]) -> tuple[str, Any] | None:
        """A turn with a reasoning item and a tool call, from the first model that gives one.

        A model that sees which tool to call skips its reasoning item (seen live): the question
        makes it work something out first.
        """
        question = [{"role": "user", "content": REASONED_CALL}]
        for model in models:
            for effort in ("medium", "high"):
                _, response = self.responses(
                    "check",
                    model,
                    question,
                    effort=effort,
                    max_tokens=4000,
                    tools=[responses_tool(WEATHER)],
                )
                kinds = {item.type for item in response.output} if response else set()
                if {"reasoning", "function_call"} <= kinds:
                    return model, response
        return None

    def _check_replay(self, light: str, thinker: str, other: str) -> None:
        found = self._reasoned_call((light, thinker))
        if found is None:
            self.check("3", "a turn with reasoning and a tool call", "not run: none came back")
            return
        source, first = found
        tools = [responses_tool(WEATHER)]
        question = [{"role": "user", "content": REASONED_CALL}]
        output = [replayable(item) for item in first.output]
        reasoning = [item for item in output if item["type"] == "reasoning"]
        results = [
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": weather(item.arguments),
            }
            for item in first.output
            if item.type == "function_call"
        ]
        full = [*question, *output, *results]
        by_reference = [
            {"type": "reasoning", "id": item["id"], "summary": []}
            if item.get("type") == "reasoning"
            else item
            for item in full
        ]
        orphan = [*question, *reasoning, {"role": "user", "content": "Never mind. Say hello."}]
        unreasoned = [*question, *(i for i in output if i["type"] != "reasoning"), *results]
        cases = [
            ("3a", f"{source}: reasoning, call and result replayed (normal path)", source, full),
            (
                "3b",
                f"{source}: reasoning item by id only, no encrypted_content",
                source,
                by_reference,
            ),
            (
                "3c",
                f"{source}: reasoning item followed by a user message, not its call",
                source,
                orphan,
            ),
            (
                "3d",
                f"{source}: call and result without the reasoning (rebuild path)",
                source,
                unreasoned,
            ),
        ]
        targets = [model for model in (light, thinker, other) if model != source]
        cases += [
            (f"4{letter}", f"reasoning from {source} replayed to {model}", model, full)
            for letter, model in zip("ab", targets, strict=True)
        ]
        for number, label, model, items in cases:
            sample, response = self.responses(
                "check", model, items, effort="low", max_tokens=2000, tools=tools
            )
            self.check(number, label, sample.error or f"accepted: {answer_of(response)}")

    def _check_strict(self, light: str) -> None:
        question = [{"role": "user", "content": ONE_CITY}]
        cases = (
            ("5a", "loose schema, strict left out", {"type": "function", **LOOSE_WEATHER}),
            ("5b", "loose schema, strict false", responses_tool(LOOSE_WEATHER)),
            (
                "5c",
                "loose schema, strict true (control)",
                responses_tool(LOOSE_WEATHER, strict=True),
            ),
        )
        for number, label, tool in cases:
            sample, response = self.responses(
                "check",
                light,
                question,
                effort="none",
                max_tokens=300,
                tools=[tool],
                tool_choice="required",
            )
            echoed = ""
            if response is not None:
                tool = response.tools[0]
                schema = redact(json.dumps(tool.parameters), 220)
                echoed = f"accepted; echoed strict={tool.strict}, parameters {schema}"
            self.check(number, label, sample.error or echoed)

    def _check_missing_parameters(self, light: str) -> None:
        prompt = [{"role": "user", "content": PLAIN}]
        for number, name, value in (
            ("6a", "stop", ["."]),
            ("6b", "seed", 7),
            ("6c", "frequency_penalty", 0.5),
            ("6d", "presence_penalty", 0.5),
        ):
            sample, response = self.responses(
                "check", light, prompt, effort="none", max_tokens=200, extra_body={name: value}
            )
            outcome = sample.error or f"accepted: {answer_of(response)}"
            self.check(number, f"{name} sent in the body", outcome)

    def _check_sampling(self, light: str, thinker: str) -> None:
        prompt = [{"role": "user", "content": PLAIN}]
        cases: tuple[tuple[str, str, Endpoint, str | None, dict[str, Any]], ...] = (
            ("7a", thinker, "responses", None, {"temperature": 0.2}),
            ("7b", thinker, "chat", None, {"temperature": 0.2}),
            ("7c", light, "responses", "none", {"temperature": 0.2, "top_p": 0.9}),
            ("7d", light, "responses", "low", {"temperature": 0.2}),
        )
        for number, model, endpoint, effort, extra in cases:
            call: Callable[..., tuple[Sample, Any]] = (
                self.chat if endpoint == "chat" else self.responses
            )
            sample, _ = call("check", model, prompt, effort=effort, max_tokens=1500, **extra)
            label = f"{model} on {endpoint}, effort {effort or 'default'}, {extra}"
            self.check(number, label, sample.error or "accepted")
        sample, response = self.responses(
            "check",
            light,
            prompt,
            effort="none",
            max_tokens=200,
            top_logprobs=2,
            include=["message.output_text.logprobs"],
        )
        outcome = sample.error or logprobs_outcome(response)
        self.check("7e", f"{light}, effort none: top_logprobs with their include", outcome)

    def _check_capabilities(self, light: str, thinker: str) -> None:
        question = [{"role": "user", "content": ONE_CITY}]
        sample, response = self.responses(
            "check",
            thinker,
            question,
            effort="low",
            max_tokens=3000,
            tools=[responses_tool(WEATHER)],
            summary="auto",
        )
        outcome = sample.error
        if response is not None:
            calls = sum(1 for item in response.output if item.type == "function_call")
            outcome = outcome or f"accepted: {calls} tool call(s); {reasoning_outcome(response)}"
        self.check("8a", f"{thinker} on responses, effort low, tools, summary auto", outcome)
        sample, response = self.responses(
            "check",
            light,
            [{"role": "user", "content": PUZZLE}],
            effort="low",
            max_tokens=2000,
            summary="auto",
        )
        self.check(
            "8b",
            f"{light} on responses, effort low, summary auto",
            sample.error or reasoning_outcome(response),
        )
        sample, _ = self.chat(
            "check", thinker, question, effort=None, max_tokens=3000, tools=[chat_tool(WEATHER)]
        )
        self.check("8c", f"{thinker} on chat with tools", sample.error or "accepted")
        sample, _ = self.chat(
            "check", light, question, effort="low", max_tokens=2000, tools=[chat_tool(WEATHER)]
        )
        self.check("8d", f"{light} on chat, effort low, with tools", sample.error or "accepted")
        sample, _ = self.chat(
            "check", thinker, question, effort="none", max_tokens=300, tools=[chat_tool(WEATHER)]
        )
        self.check("8e", f"{thinker} on chat, effort none, with tools", sample.error or "accepted")
        # The model page lists "max"; the 400 of check 8e lists efforts only up to "xhigh".
        short = [{"role": "user", "content": "Say OK."}]
        sample, _ = self.chat("check", thinker, short, effort="max", max_tokens=4000)
        self.check("8f", f"{thinker} on chat, effort max", sample.error or "accepted")
        sample, _ = self.responses("check", thinker, short, effort="max", max_tokens=4000)
        self.check("8g", f"{thinker} on responses, effort max", sample.error or "accepted")

    # -- latency ----------------------------------------------------------------------------

    def run_latency(self, light: str, thinker: str) -> None:
        self._plain(light, "none", 300)
        self._plain(thinker, "low", 2000)
        for i in range(self.repeats):
            loops = (self._chat_loop, self._responses_loop)
            for loop in loops if i % 2 == 0 else reversed(loops):
                loop(light, "none", 400)
        for model in (light, thinker):
            for _ in range(self.repeats):
                self._responses_loop(model, "low", 2500, scenario="tools+reasoning")
        self._cache(light)

    def _plain(self, model: str, effort: str, max_tokens: int) -> None:
        messages = [{"role": "user", "content": PLAIN}]
        for i in range(self.repeats):
            calls = (self.chat, self.responses)
            for stream in (False, True):
                for call in calls if i % 2 == 0 else reversed(calls):
                    call(
                        "plain",
                        model,
                        messages,
                        effort=effort,
                        max_tokens=max_tokens,
                        stream=stream,
                    )

    def _chat_loop(self, model: str, effort: str, max_tokens: int) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": TWO_CITIES}]
        tools = [chat_tool(WEATHER)]
        _, first = self.chat(
            "tools/turn 1", model, messages, effort=effort, max_tokens=max_tokens, tools=tools
        )
        message = first.choices[0].message if first is not None else None
        if message is None or not message.tool_calls:
            return
        calls = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in message.tool_calls
        ]
        messages.append({"role": "assistant", "content": message.content, "tool_calls": calls})
        messages += [
            {"role": "tool", "tool_call_id": c.id, "content": weather(c.function.arguments)}
            for c in message.tool_calls
        ]
        self.chat(
            "tools/turn 2", model, messages, effort=effort, max_tokens=max_tokens, tools=tools
        )

    def _responses_loop(
        self, model: str, effort: str, max_tokens: int, *, scenario: str = "tools"
    ) -> None:
        items: list[dict[str, Any]] = [{"role": "user", "content": TWO_CITIES}]
        tools = [responses_tool(WEATHER)]
        _, first = self.responses(
            f"{scenario}/turn 1", model, items, effort=effort, max_tokens=max_tokens, tools=tools
        )
        calls = [item for item in first.output if item.type == "function_call"] if first else []
        if not calls:
            return
        items += [replayable(item) for item in first.output]
        items += [
            {"type": "function_call_output", "call_id": c.call_id, "output": weather(c.arguments)}
            for c in calls
        ]
        self.responses(
            f"{scenario}/turn 2", model, items, effort=effort, max_tokens=max_tokens, tools=tools
        )

    def _cache(self, model: str) -> None:
        text = filler(9000)
        for call, name in ((self.chat, "chat"), (self.responses, "responses")):
            # A prefix per endpoint, so neither reads the other's cache.
            prefix = f"Session {name}-{self.run_id}.\n{text}"
            for i in range(self.repeats):
                messages = [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": f"What does rule {i + 3} say? One line."},
                ]
                call(
                    "cache",
                    model,
                    messages,
                    effort="none",
                    max_tokens=80,
                    prompt_cache_key=f"o01-{name}-{self.run_id}",
                )


# -- report -----------------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def latency_rows(samples: Iterable[Sample]) -> list[str]:
    """One table row per scenario, model, effort, endpoint and stream mode; checks left out."""
    groups: dict[tuple[str, str, str | None, str, bool], list[Sample]] = {}
    for sample in samples:
        if not sample.scenario.startswith("check"):
            key = (sample.scenario, sample.model, sample.effort, sample.endpoint, sample.stream)
            groups.setdefault(key, []).append(sample)
    rows: list[str] = []
    for (scenario, model, effort, endpoint, stream), group in groups.items():
        ok = [s for s in group if s.ok]
        totals = [s.total_s for s in ok]
        firsts = [s.ttft_s for s in ok if s.ttft_s is not None]
        server = [s.server_ms for s in ok if s.server_ms is not None]
        cells = [
            scenario,
            model,
            effort or "default",
            endpoint,
            "yes" if stream else "no",
            f"{len(ok)}/{len(group)}",
            _fmt(percentile(totals, 0.5)),
            _fmt(percentile(totals, 0.95)),
            _fmt(percentile(firsts, 0.5)),
            _fmt(percentile(server, 0.5), 0),
            _fmt(mean(s.input_tokens for s in ok), 0),
            _fmt(mean(s.cached_tokens for s in ok), 0),
            _fmt(mean(s.output_tokens for s in ok), 0),
            _fmt(mean(s.reasoning_tokens for s in ok), 0),
            f"${sum(s.cost for s in group):.4f}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def render(prober: Prober, *, started: str, models: dict[str, str]) -> str:
    budget = prober.budget
    chosen = ", ".join(f"{role} `{model}`" for role, model in models.items())
    lines = [
        "# O01 · Responses against Chat Completions (live)",
        "",
        f"- Started {started}; run `{prober.run_id}`; {prober.repeats} repeats per scenario.",
        f"- Models: {chosen}.",
        f"- Spent ${budget.spent:.4f} of ${budget.cap:.2f}; requests skipped by the cap: "
        f"{budget.skipped}.",
        f"- `phase` on Responses messages: {', '.join(sorted(prober.phases)) or 'never set'}.",
        "- Fields set on function_call items besides the usual ones: "
        f"{', '.join(sorted(prober.call_fields)) or 'none'}.",
        "",
        "## Checks",
        "",
        "| # | Question | Outcome |",
        "|---|---|---|",
        *(f"| {c.number} | {_cell(c.question)} | {_cell(c.outcome)} |" for c in prober.checks),
        "",
        "## Latency, tokens and cost",
        "",
        "Seconds; `server` is the `openai-processing-ms` header; tokens are means per request.",
        "",
        "| Scenario | Model | Effort | Endpoint | Stream | OK/N | p50 | p95 | p50 first token "
        "| p50 server ms | Input | Cached | Output | Reasoning | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *latency_rows(prober.samples),
    ]
    errors = sorted(
        {
            (s.scenario, s.endpoint, s.model, s.error)
            for s in prober.samples
            if s.error and not s.scenario.startswith("check")
        }
    )
    if errors:
        lines += ["", "## Errors outside the checks", ""]
        lines += [
            f"- {scenario} · {endpoint} · {model}: {error}"
            for scenario, endpoint, model, error in errors
        ]
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--light", default="gpt-6-luna", help="a model that takes effort none")
    parser.add_argument("--reasoning", default="gpt-6.1-sol", help="a model that always reasons")
    parser.add_argument("--other-family", default="gpt-5.5", help="a model of another family")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--max-cost", type=float, default=3.0, help="USD cap for the whole run")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.repeats < 1 or args.max_cost <= 0:
        raise SystemExit("--repeats must be >= 1 and --max-cost > 0")
    models = {"light": args.light, "reasoning": args.reasoning, "other family": args.other_family}
    for model in models.values():
        if pricing.get(model) is None:
            raise SystemExit(f"{model} has no price in the package's table")
    client = openai.OpenAI(max_retries=0, timeout=args.timeout_seconds)
    prober = Prober(client, Budget(args.max_cost), repeats=args.repeats)
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if not args.skip_checks:
            prober.run_checks(args.light, args.reasoning, args.other_family)
        if not args.skip_latency:
            prober.run_latency(args.light, args.reasoning)
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"openai-responses-{started.replace('-', '').replace(':', '')}.md"
        path.write_text(render(prober, started=started, models=models))
        data = {
            "samples": [asdict(s) for s in prober.samples],
            "checks": [asdict(c) for c in prober.checks],
        }
        path.with_suffix(".json").write_text(json.dumps(data, indent=2))
        print(f"report: {path}  spent: ${prober.budget.spent:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
