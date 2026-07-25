"""OpenAI provider — thin adapter over the ``openai`` SDK."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.core._content import DocumentPart, ImagePart, _encode_b64, _is_url
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import (
    BaseProvider,
    StreamState,
    _parse_retry_after,
    parse_tool_args,
)
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import (
    OutputSchema,
    Response,
    StreamEvent,
    ThinkingBlock,
    ToolCall,
    Usage,
    _uncached_input_tokens,
)

require_sdk("openai", "openai")
import openai  # noqa: E402

logger = logging.getLogger(__name__)

# Parameters safe to forward directly to the SDK.
_SDK_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "response_format",
    "parallel_tool_calls",
}

_MAX_COMPLETION_TOKEN_MODELS = {"o1", "o3", "o4"}
_MAX_COMPLETION_TOKEN_PREFIXES = ("gpt-5", "o1-", "o3-", "o4-")


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _content_to_sdk(content: Any) -> list[dict[str, Any]] | str:
    """Convert multimodal content to OpenAI content blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, ImagePart):
            if _is_url(part.source):
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part.source},
                    }
                )
            else:
                b64 = _encode_b64(part.source)
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.media_type};base64,{b64}"},
                    }
                )
        elif isinstance(part, DocumentPart):
            b64 = _encode_b64(part.source)
            blocks.append(
                {
                    "type": "file",
                    "file": {
                        "filename": part.name or "document",
                        "file_data": f"data:{part.media_type};base64,{b64}",
                    },
                }
            )
        else:
            blocks.append({"type": "text", "text": str(part)})
    return blocks


def _tool_to_sdk(tool: dict[str, Any]) -> dict[str, Any]:
    """Map generic tool dict to OpenAI SDK format (function wrapper)."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", tool.get("parameters", {})),
        },
    }


def _uses_max_completion_tokens(model: str) -> bool:
    """Return True for Chat Completions models that reject ``max_tokens``."""
    return model in _MAX_COMPLETION_TOKEN_MODELS or model.startswith(
        _MAX_COMPLETION_TOKEN_PREFIXES
    )


def _drop_temperature_for_reasoning(
    model: str, reasoning_effort: str | None, kwargs: dict[str, Any]
) -> None:
    """Drop sampling temperature for reasoning models unless only default sampling is used."""
    if not _uses_max_completion_tokens(model):
        return
    if reasoning_effort and reasoning_effort.lower() == "none":
        return
    if kwargs.get("temperature") not in (None, 1, 1.0):
        kwargs.pop("temperature", None)


def _messages_to_sdk(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Convert generic messages to OpenAI SDK format.

    ``tool_use_id`` is the tool-result discriminator (role is ignored).
    System stays as a regular message role. tool results use role="tool".
    If ``system`` is provided, system messages in the list are discarded.
    """
    wire: list[dict[str, Any]] = []
    if system is not None:
        wire.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        if msg.get("tool_use_id"):
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_use_id"],
                    "content": str(msg.get("content", "")),
                }
            )
        elif role == "system":
            if system is None:
                wire.append({"role": "system", "content": msg.get("content", "")})
        elif role == "assistant" and msg.get("tool_calls"):
            wire.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("input", {})),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                }
            )
        else:
            wire.append({"role": role, "content": _content_to_sdk(msg.get("content", ""))})
    return wire


def _build_output_schema_format(
    output_schema: OutputSchema,
) -> dict[str, Any]:
    """Build OpenAI ``response_format`` for structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_schema.name,
            "schema": output_schema.schema,
            "strict": output_schema.strict,
        },
    }


def _extract_usage(sdk_usage: Any) -> Usage:
    """Convert SDK usage object to our Usage dataclass."""
    cache_read = 0
    details = getattr(sdk_usage, "prompt_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    total_input = getattr(sdk_usage, "prompt_tokens", 0)
    return Usage(
        input_tokens=_uncached_input_tokens(total_input, cache_read),
        output_tokens=getattr(sdk_usage, "completion_tokens", 0),
        cache_read_tokens=cache_read,
    )


def _reasoning_text(obj: Any) -> str:
    """Extract vendor reasoning text from a streaming delta, message, or dict.

    Reads ``reasoning_content`` (DeepSeek, vLLM, LM Studio, SGLang) then
    ``reasoning`` (Ollama /v1, OpenRouter), per-field so a non-string value in
    one does not mask a valid string in the other. SDK pydantic models use
    ``extra="allow"``, so unknown wire fields surface as attributes; batch
    bodies arrive as plain dicts.
    """
    for attr in ("reasoning_content", "reasoning"):
        value = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _parse_sdk_response(
    completion: Any,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert ``openai.types.chat.ChatCompletion`` to our ``Response``."""
    choices = completion.choices or []
    if not choices:
        return Response(raw=completion, model=model)

    choice = choices[0]
    message = choice.message
    text = message.content or ""

    thinking: tuple[ThinkingBlock, ...] = ()
    if reasoning := _reasoning_text(message):
        thinking = (ThinkingBlock(text=reasoning),)

    tool_calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        tool_calls.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                input=parse_tool_args(tc.function.arguments),
            )
        )

    parsed: Any = None
    if output_schema and text:
        try:
            data = json.loads(text)
            if output_schema.model_class is not None:
                parsed = output_schema.model_class.model_validate(data)
            else:
                parsed = data
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse structured output as JSON")
        except Exception:
            logger.warning("Failed to validate structured output against schema")
            parsed = json.loads(text)  # fallback to raw dict

    usage = _extract_usage(completion.usage) if completion.usage else Usage()
    cost = _estimate_response_cost(model, usage)

    # Extract logprobs if present
    logprobs_data = getattr(choice, "logprobs", None)

    return Response(
        text=text.strip(),
        tool_calls=tuple(tool_calls),
        thinking=thinking,
        parsed=parsed,
        usage=usage,
        cost=cost,
        stop_reason=choice.finish_reason or "",
        model=completion.model or model,
        raw=completion,
        response_id=getattr(completion, "id", "") or "",
        logprobs=logprobs_data,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions API provider via the official SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model = model
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            import httpx

            client_kwargs["timeout"] = httpx.Timeout(timeout)
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def close(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_sdk_kwargs(
        self,
        wire_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build kwargs dict for ``chat.completions.create()``."""
        # Extract our special params
        thinking = kwargs.pop("thinking", False)
        thinking_effort = kwargs.pop("thinking_effort", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        output_schema: OutputSchema | None = kwargs.pop("output_schema", None)
        tool_choice: str | None = kwargs.pop("tool_choice", None)
        json_mode: bool = kwargs.pop("json_mode", False)
        logprobs_flag: bool = kwargs.pop("logprobs", False)
        kwargs.pop("structured_output_mode", None)  # Anthropic-specific; ignored here

        # Warn about unknown params
        unknown = set(kwargs) - _SDK_PARAMS
        if unknown:
            warnings.warn(
                f"Unknown parameter(s) ignored for OpenAI: {sorted(unknown)}. "
                f"Valid: {sorted(_SDK_PARAMS)}",
                stacklevel=4,
            )

        if _uses_max_completion_tokens(self._model) and "max_tokens" in kwargs:
            if "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            kwargs.pop("max_tokens", None)

        reasoning_effort = (thinking_effort or "high") if thinking else None
        if thinking:
            _drop_temperature_for_reasoning(self._model, reasoning_effort, kwargs)

        filtered = {k: v for k, v in kwargs.items() if k in _SDK_PARAMS}

        sdk_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": wire_messages,
            **filtered,
        }

        # Thinking → reasoning_effort (OpenAI naming)
        if thinking:
            sdk_kwargs["reasoning_effort"] = reasoning_effort
        if thinking_budget:
            warnings.warn(
                "thinking_budget is not supported by OpenAI (only reasoning_effort string), "
                "ignoring",
                stacklevel=4,
            )

        # Structured output via native response_format
        if output_schema:
            sdk_kwargs["response_format"] = _build_output_schema_format(output_schema)

        if tools:
            fn_tools = [_tool_to_sdk(t) for t in tools if not t.get("_server_tool")]
            server_tools = [t for t in tools if t.get("_server_tool")]
            if fn_tools:
                sdk_kwargs["tools"] = fn_tools
            for st in server_tools:
                st_type = st["type"]
                if st_type == "web_search":
                    sdk_kwargs.setdefault("tools", []).append({"type": "web_search"})
                elif st_type == "code_execution":
                    sdk_kwargs.setdefault("tools", []).append({"type": "code_interpreter"})

        # tool_choice
        if tool_choice is not None:
            if tool_choice in ("auto", "required", "none"):
                sdk_kwargs["tool_choice"] = tool_choice
            else:
                sdk_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        # json_mode
        if json_mode:
            sdk_kwargs["response_format"] = {"type": "json_object"}

        # logprobs
        if logprobs_flag:
            sdk_kwargs["logprobs"] = True

        return sdk_kwargs

    # ------------------------------------------------------------------
    # complete
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Response:
        output_schema: OutputSchema | None = kwargs.get("output_schema")
        wire = _messages_to_sdk(messages, system=system)
        sdk_kwargs = self._build_sdk_kwargs(wire, tools=tools, **kwargs)

        logger.debug("complete start model=%s messages=%d", self._model, len(messages))
        try:
            completion = await self._client.chat.completions.create(**sdk_kwargs)
        except openai.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response else None
            raise RateLimitError(
                exc.response.status_code if exc.response else 429,
                str(exc.body),
                retry_after=_parse_retry_after(retry_after),
            ) from exc
        except openai.APIStatusError as exc:
            raise APIError(
                exc.response.status_code if exc.response else 500,
                str(exc.body),
            ) from exc

        resp = _parse_sdk_response(completion, self._model, output_schema=output_schema)
        logger.debug(
            "complete done model=%s tokens_in=%d tokens_out=%d",
            self._model,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
        )
        return resp

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    def _stream_core(
        self,
        sdk_kwargs: dict[str, Any],
        state: StreamState,
    ) -> AsyncIterator[StreamEvent]:
        """Single SDK chunk loop shared by ``stream()`` and ``stream_events()``.

        Each reasoning delta is emitted as a ``partial`` thinking event in real
        time; ``state.thinking`` holds one block that is kept up to date after
        every fragment, so an early-abandoned stream still finalizes with the
        reasoning received so far.
        """

        async def _flush_tool_calls(
            tc_acc: dict[int, dict[str, str]],
        ) -> AsyncIterator[StreamEvent]:
            for _idx in sorted(tc_acc):
                acc = tc_acc[_idx]
                tool_call = ToolCall(
                    id=acc["id"],
                    name=acc["name"],
                    input=parse_tool_args(acc["arguments"]),
                )
                state.tool_calls.append(tool_call)
                yield StreamEvent(kind="tool_call", tool_call=tool_call)
            tc_acc.clear()

        async def _generate() -> AsyncIterator[StreamEvent]:
            tc_acc: dict[int, dict[str, str]] = {}
            reasoning_acc = ""

            try:
                stream = await self._client.chat.completions.create(
                    **sdk_kwargs,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    # Usage chunk (final)
                    if chunk.usage:
                        state.usage = _extract_usage(chunk.usage)

                    choices = chunk.choices or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.delta

                    # Vendor reasoning deltas (local OpenAI-compatible servers).
                    # Keep state.thinking's single block current per fragment so
                    # early-abandoned streams still finalize with the reasoning.
                    if delta and (fragment := _reasoning_text(delta)):
                        first = not reasoning_acc
                        reasoning_acc += fragment
                        block = ThinkingBlock(text=reasoning_acc)
                        if first:
                            state.thinking.append(block)
                        else:
                            state.thinking[-1] = block
                        yield StreamEvent(
                            kind="thinking", thinking=ThinkingBlock(text=fragment), partial=True
                        )

                    # Text content
                    if delta and delta.content:
                        yield StreamEvent(kind="text", text=delta.content)

                    # Accumulate tool call deltas
                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tc_acc:
                                tc_acc[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": (
                                        tc_delta.function.name
                                        if tc_delta.function and tc_delta.function.name
                                        else ""
                                    ),
                                    "arguments": "",
                                }
                            else:
                                if tc_delta.id:
                                    tc_acc[idx]["id"] = tc_delta.id
                                if tc_delta.function and tc_delta.function.name:
                                    tc_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function and tc_delta.function.arguments:
                                tc_acc[idx]["arguments"] += tc_delta.function.arguments

                    finish = choice.finish_reason
                    if finish == "tool_calls" and tc_acc:
                        async for event in _flush_tool_calls(tc_acc):
                            yield event

                    if finish:
                        state.stop_reason = finish

                    if chunk.model:
                        state.model = chunk.model

                    state.raw = chunk

                # Some OpenAI-compatible servers end tool-call turns with
                # finish_reason "stop" — flush whatever was accumulated.
                if tc_acc:
                    async for event in _flush_tool_calls(tc_acc):
                        yield event

            except openai.RateLimitError as exc:
                retry_after = exc.response.headers.get("retry-after") if exc.response else None
                raise RateLimitError(
                    exc.response.status_code if exc.response else 429,
                    str(exc.body),
                    retry_after=_parse_retry_after(retry_after),
                ) from exc
            except openai.APIStatusError as exc:
                raise APIError(
                    exc.response.status_code if exc.response else 500,
                    str(exc.body),
                ) from exc

        return _generate()

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], StreamState]:
        wire = _messages_to_sdk(messages, system=system)
        sdk_kwargs = self._build_sdk_kwargs(wire, tools=tools, **kwargs)

        logger.debug("stream start model=%s", self._model)
        state = StreamState()
        state.model = self._model
        events = self._stream_core(sdk_kwargs, state)

        async def _text_only() -> AsyncIterator[str]:
            async for event in events:
                if event.kind == "text" and event.text:
                    yield event.text

        return _text_only(), state

    def stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[StreamEvent], StreamState]:
        wire = _messages_to_sdk(messages, system=system)
        sdk_kwargs = self._build_sdk_kwargs(wire, tools=tools, **kwargs)

        logger.debug("stream_events start model=%s", self._model)
        state = StreamState()
        state.model = self._model
        return self._stream_core(sdk_kwargs, state), state

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    async def batch_submit(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Submit a batch via OpenAI's Batch API."""
        import io

        lines: list[str] = []
        for req in requests:
            custom_id = req.get("custom_id", "")
            messages = req.get("messages", [])
            req_system = req.get("system")
            tools = req.get("tools")
            req_kwargs = req.get("kwargs", {})

            wire = _messages_to_sdk(messages, system=req_system)
            body: dict[str, Any] = {
                "model": self._model,
                "messages": wire,
                "max_tokens": req_kwargs.get("max_tokens", 4096),
            }
            if tools:
                fn_tools = [t for t in tools if not t.get("_server_tool")]
                if fn_tools:
                    body["tools"] = [_tool_to_sdk(t) for t in fn_tools]

            lines.append(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )

        content = "\n".join(lines)
        try:
            file = await self._client.files.create(
                file=io.BytesIO(content.encode()), purpose="batch"
            )
            batch = await self._client.batches.create(
                input_file_id=file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            return batch.id
        except openai.APIStatusError as exc:
            raise APIError(
                exc.response.status_code if exc.response else 500,
                str(exc.body),
            ) from exc

    async def batch_status(self, batch_id: str) -> str:
        """Check batch status."""
        try:
            batch = await self._client.batches.retrieve(batch_id)
            return batch.status
        except openai.APIStatusError as exc:
            raise APIError(
                exc.response.status_code if exc.response else 500,
                str(exc.body),
            ) from exc

    async def batch_results(self, batch_id: str) -> list[Any]:
        """Retrieve completed batch results."""
        from ai_arch_toolkit.core._batch import BatchResult

        try:
            batch = await self._client.batches.retrieve(batch_id)
            if not batch.output_file_id:
                return []
            file_response = await self._client.files.content(batch.output_file_id)
            raw_text = file_response.text
        except openai.APIStatusError as exc:
            raise APIError(
                exc.response.status_code if exc.response else 500,
                str(exc.body),
            ) from exc

        results: list[BatchResult] = []
        for line in raw_text.strip().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            resp_body = entry.get("response", {}).get("body")
            error = entry.get("error")
            if error:
                results.append(BatchResult(custom_id=custom_id, error=str(error)))
            elif resp_body:
                response = self._parse_batch_response(resp_body)
                results.append(BatchResult(custom_id=custom_id, response=response))
            else:
                results.append(BatchResult(custom_id=custom_id, error="empty response"))
        return results

    def _parse_batch_response(self, body: dict[str, Any]) -> Response:
        """Build a Response from a raw batch response body dict."""
        choices = body.get("choices", [])
        if not choices:
            return Response(raw=body, model=self._model)

        choice = choices[0]
        message = choice.get("message", {})
        text = (message.get("content") or "").strip()

        thinking: tuple[ThinkingBlock, ...] = ()
        if reasoning := _reasoning_text(message):
            thinking = (ThinkingBlock(text=reasoning),)

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    input=parse_tool_args(fn.get("arguments", "{}")),
                )
            )

        raw_usage = body.get("usage", {})
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        cache_read = prompt_details.get("cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=_uncached_input_tokens(raw_usage.get("prompt_tokens", 0), cache_read),
            output_tokens=raw_usage.get("completion_tokens", 0),
            cache_read_tokens=cache_read,
        )
        cost = _estimate_response_cost(self._model, usage)

        return Response(
            text=text,
            tool_calls=tuple(tool_calls),
            thinking=thinking,
            usage=usage,
            cost=cost,
            stop_reason=choice.get("finish_reason", ""),
            model=body.get("model", self._model),
            raw=body,
        )
