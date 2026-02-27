"""Gemini provider — thin adapter over the ``google-genai`` SDK."""

from __future__ import annotations

import base64
import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

from ai_arch_toolkit.core._content import DocumentPart, ImagePart, _is_url
from ai_arch_toolkit.core._exceptions import APIError, RateLimitError
from ai_arch_toolkit.core._pricing import _estimate_response_cost
from ai_arch_toolkit.core._providers._base import (
    DEFAULT_THINKING_BUDGET,
    THINKING_EFFORT_BUDGETS,
    BaseProvider,
    StreamState,
    _parse_retry_after,
)
from ai_arch_toolkit.core._providers._imports import require_sdk
from ai_arch_toolkit.core._response import (
    Citation,
    OutputSchema,
    Response,
    ThinkingBlock,
    ToolCall,
    Usage,
)

require_sdk("google.genai", "gemini")
from google import genai  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402

logger = logging.getLogger(__name__)

# Parameters safe to forward directly to the SDK config.
_SDK_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "stop_sequences",
    "seed",
    "presence_penalty",
    "frequency_penalty",
}


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _content_parts_to_gemini(content: Any) -> list[types.Part]:
    """Convert multimodal content to Gemini Part objects."""
    if isinstance(content, str):
        return [types.Part(text=content)]
    if not isinstance(content, list):
        return [types.Part(text=str(content))]

    parts: list[types.Part] = []
    for part in content:
        if isinstance(part, str):
            parts.append(types.Part(text=part))
        elif isinstance(part, ImagePart):
            if _is_url(part.source):
                parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=part.source,
                            mime_type=part.media_type,
                        )
                    )
                )
            else:
                data = (
                    part.source
                    if isinstance(part.source, bytes)
                    else base64.b64decode(part.source)
                )
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            data=data,
                            mime_type=part.media_type,
                        )
                    )
                )
        elif isinstance(part, DocumentPart):
            data = part.source if isinstance(part.source, bytes) else base64.b64decode(part.source)
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        data=data,
                        mime_type=part.media_type,
                    )
                )
            )
        else:
            parts.append(types.Part(text=str(part)))
    return parts


def _messages_to_sdk(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[types.Content]]:
    """Extract system messages and convert the rest to SDK Content objects.

    Tool results are identified by ``tool_use_id`` (role is ignored) and batched
    into ``user`` Content with ``function_response`` Parts, matching Gemini's
    expected format.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []
    pending_fn_responses: list[types.Part] = []

    def _flush_fn_responses() -> None:
        if pending_fn_responses:
            contents.append(types.Content(role="user", parts=list(pending_fn_responses)))
            pending_fn_responses.clear()

    for msg in messages:
        role = msg.get("role", "user")

        # Tool result → FunctionResponse part, batched into user Content
        if msg.get("tool_use_id"):
            _flush_fn_responses()
            raw_content = msg.get("content", "")
            try:
                response_data = (
                    json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                )
            except (json.JSONDecodeError, TypeError):
                response_data = {"result": raw_content}
            fn_name = msg.get("name", "")
            if not fn_name:
                warnings.warn(
                    "Gemini requires 'name' in tool results. Pass name= to tool_result().",
                    stacklevel=3,
                )
            pending_fn_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_name,
                        response=response_data,
                    )
                )
            )
            continue
        if role == "system":
            system_parts.append(msg.get("content", ""))
            continue

        # Assistant with tool_calls → model Content with FunctionCall parts
        if role == "assistant" and msg.get("tool_calls"):
            _flush_fn_responses()
            parts: list[types.Part] = []
            text = msg.get("content", "")
            if text:
                parts.append(types.Part(text=text))
            for tc in msg["tool_calls"]:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tc.get("name", ""),
                            args=tc.get("input", {}),
                        )
                    )
                )
            contents.append(types.Content(role="model", parts=parts))
            continue

        # Regular message
        _flush_fn_responses()
        gemini_role = "model" if role == "assistant" else role
        raw_content = msg.get("content", "")
        parts_list = _content_parts_to_gemini(raw_content)
        contents.append(types.Content(role=gemini_role, parts=parts_list))

    _flush_fn_responses()
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, contents


def _tool_to_sdk(tool: dict[str, Any]) -> types.FunctionDeclaration:
    """Map generic tool dict to Gemini FunctionDeclaration."""
    schema = tool.get("input_schema", tool.get("parameters", {}))
    return types.FunctionDeclaration(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=schema,
    )


def _build_thinking_config(
    thinking: bool,
    thinking_effort: str | None,
    thinking_budget: int | None,
    model: str = "",
) -> types.ThinkingConfig | None:
    """Build SDK ThinkingConfig, or None if disabled."""
    if not thinking:
        return None
    cfg: dict[str, Any] = {"include_thoughts": True}
    if model.startswith("gemini-3"):
        # Gemini 3: use thinking_level string
        cfg["thinking_level"] = thinking_effort or "high"
    else:
        # Gemini 2.5: use thinking_budget tokens
        if thinking_budget:
            cfg["thinking_budget"] = thinking_budget
        elif thinking_effort:
            cfg["thinking_budget"] = THINKING_EFFORT_BUDGETS.get(
                thinking_effort, DEFAULT_THINKING_BUDGET
            )
        else:
            cfg["thinking_budget"] = DEFAULT_THINKING_BUDGET
    return types.ThinkingConfig(**cfg)


def _extract_usage(usage_meta: Any) -> Usage:
    """Convert SDK usage metadata to our Usage dataclass."""
    return Usage(
        input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        cache_read_tokens=getattr(usage_meta, "cached_content_token_count", 0) or 0,
    )


def _parse_sdk_response(
    response: Any,
    model: str,
    *,
    output_schema: OutputSchema | None = None,
) -> Response:
    """Convert ``GenerateContentResponse`` to our ``Response``."""
    candidates = response.candidates or []
    if not candidates:
        return Response(raw=response, model=model)

    candidate = candidates[0]
    parts = candidate.content.parts if candidate.content else []

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    thinking_blocks: list[ThinkingBlock] = []

    for part in parts:
        if getattr(part, "thought", False) and part.text:
            thinking_blocks.append(ThinkingBlock(text=part.text))
        elif part.text is not None and not getattr(part, "thought", False):
            text_parts.append(part.text)
        elif part.function_call:
            fc = part.function_call
            tool_calls.append(
                ToolCall(
                    id=getattr(fc, "id", "") or "",
                    name=fc.name,
                    input=dict(fc.args) if fc.args else {},
                )
            )

    text = "".join(text_parts).strip()
    parsed: Any = None
    if output_schema and text:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse structured output as JSON")

    usage = _extract_usage(response.usage_metadata) if response.usage_metadata else Usage()
    cost = _estimate_response_cost(model, usage)

    finish_reason = ""
    if candidate.finish_reason:
        finish_reason = str(candidate.finish_reason).replace("FinishReason.", "")

    # Extract citations from grounding metadata
    citations: list[Citation] = []
    grounding = getattr(candidate, "grounding_metadata", None)
    if grounding:
        for chunk in getattr(grounding, "grounding_chunks", []) or []:
            web = getattr(chunk, "web", None)
            if web:
                citations.append(
                    Citation(
                        text="",
                        url=getattr(web, "uri", ""),
                        title=getattr(web, "title", ""),
                    )
                )

    return Response(
        text=text,
        tool_calls=tuple(tool_calls),
        thinking=tuple(thinking_blocks),
        parsed=parsed,
        usage=usage,
        cost=cost,
        stop_reason=finish_reason,
        model=getattr(response, "model_version", None) or model,
        raw=response,
        response_id=getattr(response, "response_id", "") or "",
        citations=tuple(citations),
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GeminiProvider(BaseProvider):
    """Google Gemini provider via the official ``google-genai`` SDK."""

    def __init__(
        self,
        model: str,
        api_key: str,
    ) -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key)

    async def close(self) -> None:
        self._client.close()

    async def count_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Count tokens using Gemini's countTokens API."""
        msg_system, contents = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system
        cfg_kwargs: dict[str, Any] = {}
        if effective_system:
            cfg_kwargs["system_instruction"] = effective_system
        config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
        try:
            result = await self._client.aio.models.count_tokens(
                model=self._model, contents=contents, config=config
            )
            return result.total_tokens
        except genai_errors.ClientError as exc:
            raise APIError(exc.code, str(exc)) from exc
        except genai_errors.ServerError as exc:
            raise APIError(exc.code, str(exc)) from exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_config(
        self,
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` from our generic params."""
        # Extract special params
        thinking = kwargs.pop("thinking", False)
        thinking_effort = kwargs.pop("thinking_effort", None)
        thinking_budget = kwargs.pop("thinking_budget", None)
        output_schema: OutputSchema | None = kwargs.pop("output_schema", None)
        tool_choice: str | None = kwargs.pop("tool_choice", None)
        json_mode: bool = kwargs.pop("json_mode", False)
        kwargs.pop("logprobs", None)  # Not supported by Gemini

        # Translate max_tokens to Gemini's max_output_tokens
        if "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        # Warn about unknown params
        unknown = set(kwargs) - _SDK_PARAMS
        if unknown:
            warnings.warn(
                f"Unknown parameter(s) ignored for Gemini: {sorted(unknown)}. "
                f"Valid: {sorted(_SDK_PARAMS)}",
                stacklevel=4,
            )
        filtered = {k: v for k, v in kwargs.items() if k in _SDK_PARAMS}

        cfg_kwargs: dict[str, Any] = {**filtered}

        if system:
            cfg_kwargs["system_instruction"] = system

        # Thinking
        thinking_cfg = _build_thinking_config(
            thinking, thinking_effort, thinking_budget, self._model
        )
        if thinking_cfg:
            cfg_kwargs["thinking_config"] = thinking_cfg

        # Tools
        if tools:
            fn_tools = [t for t in tools if not t.get("_server_tool")]
            server_tools = [t for t in tools if t.get("_server_tool")]
            gemini_tools: list[Any] = []
            if fn_tools:
                gemini_tools.append(
                    types.Tool(function_declarations=[_tool_to_sdk(t) for t in fn_tools])
                )
            for st in server_tools:
                st_type = st["type"]
                if st_type == "web_search":
                    gemini_tools.append(types.Tool(google_search=types.GoogleSearch()))
                elif st_type == "code_execution":
                    gemini_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
            if gemini_tools:
                cfg_kwargs["tools"] = gemini_tools

        # tool_choice
        if tool_choice is not None:
            mode_map = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
            if tool_choice in mode_map:
                cfg_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=mode_map[tool_choice],
                    )
                )
            else:
                cfg_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[tool_choice],
                    )
                )

        # Structured output
        if output_schema:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_json_schema"] = output_schema.schema

        # json_mode
        if json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"

        return types.GenerateContentConfig(**cfg_kwargs)

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
        msg_system, contents = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system
        config = self._build_config(system=effective_system, tools=tools, **kwargs)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as exc:
            if exc.code == 429:
                retry_after = None
                if exc.response and hasattr(exc.response, "headers"):
                    retry_after = exc.response.headers.get("retry-after")
                raise RateLimitError(
                    429,
                    str(exc),
                    retry_after=_parse_retry_after(retry_after),
                ) from exc
            raise APIError(exc.code, str(exc)) from exc
        except genai_errors.ServerError as exc:
            raise APIError(exc.code, str(exc)) from exc

        return _parse_sdk_response(response, self._model, output_schema=output_schema)

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], StreamState]:
        msg_system, contents = _messages_to_sdk(messages)
        effective_system = system if system is not None else msg_system
        config = self._build_config(system=effective_system, tools=tools, **kwargs)

        state = StreamState()
        state.model = self._model

        async def _generate() -> AsyncIterator[str]:
            try:
                stream = await self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                async for chunk in stream:
                    # Usage metadata (may appear on any/last chunk)
                    if chunk.usage_metadata:
                        state.usage = _extract_usage(chunk.usage_metadata)

                    candidates = chunk.candidates or []
                    if not candidates:
                        continue

                    candidate = candidates[0]
                    parts = candidate.content.parts if candidate.content else []

                    for part in parts:
                        if getattr(part, "thought", False) and part.text:
                            state.thinking.append(ThinkingBlock(text=part.text))
                        elif part.text is not None and not getattr(part, "thought", False):
                            yield part.text
                        elif part.function_call:
                            fc = part.function_call
                            state.tool_calls.append(
                                ToolCall(
                                    id=getattr(fc, "id", "") or "",
                                    name=fc.name,
                                    input=dict(fc.args) if fc.args else {},
                                )
                            )

                    if candidate.finish_reason:
                        state.stop_reason = str(candidate.finish_reason).replace(
                            "FinishReason.", ""
                        )

                    state.raw = chunk
                    model_ver = getattr(chunk, "model_version", None)
                    if model_ver:
                        state.model = model_ver

            except genai_errors.ClientError as exc:
                if exc.code == 429:
                    retry_after = None
                    if exc.response and hasattr(exc.response, "headers"):
                        retry_after = exc.response.headers.get("retry-after")
                    raise RateLimitError(
                        429,
                        str(exc),
                        retry_after=_parse_retry_after(retry_after),
                    ) from exc
                raise APIError(exc.code, str(exc)) from exc
            except genai_errors.ServerError as exc:
                raise APIError(exc.code, str(exc)) from exc

        return _generate(), state
