#!/usr/bin/env python3
"""Execute one example file with runtime call tracing enabled.

This script monkeypatches ai-arch-toolkit client classes at runtime so that
example output includes explicit call boundaries, request summaries, and
result summaries. It is intended to be called by ``scripts/run_examples.py``.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import runpy
import time
from itertools import count
from pathlib import Path
from typing import Any

CALL_ID_GEN = count(1)
TOTAL_CALLS = 0
COMPLETED_CALLS = 0
FAILED_CALLS = 0


def _call_id() -> str:
    return f"{next(CALL_ID_GEN):03d}"


def _register_call_start() -> None:
    global TOTAL_CALLS
    TOTAL_CALLS += 1


def _register_call_end() -> None:
    global COMPLETED_CALLS
    COMPLETED_CALLS += 1


def _register_call_fail() -> None:
    global FAILED_CALLS
    FAILED_CALLS += 1


def _message_summary(prompt_or_messages: Any) -> str:
    if isinstance(prompt_or_messages, str):
        return f"messages=1(prompt str, chars={len(prompt_or_messages)})"
    if isinstance(prompt_or_messages, (list, tuple)):
        return f"messages={len(prompt_or_messages)}(sequence)"
    return f"messages=unknown(type={type(prompt_or_messages).__name__})"


def _result_summary(result: Any) -> str:
    if hasattr(result, "text"):
        text = str(getattr(result, "text", ""))
        thinking = str(getattr(result, "thinking", "") or "")
        tool_calls = getattr(result, "tool_calls", ()) or ()
        stop_reason = getattr(result, "stop_reason", "")
        return (
            f"result=response(text_chars={len(text)}, "
            f"thinking_chars={len(thinking)}, "
            f"tool_calls={len(tool_calls)}, "
            f"stop_reason={stop_reason or 'n/a'})"
        )
    if isinstance(result, list):
        return f"result=list(len={len(result)})"
    return f"result={type(result).__name__}"


def _print_trace(message: str) -> None:
    print(f"[TRACE] {message}")


def _print_call_summary() -> None:
    _print_trace(
        "CALL_SUMMARY "
        f"total_calls={TOTAL_CALLS} "
        f"completed_calls={COMPLETED_CALLS} "
        f"failed_calls={FAILED_CALLS}"
    )


def _print_trace_block(call_id: str, label: str, text: str) -> None:
    normalized = text if text else "<empty>"
    _print_trace(f"[CALL {call_id}] {label}_BEGIN chars={len(text)}")
    print(normalized)
    _print_trace(f"[CALL {call_id}] {label}_END")


def _trace_response_payload(call_id: str, result: Any) -> None:
    if not hasattr(result, "text"):
        return
    text = str(getattr(result, "text", ""))
    thinking = str(getattr(result, "thinking", "") or "")
    thinking_blocks = getattr(result, "thinking_blocks", ()) or ()
    tool_calls = getattr(result, "tool_calls", ()) or ()

    _print_trace_block(call_id, "RESPONSE_TEXT", text)
    if thinking:
        _print_trace_block(call_id, "RESPONSE_THINKING", thinking)
    if thinking_blocks:
        _print_trace(f"[CALL {call_id}] RESPONSE_THINKING_BLOCKS count={len(thinking_blocks)}")
        for index, block in enumerate(thinking_blocks, start=1):
            block_text = str(getattr(block, "text", ""))
            _print_trace_block(call_id, f"RESPONSE_THINKING_BLOCK_{index}", block_text)
    if tool_calls:
        _print_trace(f"[CALL {call_id}] RESPONSE_TOOL_CALLS count={len(tool_calls)}")
        for index, tool_call in enumerate(tool_calls, start=1):
            tool_id = getattr(tool_call, "id", "")
            tool_name = getattr(tool_call, "name", "")
            arguments = getattr(tool_call, "arguments", {})
            _print_trace(
                f"[CALL {call_id}] RESPONSE_TOOL_CALL_{index} "
                f"id={tool_id} name={tool_name} arguments={arguments}"
            )


def _trace_batch_list_payload(call_id: str, result: Any) -> None:
    if not isinstance(result, list):
        return
    for index, item in enumerate(result, start=1):
        custom_id = str(getattr(item, "custom_id", f"item-{index}"))
        error = getattr(item, "error", None)
        response = getattr(item, "response", None)
        if error:
            _print_trace(f"[CALL {call_id}] BATCH_ITEM_{index} custom_id={custom_id} error={error}")
        if response is not None and hasattr(response, "text"):
            text = str(getattr(response, "text", ""))
            _print_trace(
                f"[CALL {call_id}] BATCH_ITEM_{index} custom_id={custom_id} "
                f"response_text_chars={len(text)}"
            )
            _print_trace_block(call_id, f"BATCH_ITEM_{index}_RESPONSE_TEXT", text)


def _wrap_sync_call(cls: type[Any], method_name: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return

    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_id = _call_id()
        _register_call_start()
        started = time.monotonic()
        provider = getattr(self, "_provider_name", getattr(self, "_provider", "n/a"))
        model = getattr(self, "_model", "n/a")
        msg = _message_summary(args[0]) if args else "messages=n/a"
        _print_trace(
            f"[CALL {call_id}] START {cls.__name__}.{method_name} "
            f"provider={provider} model={model} {msg}"
        )
        try:
            result = original(self, *args, **kwargs)
            duration = time.monotonic() - started
            _trace_response_payload(call_id, result)
            _trace_batch_list_payload(call_id, result)
            _register_call_end()
            _print_trace(
                f"[CALL {call_id}] END   {cls.__name__}.{method_name} "
                f"duration={duration:.3f}s {_result_summary(result)}"
            )
            return result
        except Exception as exc:
            duration = time.monotonic() - started
            _register_call_fail()
            _print_trace(
                f"[CALL {call_id}] FAIL  {cls.__name__}.{method_name} "
                f"duration={duration:.3f}s error={type(exc).__name__}: {exc}"
            )
            raise

    setattr(cls, method_name, wrapped)


def _wrap_sync_stream(cls: type[Any], method_name: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return

    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_id = _call_id()
        _register_call_start()
        started = time.monotonic()
        provider = getattr(self, "_provider_name", getattr(self, "_provider", "n/a"))
        model = getattr(self, "_model", "n/a")
        msg = _message_summary(args[0]) if args else "messages=n/a"
        _print_trace(
            f"[CALL {call_id}] START {cls.__name__}.{method_name} "
            f"provider={provider} model={model} {msg}"
        )
        stream = original(self, *args, **kwargs)

        def generator() -> Any:
            item_count = 0
            text_char_count = 0
            text_chunks: list[str] = []
            thinking_chunks: list[str] = []
            event_type_counts: dict[str, int] = {}
            try:
                for item in stream:
                    item_count += 1
                    if isinstance(item, str):
                        text_char_count += len(item)
                        text_chunks.append(item)
                    if method_name == "stream_events":
                        event_type = str(getattr(item, "type", "unknown"))
                        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                        text = str(getattr(item, "text", "") or "")
                        thinking = str(getattr(item, "thinking", "") or "")
                        if text:
                            text_chunks.append(text)
                            text_char_count += len(text)
                        if thinking:
                            thinking_chunks.append(thinking)
                    yield item
                duration = time.monotonic() - started
                if method_name == "stream":
                    _print_trace_block(call_id, "STREAM_TEXT", "".join(text_chunks))
                elif method_name == "stream_events":
                    if text_chunks:
                        _print_trace_block(call_id, "STREAM_EVENTS_TEXT", "".join(text_chunks))
                    if thinking_chunks:
                        _print_trace_block(call_id, "STREAM_EVENTS_THINKING", "".join(thinking_chunks))
                    _print_trace(f"[CALL {call_id}] STREAM_EVENTS_COUNTS {event_type_counts}")
                _register_call_end()
                _print_trace(
                    f"[CALL {call_id}] END   {cls.__name__}.{method_name} "
                    f"duration={duration:.3f}s stream_items={item_count} text_chars={text_char_count}"
                )
            except Exception as exc:
                duration = time.monotonic() - started
                if method_name == "stream" and text_chunks:
                    _print_trace_block(call_id, "STREAM_TEXT_PARTIAL", "".join(text_chunks))
                elif method_name == "stream_events":
                    if text_chunks:
                        _print_trace_block(call_id, "STREAM_EVENTS_TEXT_PARTIAL", "".join(text_chunks))
                    if thinking_chunks:
                        _print_trace_block(
                            call_id, "STREAM_EVENTS_THINKING_PARTIAL", "".join(thinking_chunks)
                        )
                _register_call_fail()
                _print_trace(
                    f"[CALL {call_id}] FAIL  {cls.__name__}.{method_name} "
                    f"duration={duration:.3f}s stream_items={item_count} "
                    f"error={type(exc).__name__}: {exc}"
                )
                raise

        return generator()

    setattr(cls, method_name, wrapped)


def _wrap_async_call(cls: type[Any], method_name: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None or not inspect.iscoroutinefunction(original):
        return

    @functools.wraps(original)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_id = _call_id()
        _register_call_start()
        started = time.monotonic()
        provider = getattr(self, "_provider_name", getattr(self, "_provider", "n/a"))
        model = getattr(self, "_model", "n/a")
        msg = _message_summary(args[0]) if args else "messages=n/a"
        _print_trace(
            f"[CALL {call_id}] START {cls.__name__}.{method_name} "
            f"provider={provider} model={model} {msg}"
        )
        try:
            result = await original(self, *args, **kwargs)
            duration = time.monotonic() - started
            _trace_response_payload(call_id, result)
            _trace_batch_list_payload(call_id, result)
            _register_call_end()
            _print_trace(
                f"[CALL {call_id}] END   {cls.__name__}.{method_name} "
                f"duration={duration:.3f}s {_result_summary(result)}"
            )
            return result
        except Exception as exc:
            duration = time.monotonic() - started
            _register_call_fail()
            _print_trace(
                f"[CALL {call_id}] FAIL  {cls.__name__}.{method_name} "
                f"duration={duration:.3f}s error={type(exc).__name__}: {exc}"
            )
            raise

    setattr(cls, method_name, wrapped)


def _wrap_async_stream(cls: type[Any], method_name: str) -> None:
    original = getattr(cls, method_name, None)
    if original is None:
        return

    @functools.wraps(original)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_id = _call_id()
        _register_call_start()
        started = time.monotonic()
        provider = getattr(self, "_provider_name", getattr(self, "_provider", "n/a"))
        model = getattr(self, "_model", "n/a")
        msg = _message_summary(args[0]) if args else "messages=n/a"
        _print_trace(
            f"[CALL {call_id}] START {cls.__name__}.{method_name} "
            f"provider={provider} model={model} {msg}"
        )
        stream = original(self, *args, **kwargs)
        item_count = 0
        text_char_count = 0
        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        event_type_counts: dict[str, int] = {}
        try:
            async for item in stream:
                item_count += 1
                if isinstance(item, str):
                    text_char_count += len(item)
                    text_chunks.append(item)
                if method_name == "stream_events":
                    event_type = str(getattr(item, "type", "unknown"))
                    event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                    text = str(getattr(item, "text", "") or "")
                    thinking = str(getattr(item, "thinking", "") or "")
                    if text:
                        text_chunks.append(text)
                        text_char_count += len(text)
                    if thinking:
                        thinking_chunks.append(thinking)
                yield item
            duration = time.monotonic() - started
            if method_name == "stream":
                _print_trace_block(call_id, "STREAM_TEXT", "".join(text_chunks))
            elif method_name == "stream_events":
                if text_chunks:
                    _print_trace_block(call_id, "STREAM_EVENTS_TEXT", "".join(text_chunks))
                if thinking_chunks:
                    _print_trace_block(call_id, "STREAM_EVENTS_THINKING", "".join(thinking_chunks))
                _print_trace(f"[CALL {call_id}] STREAM_EVENTS_COUNTS {event_type_counts}")
            _register_call_end()
            _print_trace(
                f"[CALL {call_id}] END   {cls.__name__}.{method_name} "
                f"duration={duration:.3f}s stream_items={item_count} text_chars={text_char_count}"
            )
        except Exception as exc:
            duration = time.monotonic() - started
            if method_name == "stream" and text_chunks:
                _print_trace_block(call_id, "STREAM_TEXT_PARTIAL", "".join(text_chunks))
            elif method_name == "stream_events":
                if text_chunks:
                    _print_trace_block(call_id, "STREAM_EVENTS_TEXT_PARTIAL", "".join(text_chunks))
                if thinking_chunks:
                    _print_trace_block(
                        call_id, "STREAM_EVENTS_THINKING_PARTIAL", "".join(thinking_chunks)
                    )
            _register_call_fail()
            _print_trace(
                f"[CALL {call_id}] FAIL  {cls.__name__}.{method_name} "
                f"duration={duration:.3f}s stream_items={item_count} "
                f"error={type(exc).__name__}: {exc}"
            )
            raise

    setattr(cls, method_name, wrapped)


def _install_tracing() -> None:
    from ai_arch_toolkit import AsyncBatchClient, AsyncClient, BatchClient, Client

    _wrap_sync_call(Client, "chat")
    _wrap_sync_stream(Client, "stream")
    _wrap_sync_stream(Client, "stream_events")

    _wrap_async_call(AsyncClient, "chat")
    _wrap_async_stream(AsyncClient, "stream")
    _wrap_async_stream(AsyncClient, "stream_events")

    _wrap_sync_call(BatchClient, "submit")
    _wrap_sync_call(BatchClient, "status")
    _wrap_sync_call(BatchClient, "results")

    _wrap_async_call(AsyncBatchClient, "submit")
    _wrap_async_call(AsyncBatchClient, "status")
    _wrap_async_call(AsyncBatchClient, "results")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one example with ai-arch call tracing enabled.")
    parser.add_argument("example", type=Path, help="Path to the example script to execute.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    example_path = args.example.resolve()
    if not example_path.exists():
        raise FileNotFoundError(f"Example not found: {example_path}")

    _print_trace("Runtime tracing enabled for ai-arch-toolkit clients.")
    _print_trace(f"Executing example: {example_path}")
    _install_tracing()
    try:
        runpy.run_path(str(example_path), run_name="__main__")
        _print_trace("Example execution completed.")
    finally:
        _print_call_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
