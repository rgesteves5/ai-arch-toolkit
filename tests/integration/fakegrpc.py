"""A loopback gRPC server that speaks xAI's Chat service, for the real ``xai-sdk`` in tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import grpc
import xai_sdk
from xai_sdk.proto import chat_pb2, chat_pb2_grpc, sample_pb2, usage_pb2

from ai_arch_toolkit.core._providers._xai import _CHANNEL_OPTIONS, XAIProvider

ASSISTANT = chat_pb2.MessageRole.ROLE_ASSISTANT
USAGE = usage_pb2.SamplingUsage(prompt_tokens=30, completion_tokens=12, reasoning_tokens=5)

type Call = tuple[str, str, str]  # id, name, JSON arguments


def _calls(calls: Sequence[Call]) -> list[chat_pb2.ToolCall]:
    return [
        chat_pb2.ToolCall(
            id=call_id,
            type=chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            function=chat_pb2.FunctionCall(name=name, arguments=arguments),
        )
        for call_id, name, arguments in calls
    ]


def answer(
    text: str = "ok",
    *,
    reasoning: str = "",
    calls: Sequence[Call] = (),
    usage: usage_pb2.SamplingUsage | None = USAGE,
    finish: int = sample_pb2.FinishReason.REASON_STOP,
) -> chat_pb2.GetChatCompletionResponse:
    """A whole ``GetCompletion`` answer."""
    message = chat_pb2.CompletionMessage(
        content=text, reasoning_content=reasoning, role=ASSISTANT, tool_calls=_calls(calls)
    )
    return chat_pb2.GetChatCompletionResponse(
        id="grok-1",
        model="grok-4.6",
        outputs=[chat_pb2.CompletionOutput(message=message, finish_reason=finish)],
        usage=usage,
    )


def chunk(
    text: str = "",
    *,
    reasoning: str = "",
    calls: Sequence[Call] = (),
    usage: usage_pb2.SamplingUsage | None = None,
    finish: int = sample_pb2.FinishReason.REASON_INVALID,
) -> chat_pb2.GetChatCompletionChunk:
    """One streamed chunk; every chunk carries the assistant role, as xAI's do."""
    delta = chat_pb2.Delta(
        content=text, reasoning_content=reasoning, role=ASSISTANT, tool_calls=_calls(calls)
    )
    return chat_pb2.GetChatCompletionChunk(
        id="grok-1",
        model="grok-4.6",
        outputs=[chat_pb2.CompletionOutputChunk(delta=delta, finish_reason=finish)],
        usage=usage,
    )


def _ok_chunks() -> list[chat_pb2.GetChatCompletionChunk]:
    return [
        chunk("o"),
        chunk("k", finish=sample_pb2.FinishReason.REASON_STOP, usage=USAGE),
    ]


@dataclass(slots=True, kw_only=True)
class Script:
    """What the server does with every call.

    ``GetCompletion`` answers ``response`` and ``GetCompletionChunk`` streams ``chunks``. With
    ``code`` the call aborts with that status instead, a stream after its first ``abort_after``
    chunks; with ``drop`` the server shuts down there (the connection is cut mid-stream).
    ``hang`` waits that many seconds before answering. ``requests`` records every request.
    """

    response: chat_pb2.GetChatCompletionResponse = field(default_factory=answer)
    chunks: list[chat_pb2.GetChatCompletionChunk] = field(default_factory=_ok_chunks)
    code: grpc.StatusCode | None = None
    drop: bool = False
    abort_after: int = 0
    hang: float = 0.0
    requests: list[chat_pb2.GetCompletionsRequest] = field(default_factory=list)


class _Chat(chat_pb2_grpc.ChatServicer):
    def __init__(self, script: Script, server: grpc.aio.Server) -> None:
        self.script = script
        self.server = server
        self.stopping: asyncio.Task[None] | None = None

    async def _arrive(self, request: chat_pb2.GetCompletionsRequest) -> None:
        self.script.requests.append(request)
        if self.script.hang:
            await asyncio.sleep(self.script.hang)

    async def GetCompletion(
        self, request: chat_pb2.GetCompletionsRequest, context: grpc.aio.ServicerContext
    ) -> chat_pb2.GetChatCompletionResponse:
        await self._arrive(request)
        if self.script.code is not None:
            await context.abort(self.script.code, "scripted failure")
        return self.script.response

    async def GetCompletionChunk(
        self, request: chat_pb2.GetCompletionsRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[chat_pb2.GetChatCompletionChunk]:
        await self._arrive(request)
        cut = self.script.code is not None or self.script.drop
        for piece in self.script.chunks[: self.script.abort_after] if cut else self.script.chunks:
            yield piece
        if self.script.drop:
            # From outside this call, which the shutdown cancels.
            self.stopping = asyncio.create_task(self.server.stop(None))
            await asyncio.sleep(5)
        if self.script.code is not None:
            await context.abort(self.script.code, "scripted failure")


@asynccontextmanager
async def serving(script: Script | None = None) -> AsyncIterator[tuple[Script, int]]:
    """Serve ``script`` on an ephemeral loopback port (plaintext)."""
    script = script or Script()
    server = grpc.aio.server()
    chat = _Chat(script, server)
    chat_pb2_grpc.add_ChatServicer_to_server(chat, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield script, port
    finally:
        await (chat.stopping or server.stop(None))


async def provider(model: str, port: int, *, timeout: float = 2.0) -> XAIProvider:
    """An adapter whose SDK client, configured as the adapter's own, talks to 127.0.0.1."""
    adapter = XAIProvider(model, "local-test", timeout=timeout)
    await adapter.close()  # its client for api.x.ai never connected
    adapter._install_client(
        lambda: xai_sdk.AsyncClient(
            api_key="local-test",
            api_host=f"127.0.0.1:{port}",
            use_insecure_channel=True,
            channel_options=_CHANNEL_OPTIONS,
            timeout=timeout,
        )
    )
    return adapter
