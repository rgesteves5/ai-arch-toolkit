"""Gemini: replay of `_raw` from complete() vs from LLM.stream()/stream_events(). Mocked SDK client, no network."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from google.genai import types

from ai_arch_toolkit.core import LLM
from ai_arch_toolkit.core._content import tool_result, user
from ai_arch_toolkit.core._providers import _gemini as G

SIG = b"SIG-ON-FIRST-CALL"


def fc_part(city: str, sig: bytes | None = None) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(name="get_weather", args={"city": city}),
        thought_signature=sig,
    )


def chunk(parts: list[types.Part] | None, finish: str | None = None) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=parts) if parts is not None else None
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5
        ),
    )


def show(title: str, contents: list) -> None:
    print(f"\n--- {title}")
    for i, c in enumerate(contents):
        kinds = []
        for p in c.parts or []:
            sig = f",sig={p.thought_signature!r}" if p.thought_signature else ""
            if p.function_call:
                kinds.append(f"functionCall({dict(p.function_call.args)}{sig})")
            elif p.function_response:
                kinds.append(f"functionResponse({p.function_response.response})")
            else:
                kinds.append(f"text({p.text!r}{sig})")
        print(f"    Content[{i}] role={c.role} parts={len(c.parts or [])}: {kinds}")


def next_request(resp) -> list:
    hist = [user("Weather in Paris and Tokyo?"), resp.to_message()]
    for tc in resp.tool_calls:
        hist.append(tool_result('{"temp": 1}', tool_use_id=tc.id, name=tc.name))
    return G._messages_to_sdk(hist)[1]


class FakeModels:
    def __init__(self, chunks):
        self._chunks = chunks

    async def generate_content(self, **kw):
        return self._chunks[0]

    async def generate_content_stream(self, **kw):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


def make_llm(chunks) -> LLM:
    llm = LLM("gemini-3-flash-preview", api_key="not-a-key")
    llm._provider._client = SimpleNamespace(aio=SimpleNamespace(models=FakeModels(chunks)))
    return llm


async def main() -> None:
    tools = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]

    # (a) complete(): one response, both calls, signature on the FIRST functionCall part
    full = chunk([fc_part("Paris", SIG), fc_part("Tokyo")], "STOP")
    resp = await make_llm([full]).complete("q", tools=tools)
    print("complete(): tool_calls =", [tc.input for tc in resp.tool_calls])
    show("next request after complete()", next_request(resp))

    # (b) stream(): each call in its own chunk, final chunk carries only finish_reason
    for label, chunks in {
        "stream: [callA+sig] [callB] [finish, parts=[]]": [chunk([fc_part("Paris", SIG)]), chunk([fc_part("Tokyo")]), chunk([], "STOP")],
        "stream: [callA+sig] [callB+finish]": [chunk([fc_part("Paris", SIG)]), chunk([fc_part("Tokyo")], "STOP")],
        "stream: [callA+sig] [callB] [finish, content=None]": [chunk([fc_part("Paris", SIG)]), chunk([fc_part("Tokyo")]), chunk(None, "STOP")],
    }.items():
        s = make_llm(chunks).stream("q", tools=tools)
        async for _ in s:
            pass
        r = s.response
        print(f"\n{label}\n    neutral tool_calls = {[tc.input for tc in r.tool_calls]}  raw is last chunk = {r.raw is chunks[-1]}")
        show("next request after stream()", next_request(r))

    # (c) stream_events() goes through the same provider.stream()
    chunks = [chunk([fc_part("Paris", SIG)]), chunk([fc_part("Tokyo")]), chunk([], "STOP")]
    s = make_llm(chunks).stream_events("q", tools=tools)
    async for _ in s:
        pass
    show("next request after stream_events() [callA+sig][callB][finish, parts=[]]", next_request(s.response))


asyncio.run(main())
