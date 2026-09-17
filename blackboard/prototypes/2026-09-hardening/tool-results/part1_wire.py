"""Part 1: convert one neutral history (2 parallel tool calls) with every adapter. No network."""
from __future__ import annotations

import json

from ai_arch_toolkit.core._content import tool_result, user
from ai_arch_toolkit.core._response import Response, ToolCall

resp = Response(
    text="",
    tool_calls=(
        ToolCall(id="call_A", name="get_weather", input={"city": "Paris"}),
        ToolCall(id="call_B", name="get_weather", input={"city": "Tokyo"}),
    ),
)
# exactly what _react.py builds: [*messages, response.to_message(), *tool_result_dicts]
history = [
    user("Weather in Paris and Tokyo?"),
    resp.to_message(),
    tool_result('{"temp": 15}', tool_use_id="call_A", name="get_weather"),
    tool_result('{"temp": 22}', tool_use_id="call_B", name="get_weather"),
]
print("NEUTRAL HISTORY")
for m in history:
    print("  ", m)


def hdr(t: str) -> None:
    print("\n" + "=" * 8, t, "=" * 8)


hdr("ANTHROPIC  _messages_to_sdk")
from ai_arch_toolkit.core._providers import _anthropic as A

_, wire = A._messages_to_sdk(history)
for m in wire:
    print("  ", json.dumps(m))

hdr("OPENAI  _messages_to_sdk")
from ai_arch_toolkit.core._providers import _openai as O

for m in O._messages_to_sdk(history):
    print("  ", json.dumps(m))

hdr("GEMINI  _messages_to_sdk (no _raw)")
from ai_arch_toolkit.core._providers import _gemini as G

_, contents = G._messages_to_sdk(history)
for i, c in enumerate(contents):
    kinds = []
    for p in c.parts:
        if p.function_call:
            kinds.append(f"functionCall({p.function_call.name},{dict(p.function_call.args)},id={p.function_call.id})")
        elif p.function_response:
            fr = p.function_response
            kinds.append(f"functionResponse({fr.name},{fr.response},id={fr.id})")
        else:
            kinds.append(f"text({p.text!r})")
    print(f"   Content[{i}] role={c.role} parts={len(c.parts)}: {kinds}")

hdr("XAI  _messages_to_sdk")
from ai_arch_toolkit.core._providers import _xai as X
from xai_sdk.proto import chat_pb2

msgs, _ = X._messages_to_sdk(history)
for i, m in enumerate(msgs):
    role = chat_pb2.MessageRole.Name(m.role)
    tcs = [(t.id, t.function.name, t.function.arguments) for t in m.tool_calls]
    texts = [c.text for c in m.content]
    print(f"   Message[{i}] role={role} content={texts} tool_calls={tcs} tool_call_id={m.tool_call_id!r}")

hdr("META  _input_items (no _raw)")
from ai_arch_toolkit.core._providers import _meta as M

for it in M._input_items(history):
    print("  ", json.dumps(it))
