"""What-if (scratch only, tracked file untouched): drop the flush at _gemini.py:156 and re-run the conversion."""
from __future__ import annotations

import inspect

from ai_arch_toolkit.core._content import tool_result, user
from ai_arch_toolkit.core._providers import _gemini as G
from ai_arch_toolkit.core._response import Response, ToolCall

src = inspect.getsource(G._messages_to_sdk)
needle = '        if msg.get("tool_use_id"):\n            _flush_fn_responses()\n'
assert src.count(needle) == 1
patched = src.replace(needle, '        if msg.get("tool_use_id"):\n')
ns = dict(vars(G))
exec(patched, ns)

resp = Response(tool_calls=(ToolCall(id="a", name="get_weather", input={"city": "Paris"}), ToolCall(id="b", name="get_weather", input={"city": "Tokyo"})))
hist = [user("q"), resp.to_message(), tool_result('{"t":1}', tool_use_id="a", name="get_weather"), tool_result('{"t":2}', tool_use_id="b", name="get_weather"), user("thanks")]
for label, fn in (("current ", G._messages_to_sdk), ("what-if ", ns["_messages_to_sdk"])):
    _, contents = fn(hist)
    print(label, [(c.role, ["FC" if p.function_call else "FR" if p.function_response else "text" for p in c.parts]) for c in contents])
