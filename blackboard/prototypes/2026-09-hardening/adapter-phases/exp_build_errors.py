"""Which exception types escape the adapters' BUILD code (pure, no I/O)? Dummy keys, no requests."""

from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

from ai_arch_toolkit.core._content import ImagePart  # noqa: E402
from ai_arch_toolkit.core._providers import _anthropic as A  # noqa: E402
from ai_arch_toolkit.core._providers import _gemini as G  # noqa: E402
from ai_arch_toolkit.core._providers import _meta as M  # noqa: E402
from ai_arch_toolkit.core._providers import _openai as O  # noqa: E402
from ai_arch_toolkit.core._providers import _xai as X  # noqa: E402

SYS_IMG = [{"role": "system", "content": [ImagePart(source=b"x")]}]
BAD_TOOLCALL = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "f", "input": {"x": object()}}]}]
BAD_B64 = [{"role": "user", "content": [ImagePart(source="@@not-base64@@")]}]
NAMELESS_TOOL = [{"description": "no name"}]


def t(label: str, fn) -> None:
    try:
        fn()
        print(f"  {label:58s} -> ok (no exception)")
    except BaseException as exc:  # noqa: BLE001
        bases = [c.__name__ for c in type(exc).__mro__[1:-2]]
        print(f"  {label:58s} -> {type(exc).__module__}.{type(exc).__name__} {bases}")


a = A.AnthropicProvider("claude-sonnet-4-5", "dummy")
o = O.OpenAIProvider("gpt-4o-mini", "dummy")
astra = O.OpenAIProvider("gpt-6-astra", "dummy")
g = G.GeminiProvider("gemini-2.5-flash", "dummy")
x = X.XAIProvider("grok-4", "dummy")
m = M.MetaProvider("muse-spark-1", "dummy")

print("anthropic")
t("system message holding an image", lambda: A._messages_to_sdk(SYS_IMG))
t("structured_output_mode='bogus'", lambda: a._build_sdk_kwargs([], structured_output_mode="bogus"))
t("tool without 'name'", lambda: a._build_sdk_kwargs([], tools=NAMELESS_TOOL))
print("openai")
t("system message holding an image", lambda: O._messages_to_sdk(SYS_IMG))
t("non-JSON-serializable tool_call input", lambda: O._messages_to_sdk(BAD_TOOLCALL))
t("gpt-6-astra + tools", lambda: astra._build_sdk_kwargs([], tools=[{"name": "f"}]))
t("tool without 'name'", lambda: o._build_sdk_kwargs([], tools=NAMELESS_TOOL))
print("gemini")
t("system message holding an image", lambda: G._messages_to_sdk(SYS_IMG))
t("image source that is not base64", lambda: G._messages_to_sdk(BAD_B64))
t("temperature='hot' (pydantic config validation)", lambda: g._build_config(temperature="hot"))
t("tool without 'name'", lambda: g._build_config(tools=NAMELESS_TOOL))
t("role='tool-ish' unknown role", lambda: G._messages_to_sdk([{"role": "weird", "content": "x"}]))
print("xai")
t("system message holding an image", lambda: X._messages_to_sdk(SYS_IMG))
t("non-JSON-serializable tool_call input", lambda: X._messages_to_sdk(BAD_TOOLCALL))
t("user content that is not str (int)", lambda: X._messages_to_sdk([{"role": "user", "content": 5}]))
t("server tool", lambda: x._build_create_kwargs([], tools=[{"_server_tool": True, "type": "web_search"}]))
print("meta")
t("system message holding an image", lambda: M._input_items(SYS_IMG))
t("non-JSON-serializable tool_call input", lambda: M._input_items(BAD_TOOLCALL))
t("tool_choice='required'", lambda: m._build_request([], tool_choice="required"))
t("tool without 'name'", lambda: m._build_request([], tools=NAMELESS_TOOL))

import pydantic  # noqa: E402

print("\npydantic.ValidationError is ValueError:", issubclass(pydantic.ValidationError, ValueError))
