"""Offline probes of adapter preparation (no network)."""

from __future__ import annotations

import os
import warnings

os.environ.pop("OPENAI_API_KEY", None)

from ai_arch_toolkit.core._content import document, image, user
from ai_arch_toolkit.core._middleware import Request


def req(messages, **kwargs):
    return Request(messages=messages, system=None, tools=None, model="m", kwargs=kwargs)


print("--- A: Anthropic DocumentPart from a URL / data URL")
from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider

a = AnthropicProvider("claude-opus-5", "k")
for src in ("https://example.com/doc.pdf", "data:application/pdf;base64,JVBERi0="):
    p = a.prepare(req([user(["read", document(src)])], max_tokens=10))
    print(src[:30], "->", p.params["messages"][0]["content"][1]["source"])
p = a.prepare(req([user(["see", image("data:image/png;base64,iVBORw0KGgo=")])], max_tokens=10))
print("image data URL ->", p.params["messages"][0]["content"][1]["source"])

print("--- B: OpenAI-compatible DocumentPart from a data URL / URL")
from ai_arch_toolkit.core._providers._openai_compatible import OpenAICompatibleProvider

c = OpenAICompatibleProvider("llama3", "k", base_url="http://localhost:1/v1")
for src in ("https://example.com/doc.pdf", "data:application/pdf;base64,JVBERi0="):
    p = c.prepare(req([user(["read", document(src)])], max_tokens=10))
    print(src[:30], "->", p.params["messages"][0]["content"][1])

print("--- C: Responses DocumentPart from a data URL")
from ai_arch_toolkit.core._providers._openai import OpenAIProvider

o = OpenAIProvider("gpt-5.5", "k")
p = o.prepare(req([user(["read", document("data:application/pdf;base64,JVBERi0=")])], max_tokens=10))
print("->", p.params["input"][0]["content"][1])

print("--- D: Gemini max_output_tokens vs the LLM's max_tokens default")
from ai_arch_toolkit.core._providers._gemini import GeminiProvider

g = GeminiProvider("gemini-3.1-pro", "k")
p = g.prepare(req([user("hi")], max_tokens=4096, max_output_tokens=20000))
print("max_output_tokens sent:", p.params["config"].max_output_tokens)

print("--- E: Gemini DocumentPart from a URL")
try:
    g.prepare(req([user(["read", document("https://example.com/doc.pdf")])], max_tokens=10))
except Exception as exc:  # noqa: BLE001
    print(type(exc).__module__, type(exc).__name__, exc)

print("--- F: OpenAI max_completion_tokens vs the worst case")
p = o.prepare(req([user("hi")], max_tokens=4096, max_completion_tokens=100_000))
print("max_output_tokens sent:", p.params.get("max_output_tokens"))
