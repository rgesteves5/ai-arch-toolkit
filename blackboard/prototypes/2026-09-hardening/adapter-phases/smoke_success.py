"""Success-path smoke through the REAL SDKs against a loopback fake server (no network, dummy keys).

Checks that, on the installed SDK versions, each HTTP adapter sends a request the SDK accepts and
parses a canonical 200 response into the toolkit ``Response`` (text, tool call, usage).
"""

from __future__ import annotations

import asyncio
import json
import sys
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import fakeserver as fs  # noqa: E402

from ai_arch_toolkit.core._providers._anthropic import AnthropicProvider  # noqa: E402
from ai_arch_toolkit.core._providers._gemini import GeminiProvider  # noqa: E402
from ai_arch_toolkit.core._providers._meta import MetaProvider  # noqa: E402
from ai_arch_toolkit.core._providers._openai import OpenAIProvider  # noqa: E402

warnings.simplefilter("ignore")

TOOLS = [
    {
        "name": "get_weather",
        "description": "Weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]
MESSAGES = [{"role": "user", "content": "Weather in Lisbon?"}]

ANTHROPIC_200 = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
    "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Lisbon"}},
    ],
    "stop_reason": "tool_use", "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 7},
}
OPENAI_200 = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
    "choices": [{
        "index": 0, "finish_reason": "tool_calls",
        "message": {
            "role": "assistant", "content": "Let me check.",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})},
            }],
        },
    }],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
}
META_200 = {
    "id": "resp_1", "object": "response", "created_at": 1, "status": "completed",
    "model": "muse-spark-1.3", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    "output": [
        {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Let me check.", "annotations": []}]},
        {"type": "function_call", "id": "fc1", "call_id": "call_1", "name": "get_weather",
         "arguments": json.dumps({"city": "Lisbon"}), "status": "completed"},
    ],
    "usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19,
              "input_tokens_details": {"cached_tokens": 0},
              "output_tokens_details": {"reasoning_tokens": 0}},
}
GEMINI_200 = {
    "candidates": [{
        "content": {"role": "model", "parts": [
            {"text": "Let me check."},
            {"functionCall": {"name": "get_weather", "args": {"city": "Lisbon"}}},
        ]},
        "finishReason": "STOP", "index": 0,
    }],
    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7, "totalTokenCount": 19},
    "modelVersion": "gemini-2.5-flash",
}


async def run(name: str, make, body: dict) -> None:
    server, port, stats = await fs.start("status", status=200, body=body)
    try:
        provider = make(port)
        resp = await provider.complete(MESSAGES, tools=TOOLS, max_tokens=64)
        calls = [(c.name, c.input) for c in resp.tool_calls]
        sent = json.loads(stats.bodies[0]) if stats.bodies else {}
        print(
            f"{name:<10} path={stats.paths[0]:<42} text={resp.text!r} tool_calls={calls} "
            f"usage=({resp.usage.input_tokens},{resp.usage.output_tokens}) "
            f"sent_keys={sorted(sent)[:8]}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{name:<10} FAILED: {type(exc).__name__}: {exc}")
    finally:
        server.close()


def _gemini(port: int) -> GeminiProvider:
    from google import genai

    p = GeminiProvider("gemini-2.5-flash", "dummy-key", timeout=5.0)
    opts = {"base_url": f"http://127.0.0.1:{port}", "retry_options": {"attempts": 1}, "timeout": 5000}
    p._client = genai.Client(api_key="dummy-key", http_options=opts)  # loopback only
    return p


async def main() -> None:
    await run("anthropic", lambda port: AnthropicProvider(
        "claude-sonnet-4-6", "dummy-key", base_url=f"http://127.0.0.1:{port}"), ANTHROPIC_200)
    await run("openai", lambda port: OpenAIProvider(
        "gpt-4o-mini", "dummy-key", base_url=f"http://127.0.0.1:{port}/v1"), OPENAI_200)
    await run("meta", lambda port: MetaProvider(
        "muse-spark-1.3", "dummy-key", base_url=f"http://127.0.0.1:{port}/v1"), META_200)
    await run("gemini", _gemini, GEMINI_200)


asyncio.run(main())
