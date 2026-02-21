"""24 — Custom Middleware + Short-Circuit (OpenAI).

Demonstrates a user-defined middleware that:
  - attaches request metadata
  - short-circuits selected prompts without calling a provider
  - post-processes responses in ``after()``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from uuid import uuid4

from ai_arch_toolkit import Client, Message, Middleware, Request, Response, ToolResult

_SHORT_CIRCUIT_RESULT_KEY = "middleware.short_circuit_result"


def _extract_first_user_text(
    messages: Sequence[Message | ToolResult],
) -> str:
    for item in messages:
        if isinstance(item, Message) and item.role == "user" and isinstance(item.content, str):
            return item.content
    return ""


class RequestAuditMiddleware(Middleware):
    """Simple middleware for request tagging + optional local short-circuit."""

    def before(self, request: Request) -> Request:
        request_id = uuid4().hex[:8]
        request.context["request_id"] = request_id

        user_text = _extract_first_user_text(request.messages)
        if request.operation == "chat" and user_text.lower().startswith("local:"):
            payload = user_text.split(":", maxsplit=1)[1].strip()
            request.context[_SHORT_CIRCUIT_RESULT_KEY] = Response(
                text=f"[local middleware response] {payload}"
            )
            request.context["short_circuit"] = True
        return request

    def after(self, request: Request, result: object) -> object:
        if not isinstance(result, Response):
            return result
        mode = "short-circuit" if request.context.get("short_circuit") else "provider"
        request_id = request.context.get("request_id", "unknown")
        suffix = f"\n\n[middleware mode={mode} request_id={request_id}]"
        return replace(result, text=f"{result.text}{suffix}")

    async def abefore(self, request: Request) -> Request:
        return self.before(request)

    async def aafter(self, request: Request, result: object) -> object:
        return self.after(request, result)


client = Client("openai", model="gpt-5-nano", middleware=[RequestAuditMiddleware()])

print("=== Normal provider-backed call ===")
normal = client.chat("Explain ACID transactions in one short paragraph.")
print(normal.text)

print("\n=== Middleware short-circuit call ===")
local = client.chat("local:This response should come from middleware, not the provider.")
print(local.text)
