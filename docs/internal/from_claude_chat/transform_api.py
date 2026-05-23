"""
Transform: LLM Calls

The public surface for making LLM calls.
Everything here is content → content.

Users need exactly four things:
  1. Build content     (messages)
  2. Pick a transform  (model)
  3. Run it            (call)
  4. Read the result   (response)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# 1. CONTENT — building messages
#
# Messages are dicts. Always were, always will be.
# These constructors are convenience, not abstraction.
# Users can always just pass raw dicts.
# ═══════════════════════════════════════════════════════════════════


def system(content: str) -> dict:
    """System instruction."""
    return {"role": "system", "content": content}


def user(content: str | list) -> dict:
    """User message. Accepts str or multimodal list."""
    return {"role": "user", "content": content}


def assistant(content: str) -> dict:
    """Assistant message (for conversation history)."""
    return {"role": "assistant", "content": content}


def tool_result(content: Any, tool_use_id: str) -> dict:
    """Result from a tool call."""
    return {"role": "tool", "content": content, "tool_use_id": tool_use_id}


# ═══════════════════════════════════════════════════════════════════
# 2. RESPONSE — what comes back
#
# Behaves like a string (the common case).
# But carries everything when you need it.
# ═══════════════════════════════════════════════════════════════════


class Response:
    """
    The result of a transform.

    Behaves like a string for the simple case:
        result = await llm(messages)
        print(result)                  # just the text

    Rich when you need it:
        result.text                    # the generated text
        result.tool_calls              # list of tool calls (if any)
        result.input_tokens            # tokens in
        result.output_tokens           # tokens out
        result.cost                    # estimated cost in USD
        result.stop_reason             # why it stopped
        result.model                   # which model actually ran
        result.raw                     # the raw provider response, untouched
    """

    __slots__ = (
        "cost",
        "input_tokens",
        "model",
        "output_tokens",
        "raw",
        "stop_reason",
        "text",
        "tool_calls",
    )

    def __init__(
        self,
        text: str = "",
        tool_calls: list[dict] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        stop_reason: str = "",
        model: str = "",
        raw: Any = None,
    ):
        self.text = text
        self.tool_calls = tool_calls or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost = cost
        self.stop_reason = stop_reason
        self.model = model
        self.raw = raw

    # Behaves like a string
    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text

    def __bool__(self) -> bool:
        return bool(self.text or self.tool_calls)

    def __contains__(self, item: str) -> bool:
        return item in self.text

    def __add__(self, other: str) -> str:
        return self.text + other

    def __radd__(self, other: str) -> str:
        return other + self.text

    # Convenience
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ═══════════════════════════════════════════════════════════════════
# 3. TRANSFORM — the function
#
# One class. One method that matters: __call__.
# messages in → Response out.
#
# Everything else (provider, auth, HTTP) is hidden.
# The user should not care.
# ═══════════════════════════════════════════════════════════════════


class Transform:
    """
    An LLM. Content → Content.

    Create:
        llm = Transform("claude-sonnet-4-5-20250929")
        llm = Transform("gpt-4o")
        llm = Transform("ollama:llama3")
        llm = Transform("claude-sonnet-4-5-20250929", temperature=0.7, max_tokens=1000)

    Call:
        response = await llm(messages)
        response = await llm("single string shorthand")

    Stream:
        async for chunk in llm.stream(messages):
            print(chunk, end="")

    Configure per-call (override defaults):
        response = await llm(messages, temperature=0.9)
        response = await llm(messages, max_tokens=100)
        response = await llm(messages, tools=[...])

    That's the entire public API for LLM calls.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        # Provider-specific
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ): ...

    async def __call__(
        self,
        messages: list[dict] | str,
        *,
        # Per-call overrides
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Response:
        """
        The one method.

        messages in → Response out.

        Accepts:
          - list[dict]: standard messages
          - str: shorthand for [user("your string")]
        """
        ...

    async def stream(
        self,
        messages: list[dict] | str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Same transform, incremental delivery.

        async for chunk in llm.stream(messages):
            print(chunk, end="")
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# 4. CONSTRUCTOR — picking a transform
#
# Optional. Transform("model") already works.
# This exists only if you want a shorter name.
# ═══════════════════════════════════════════════════════════════════


def model(name: str, **kwargs) -> Transform:
    """
    Shorthand constructor.

        llm = model("claude-sonnet-4-5-20250929")

    Identical to Transform("claude-sonnet-4-5-20250929").
    Exists because 'model' reads better than 'Transform' in user code.
    """
    return Transform(name, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# FULL USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════
#
# --- Simplest possible ---
#
#   llm = Transform("claude-sonnet-4-5-20250929")
#   result = await llm("What is 2+2?")
#   print(result)  # "4"
#
#
# --- With conversation ---
#
#   result = await llm([
#       system("You are a math tutor."),
#       user("What is 2+2?"),
#   ])
#
#
# --- Streaming ---
#
#   async for chunk in llm.stream("Tell me a story"):
#       print(chunk, end="")
#
#
# --- Check usage ---
#
#   result = await llm("Explain gravity")
#   print(result)              # the text
#   print(result.tokens)       # total tokens
#   print(result.cost)         # estimated USD
#
#
# --- Per-call overrides ---
#
#   result = await llm(messages, temperature=0.9, max_tokens=100)
#
#
# --- With tools (next layer, but the call surface is the same) ---
#
#   result = await llm(messages, tools=[search_schema])
#   if result.has_tool_calls:
#       for call in result.tool_calls:
#           print(call["name"], call["input"])
#
#
# --- Raw provider access (escape hatch) ---
#
#   result = await llm(messages)
#   anthropic_response = result.raw   # the original SDK response object
#
# ═══════════════════════════════════════════════════════════════════
