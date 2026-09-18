"""Direct calls to an adapter through the provider contract, for the adapters' own tests."""

from __future__ import annotations

from typing import Any

from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import Answer, BaseProvider, Prepared
from ai_arch_toolkit.core._response import OutputSchema, Response, StreamEvent


def prepare(
    provider: BaseProvider[Any, Any],
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """What the adapter would send for this call."""
    request = Request(
        messages=messages, system=system, tools=tools, model=provider._model, kwargs=kwargs
    )
    return provider.prepare(request)


async def complete(
    provider: BaseProvider[Any, Any], messages: list[dict[str, Any]], **kwargs: Any
) -> Response:
    """The assembled response of one call."""
    return (await provider.complete(prepare(provider, messages, **kwargs))).response


async def stream(
    provider: BaseProvider[Any, Any], messages: list[dict[str, Any]], **kwargs: Any
) -> tuple[list[StreamEvent], Response]:
    """Every event of one streamed call, and its assembled response."""
    events: list[StreamEvent] = []
    response: Response | None = None
    async for item in provider.stream(prepare(provider, messages, **kwargs)):
        if isinstance(item, Answer):
            response = item.response
        else:
            events.append(item)
    assert response is not None
    return events, response


def assembled(
    provider: BaseProvider[Any, Any], final: Any, *, output_schema: OutputSchema | None = None
) -> Response:
    """The response the base builds from an SDK final object: assembly, usage and cost."""
    return provider._answer(final, Prepared({}, output_schema=output_schema)).response
