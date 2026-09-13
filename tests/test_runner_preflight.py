"""run_tools refuses a response naming an unknown tool before running any of its calls."""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._response import Response, ToolCall
from ai_arch_toolkit.core._tools._decorator import tool
from ai_arch_toolkit.core._tools._group import ToolGroup
from ai_arch_toolkit.toolkit import run_tools, run_tools_sync

SENT: list[str] = []


@tool
def send_email(to: str) -> str:
    """Send an email (a side effect)."""
    SENT.append(to)
    return "sent"


def _response() -> Response:
    return Response(
        tool_calls=(
            ToolCall(id="1", name="send_email", input={"to": "a@example.com"}),
            ToolCall(id="2", name="does_not_exist", input={}),
        )
    )


@pytest.fixture(autouse=True)
def _reset() -> None:
    SENT.clear()


@pytest.mark.parametrize("as_group", [False, True])
async def test_run_tools_checks_every_name_before_running_anything(as_group: bool) -> None:
    tools = ToolGroup(send_email) if as_group else [send_email]

    with pytest.raises(KeyError, match="does_not_exist"):
        await run_tools(_response(), tools)

    assert SENT == []


@pytest.mark.parametrize("as_group", [False, True])
def test_run_tools_sync_checks_every_name_before_running_anything(as_group: bool) -> None:
    tools = ToolGroup(send_email) if as_group else [send_email]

    with pytest.raises(KeyError, match="does_not_exist"):
        run_tools_sync(_response(), tools)

    assert SENT == []
