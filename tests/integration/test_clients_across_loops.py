"""SDK clients across the event loops of sync calls (G-17), against loopback servers.

The sync wrappers run every call on an event loop of its own, which closes after it. A client
built in one keeps pooled connections that died with that loop: a later call, a stream, and
``close()`` must neither reuse nor touch them. A provider's client is built inside the loop that
uses it, so no call needs a loop of its caller's.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core._providers import OWN_BASE_URLS
from ai_arch_toolkit.toolkit.moderation import OpenAIModerator
from tests.integration import fakegrpc, fakeserver

pytestmark = pytest.mark.integration

COMPATIBLE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "llama3.2",
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
CLAUDE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
CATEGORIES = (
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
)
MODERATION: dict[str, Any] = {
    "id": "modr-1",
    "model": "omni-moderation-latest",
    "results": [
        {
            "flagged": False,
            "categories": dict.fromkeys(CATEGORIES, False),
            "category_scores": dict.fromkeys(CATEGORIES, 0.0),
            "category_applied_input_types": {name: ["text"] for name in CATEGORIES},
        }
    ],
}


@pytest.mark.parametrize(
    ("model", "body", "path"),
    [("llama3.2", COMPATIBLE, "/v1"), ("claude-sonnet-4-6", CLAUDE, "")],
    ids=["compatible", "anthropic"],
)
def test_an_llm_closes_after_sync_calls(model: str, body: dict[str, Any], path: str) -> None:
    with fakeserver.KeepAlive(body) as server:
        base_url = f"http://127.0.0.1:{server.port}{path}"
        with LLM(model, base_url=base_url, api_key="local-test", max_tokens=16) as llm:
            assert llm.complete_sync("hi").text == "ok"
            assert llm.complete_sync("hi").text == "ok"

    assert server.stats.requests == 2


def test_an_xai_sync_stream_needs_no_loop_of_its_callers() -> None:
    asyncio.run(asyncio.sleep(0))  # this thread ran a loop, so none is current now (G-17)
    with (
        fakegrpc.serving_in_thread() as (script, port),
        LLM("grok-4.7", api_key="local-test", max_tokens=16) as llm,
    ):
        fakegrpc.point(llm._provider, port)
        assert "".join(llm.stream_sync("hi")) == "ok"
        assert llm.complete_sync("hi").text == "ok"
        # A later stream gets a client of its own loop, not the one of the closed loop.
        assert "".join(llm.stream_sync("hi")) == "ok"

    assert len(script.requests) == 3


def test_the_openai_moderator_survives_a_second_sync_call(monkeypatch: pytest.MonkeyPatch) -> None:
    with fakeserver.KeepAlive(MODERATION) as server:
        monkeypatch.setitem(OWN_BASE_URLS, "openai", f"http://127.0.0.1:{server.port}/v1")
        moderator = OpenAIModerator(api_key="local-test")
        assert moderator.moderate_sync("hi").flagged is False
        assert moderator.moderate_sync("hi").flagged is False
        asyncio.run(moderator.close())

    assert server.stats.requests == 2
