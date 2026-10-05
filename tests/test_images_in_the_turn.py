"""Images inside a turn (I04): the hosted tool, image events in streams, and the replay."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from google.genai import types
from openai.types.responses import Response as SDKResponse

from ai_arch_toolkit.core import (
    GeneratedImage,
    RequestError,
    StreamEvent,
    image_generation,
    user,
)
from ai_arch_toolkit.core._attempts import _partial
from ai_arch_toolkit.core._metering._scope import MeterScope
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._providers._base import CallPieces
from ai_arch_toolkit.core._providers._gemini import _chunk_events
from ai_arch_toolkit.core._providers._meta import MetaProvider, _input_items
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._tools import prepare_tools
from tests.provider_calls import assembled, prepare, stream
from tests.sdk_streams import OpenAIStream

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16
PNG_B64 = base64.b64encode(PNG).decode("ascii")
WEBP_B64 = base64.b64encode(WEBP).decode("ascii")
TOOL = image_generation(model="gpt-image-2.5-flare", quality="low", aspect_ratio="16:9")


def _turn(*output: dict[str, Any], model: str = "gpt-5-nano", **extra: Any) -> SDKResponse:
    body = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1.0,
        "model": model,
        "status": "completed",
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [{"type": "image_generation", "model": "gpt-image-2.5-flare"}],
        "usage": {
            "input_tokens": 2203,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 293,
            "output_tokens_details": {"reasoning_tokens": 256},
            "total_tokens": 2496,
        },
        **extra,
    }
    return SDKResponse.model_construct(**body)


def _drawn(result: str = PNG_B64, item_id: str = "ig_1") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "image_generation_call",
        "status": "completed",
        "result": result,
        "revised_prompt": "A red lighthouse",
        "output_format": "png",
    }


def _message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


TOOL_USAGE = {
    "image_gen": {
        "input_tokens": 21,
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 21},
        "output_tokens": 196,
        "output_tokens_details": {"image_tokens": 196, "text_tokens": 0},
        "total_tokens": 217,
    },
    "web_search": {"num_requests": 0},
}


def _openai(model: str = "gpt-5-nano", client: Any = None) -> OpenAIProvider:
    provider = OpenAIProvider(model, "test-key")
    provider._client = client or AsyncMock()
    return provider


# ── the tool ─────────────────────────────────────────────────────────────────


def test_the_tool_needs_its_image_model_and_checks_its_options() -> None:
    assert TOOL.config == {
        "model": "gpt-image-2.5-flare",
        "quality": "low",
        "aspect_ratio": "16:9",
    }
    with pytest.raises(RequestError, match="needs the image model"):
        image_generation(model="")
    with pytest.raises(RequestError, match="partial_images must be 0 to 3"):
        image_generation(model="gpt-image-2", partial_images=4)
    with pytest.raises(RequestError, match="aspect_ratio must be"):
        image_generation(model="gpt-image-2", aspect_ratio="wide")


def test_openai_sends_the_hosted_tool_with_the_portable_options_mapped() -> None:
    tool = image_generation(
        model="gpt-image-2.5-flare", quality="low", aspect_ratio="16:9", partial_images=2
    )
    params = prepare(_openai(), [user("draw a lighthouse")], tools=prepare_tools([tool])).params
    assert params["tools"] == [
        {
            "type": "image_generation",
            "model": "gpt-image-2.5-flare",
            "size": "1360x768",
            "quality": "low",
            "partial_images": 2,
        }
    ]


@pytest.mark.parametrize(
    ("tool", "message"),
    [
        (image_generation(model="gpt-5.5"), "not a GPT Image model"),
        (image_generation(model="gpt-image-2", quality="max"), "quality in"),
        (image_generation(model="gpt-image-1.5", aspect_ratio="16:9"), "1:1, 2:3 and 3:2"),
    ],
)
def test_openai_refuses_what_the_image_model_does_not_take(tool, message) -> None:
    with pytest.raises(RequestError, match=message):
        prepare(_openai(), [user("draw")], tools=prepare_tools([tool]))


def test_meta_runs_no_hosted_image_tool() -> None:
    provider = MetaProvider("muse-spark-1.3", "test-key")
    with pytest.raises(RequestError, match="not run by Meta"):
        prepare(provider, [user("draw")], tools=prepare_tools([TOOL]))


async def test_the_previews_are_asked_only_of_a_stream() -> None:
    # OpenAI answers partial_images without a stream with a 400 (live, I04).
    tool = image_generation(model="gpt-image-2.5-flare", partial_images=2)
    client = AsyncMock()
    client.responses.create = AsyncMock(return_value=_turn(_message("Done.")))
    provider = _openai(client=client)
    await provider.complete(prepare(provider, [user("draw")], tools=prepare_tools([tool])))
    sent = client.responses.create.call_args.kwargs["tools"]
    assert sent == [{"type": "image_generation", "model": "gpt-image-2.5-flare"}]


# ── the image and its cost ───────────────────────────────────────────────────


def test_the_hosted_image_comes_back_priced_at_its_image_model() -> None:
    final = _turn(_drawn(), _message("Here it is."), tool_usage=TOOL_USAGE)
    response = assembled(_openai(), final)
    assert response.text == "Here it is."
    assert response.images == (
        GeneratedImage(data=PNG, media_type="image/png", revised_prompt="A red lighthouse"),
    )
    # The tool's tokens count with the turn's...
    assert response.usage.input_tokens == 2203 + 21
    assert response.usage.image_output_tokens == 196
    # ...and its price is the image model's, added to the turn's own.
    turn = pricing.estimate_cost("gpt-5-nano", input_tokens=2203, output_tokens=293)
    image = pricing.estimate_cost("gpt-image-2.5-flare", input_tokens=21, image_output_tokens=196)
    assert turn is not None and image is not None
    assert response.cost == pytest.approx(turn + image)


def test_a_turn_without_the_image_tokens_keeps_its_own_cost() -> None:
    response = assembled(_openai(), _turn(_message("No image.")))
    assert response.images == ()
    assert response.cost == pytest.approx(
        pricing.estimate_cost("gpt-5-nano", input_tokens=2203, output_tokens=293)
    )


async def test_the_meter_takes_the_whole_cost_of_a_turn_that_drew() -> None:
    from ai_arch_toolkit import LLM

    client = AsyncMock()
    client.responses.create = AsyncMock(
        return_value=_turn(_drawn(), _message("Done."), tool_usage=TOOL_USAGE)
    )
    llm = LLM("gpt-5-nano", api_key="test-key")
    llm._provider = _openai(client=client)
    with MeterScope() as scope:
        out = await llm.complete("draw a lighthouse", tools=[TOOL])
    assert len(out.images) == 1
    snap = scope.snapshot()
    assert snap.unknown_cost_count == 0
    assert snap.cost.to_float() == pytest.approx(out.cost)


# ── streams ──────────────────────────────────────────────────────────────────


async def test_an_openai_stream_sends_the_previews_and_the_finished_image() -> None:
    final = _turn(_drawn(), _message("Done."), tool_usage=TOOL_USAGE)
    events = [
        {"type": "response.image_generation_call.partial_image", "partial_image_b64": WEBP_B64},
        {"type": "response.output_item.done", "item": SimpleNamespace(**_drawn())},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Done."},
        {"type": "response.completed", "response": final},
    ]
    client = AsyncMock()
    client.responses.create = AsyncMock(
        return_value=OpenAIStream([SimpleNamespace(**event) for event in events])
    )
    streamed, response = await stream(_openai(client=client), [user("draw")])
    images = [(e.image.media_type, e.partial) for e in streamed if e.kind == "image" and e.image]
    assert images == [("image/webp", True), ("image/png", False)]
    assert [i.data for i in response.images] == [PNG]


def test_a_gemini_chunk_sends_thought_images_as_previews() -> None:
    chunk = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=WEBP, mime_type="image/webp"), thought=True
                        ),
                        types.Part(inline_data=types.Blob(data=PNG, mime_type="image/png")),
                    ],
                )
            )
        ]
    )
    events = _chunk_events(chunk, CallPieces())
    assert [(e.kind, e.partial, e.image.data if e.image else None) for e in events] == [
        ("image", True, WEBP),
        ("image", False, PNG),
    ]


def test_an_abandoned_stream_keeps_the_finished_images_seen() -> None:
    finished = GeneratedImage(data=PNG, media_type="image/png")
    preview = GeneratedImage(data=WEBP, media_type="image/webp")
    events = [
        StreamEvent(kind="image", image=preview, partial=True),
        StreamEvent(kind="image", image=finished),
    ]
    assert _partial(events, "gpt-5-nano", "").images == (finished,)


# ── the replay ───────────────────────────────────────────────────────────────


def _history(final: SDKResponse, text: str) -> list[dict[str, Any]]:
    return [
        user("draw a lighthouse"),
        {"role": "assistant", "content": text, "_raw": final},
        user("now make the sky purple"),
    ]


def test_openai_sends_a_drawn_image_back_as_an_input_image() -> None:
    final = _turn(
        {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "enc"},
        _drawn(),
        _message("Here it is."),
    )
    params = prepare(_openai(), _history(final, "Here it is."), tools=prepare_tools([TOOL])).params
    kinds = [item.get("type") or item.get("role") for item in params["input"]]
    # The call itself is not replayed: OpenAI answers it with a 404 under store: false (I01).
    assert "image_generation_call" not in kinds
    assert kinds == ["user", "reasoning", "message", "user", "user"]
    drawn = params["input"][3]["content"]
    assert drawn == [{"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"}]


def test_openai_puts_the_image_before_a_turn_that_called_a_function() -> None:
    call = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "save",
        "arguments": "{}",
        "status": "completed",
    }
    final = _turn(_drawn(), call)
    history = [
        user("draw and save"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "save", "input": {}}],
            "_raw": final,
        },
        {"role": "tool", "content": "saved", "tool_use_id": "call_1"},
    ]
    params = prepare(_openai(), history, tools=prepare_tools([TOOL])).params
    kinds = [item.get("type") or item.get("role") for item in params["input"]]
    assert kinds == ["user", "user", "function_call", "function_call_output"]


def test_meta_sends_a_drawn_image_back_by_reference() -> None:
    final = _turn(_drawn(), _message("Here it is."), model="muse-image-1.0")
    items = _input_items(_history(final, "Here it is."), "muse-image-1.0")
    assert {
        "type": "image_generation_call",
        "id": "ig_1",
        "status": "completed",
        "result": None,
    } in items
    # A Muse Spark request does not replay Muse Image's turn: another family.
    spark = _input_items(_history(final, "Here it is."), "muse-spark-1.3")
    assert all(item.get("type") != "image_generation_call" for item in spark)
