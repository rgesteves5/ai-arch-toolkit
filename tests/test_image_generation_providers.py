"""Image generation in each adapter (I03): the request each prepares, and the images it reads."""

from __future__ import annotations

import base64
import dataclasses
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from google.genai import types
from openai.types import Image, ImagesResponse
from openai.types.images_response import Usage as ImagesUsage
from openai.types.responses import Response as SDKResponse
from xai_sdk.proto import usage_pb2

from ai_arch_toolkit import LLM
from ai_arch_toolkit.core import GeneratedImage, RequestError, ResponseError, Usage, image, user
from ai_arch_toolkit.core._images import ImageRequest
from ai_arch_toolkit.core._metering._scope import MeterScope
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._providers import create_provider
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._providers._openai_images import ImagesCall, image_size
from ai_arch_toolkit.core._providers._xai import XAIProvider
from tests.provider_calls import (
    assembled,
    generate_image,
    image_request,
    prepare,
    prepare_image,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16
PNG_B64 = base64.b64encode(PNG).decode("ascii")


def _images_response(
    *data: bytes, usage: ImagesUsage | None = None, **fields: Any
) -> ImagesResponse:
    items = [Image(b64_json=base64.b64encode(d).decode("ascii")) for d in data]
    return ImagesResponse(created=1, data=items, usage=usage, **fields)


def _openai_usage(text_in: int, image_in: int, image_out: int) -> ImagesUsage:
    return ImagesUsage.model_validate(
        {
            "input_tokens": text_in + image_in,
            "input_tokens_details": {"text_tokens": text_in, "image_tokens": image_in},
            "output_tokens": image_out,
            "output_tokens_details": {"text_tokens": 0, "image_tokens": image_out},
            "total_tokens": text_in + image_in + image_out,
        }
    )


def _openai(model: str = "gpt-image-2.5-flare", **calls: Any) -> OpenAIProvider:
    provider = OpenAIProvider(model, "test-key")
    client = AsyncMock()
    client.images.generate = AsyncMock(return_value=calls.get("generate"))
    client.images.edit = AsyncMock(return_value=calls.get("edit"))
    provider._client = client
    return provider


# ── the portable size ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("options", "rule", "size"),
    [
        ({}, "free", None),
        ({"aspect_ratio": "16:9"}, "free", "1360x768"),
        ({"aspect_ratio": "1:1", "resolution": "2K"}, "free", "2048x2048"),
        ({"aspect_ratio": "16:9", "resolution": "4K"}, "free", "3840x2160"),
        ({"resolution": "4K"}, "free", "2880x2880"),
        ({"aspect_ratio": "3:1"}, "free", "1776x592"),
        ({"aspect_ratio": "3:2"}, "fixed", "1536x1024"),
        ({"aspect_ratio": "2:3", "resolution": "1K"}, "fixed", "1024x1536"),
        ({"resolution": "1K"}, "fixed", "1024x1024"),
        ({"aspect_ratio": "4:7"}, "ratio", "768x1360"),
    ],
)
def test_the_size_follows_the_ratio_and_the_resolution(options, rule, size) -> None:
    assert image_size(ImageRequest(**options), rule) == size


@pytest.mark.parametrize(
    ("options", "rule", "message"),
    [
        ({"resolution": "512"}, "free", "below this model's 655,360 pixels"),
        ({"aspect_ratio": "4:1"}, "free", "between 1:3 and 3:1"),
        ({"aspect_ratio": "16:9"}, "fixed", "1:1, 2:3 and 3:2 only"),
        ({"resolution": "2K"}, "fixed", "1K only"),
        ({"resolution": "2K"}, "ratio", "no resolution"),
    ],
)
def test_a_size_the_model_cannot_draw_is_refused(options, rule, message) -> None:
    with pytest.raises(RequestError, match=message):
        image_size(ImageRequest(**options), rule)


# ── OpenAI ───────────────────────────────────────────────────────────────────


class TestOpenAI:
    def test_a_generation_goes_to_the_images_api(self) -> None:
        call = prepare_image(
            _openai(), "a lighthouse", aspect_ratio="16:9", quality="max", output_format="webp"
        )
        assert isinstance(call, ImagesCall) and not call.edit
        assert call.params == {
            "model": "gpt-image-2.5-flare",
            "prompt": "a lighthouse",
            "size": "1360x768",
            "quality": "max",
            "output_format": "webp",
        }
        assert prepare_image(_openai(), "a", n=3).params["n"] == 3

    def test_input_images_make_an_edit_with_their_bytes(self) -> None:
        call = prepare_image(
            _openai(),
            "make the sky purple",
            image(PNG),
            image(PNG_B64),
            image(f"data:image/png;base64,{PNG_B64}"),
        )
        assert call.edit
        assert call.params["image"] == [
            ("image-0", PNG, "image/png"),
            ("image-1", PNG, "image/png"),
            ("image-2", PNG, "image/png"),
        ]

    def test_one_input_image_goes_as_one_file(self) -> None:
        # The SDK names a list's files "image[]", which Meta refuses (live, I03).
        call = prepare_image(_openai(), "make the sky purple", image(PNG))
        assert call.params["image"] == ("image-0", PNG, "image/png")

    def test_a_web_url_is_not_uploaded(self) -> None:
        with pytest.raises(RequestError, match="pass its bytes, not a URL"):
            prepare_image(_openai(), "edit", image("https://example.com/a.png"))

    @pytest.mark.parametrize(
        ("model", "options", "message"),
        [
            ("gpt-image-2", {"quality": "max"}, "quality in"),
            ("gpt-image-1.5", {"aspect_ratio": "16:9"}, "1:1, 2:3 and 3:2 only"),
            ("gpt-image-2.5-flare", {"n": 11}, "at most 10 images"),
            ("gpt-5.5", {}, "not an image model"),
        ],
    )
    def test_what_the_model_does_not_take_is_refused(self, model, options, message) -> None:
        with pytest.raises(RequestError, match=message):
            prepare_image(_openai(model), "a lighthouse", **options)

    def test_too_many_input_images_are_refused(self) -> None:
        with pytest.raises(RequestError, match="at most 16 input images"):
            prepare_image(_openai(), "merge", *[image(PNG)] * 17)

    @pytest.mark.parametrize(
        "model", ["gpt-image-2.5-flare-2026-09-08", "gpt-image-3", "chatgpt-image-latest"]
    )
    def test_snapshots_and_newer_models_are_image_models(self, model) -> None:
        assert isinstance(prepare_image(_openai(model), "a lighthouse"), ImagesCall)

    def test_an_image_model_takes_no_chat(self) -> None:
        with pytest.raises(RequestError, match="is an image model: call generate_image"):
            prepare(_openai(), [user("hi")])

    def test_call_options_added_by_a_middleware_are_refused(self) -> None:
        provider = _openai()
        request = image_request(provider, "a lighthouse")
        request = dataclasses.replace(request, kwargs={"thinking": True})
        with pytest.raises(RequestError, match="takes no call options"):
            provider.prepare_image(request)

    async def test_the_images_come_back_with_usage_and_cost(self) -> None:
        answer = _images_response(PNG, usage=_openai_usage(21, 0, 196), output_format="png")
        provider = _openai(generate=answer)
        response = await generate_image(provider, "a lighthouse")
        assert response.images == (GeneratedImage(data=PNG, media_type="image/png"),)
        assert response.usage == Usage(input_tokens=21, image_output_tokens=196, image_count=1)
        assert response.cost == pytest.approx((21 * 5.0 + 196 * 30.0) / 1e6)
        provider._client.images.generate.assert_awaited_once()

    async def test_an_edit_counts_the_input_image_tokens(self) -> None:
        answer = _images_response(PNG, usage=_openai_usage(28, 1024, 196))
        provider = _openai(edit=answer)
        response = await generate_image(provider, "make the sky purple", image(PNG))
        assert response.usage.image_input_tokens == 1024
        assert response.cost == pytest.approx((28 * 5.0 + 1024 * 8.0 + 196 * 30.0) / 1e6)
        provider._client.images.edit.assert_awaited_once()
        provider._client.images.generate.assert_not_awaited()

    async def test_llm_generate_image_is_metered_end_to_end(self) -> None:
        answer = _images_response(PNG, usage=_openai_usage(21, 0, 196))
        llm = LLM("gpt-image-2.5-flare", api_key="test-key")
        llm._provider = _openai(generate=answer)
        with MeterScope() as scope:
            out = await llm.generate_image("a lighthouse", aspect_ratio="16:9")
        assert out.images[0].data == PNG
        assert scope.snapshot().cost.to_float() == pytest.approx((21 * 5.0 + 196 * 30.0) / 1e6)
        sent = llm._provider._client.images.generate.call_args.kwargs
        assert sent["size"] == "1360x768"

    def test_an_image_from_the_hosted_tool_comes_back_in_a_response(self) -> None:
        body = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1.0,
            "model": "gpt-5-nano",
            "status": "completed",
            "output": [
                {
                    "id": "ig_1",
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": base64.b64encode(WEBP).decode("ascii"),
                    "output_format": "webp",
                    "revised_prompt": "A red lighthouse",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        response = assembled(_openai("gpt-5-nano"), SDKResponse.model_construct(**body))
        assert response.images == (
            GeneratedImage(data=WEBP, media_type="image/webp", revised_prompt="A red lighthouse"),
        )


# ── Meta ─────────────────────────────────────────────────────────────────────


class TestMeta:
    @staticmethod
    def provider(model: str = "muse-image-1.0", answer: Any = None) -> MetaProvider:
        provider = MetaProvider(model, "test-key")
        client = AsyncMock()
        client.images.generate = AsyncMock(return_value=answer)
        provider._client = client
        return provider

    def test_muse_image_is_routed_to_meta(self) -> None:
        assert isinstance(create_provider("muse-image-1.0", api_key="k"), MetaProvider)

    def test_the_size_carries_only_the_ratio(self) -> None:
        call = prepare_image(self.provider(), "a lighthouse", aspect_ratio="1:1")
        assert call.params == {
            "model": "muse-image-1.0",
            "prompt": "a lighthouse",
            "size": "1024x1024",
        }

    @pytest.mark.parametrize(
        ("options", "message"),
        [({"quality": "high"}, "takes no quality"), ({"resolution": "2K"}, "no resolution")],
    )
    def test_what_muse_image_does_not_take_is_refused(self, options, message) -> None:
        with pytest.raises(RequestError, match=message):
            prepare_image(self.provider(), "a lighthouse", **options)

    def test_muse_image_edits_one_image_at_a_time(self) -> None:
        with pytest.raises(RequestError, match="at most 1 input images"):
            prepare_image(self.provider(), "merge", image(PNG), image(PNG))

    def test_a_drawn_image_in_a_chat_is_charged_its_flat_price(self) -> None:
        body = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1.0,
            "model": "muse-image-1.0",
            "status": "completed",
            "output": [
                {
                    "id": "ig_1",
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": base64.b64encode(WEBP).decode(),
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 9744,
                "input_tokens_details": {"cached_tokens": 7936},
                "output_tokens": 678,
                "output_tokens_details": {"reasoning_tokens": 100},
                "total_tokens": 10422,
            },
        }
        response = assembled(self.provider(), SDKResponse.model_construct(**body))
        assert response.usage.image_count == 1
        assert response.cost == pytest.approx(0.01)

    def test_muse_spark_is_not_an_image_model(self) -> None:
        with pytest.raises(RequestError, match="not an image model"):
            prepare_image(self.provider("muse-spark-1.3"), "a lighthouse")

    def test_muse_image_takes_no_tools_in_a_chat(self) -> None:
        tool = {"name": "f", "description": "", "parameters": {"type": "object"}}
        with pytest.raises(RequestError, match="takes no tools"):
            prepare(self.provider(), [user("draw")], tools=[tool])

    async def test_each_image_costs_its_flat_price(self) -> None:
        # Meta's usage comes without details, in tokens that are not the bill (I01).
        usage = ImagesUsage.model_construct(
            input_tokens=10025, output_tokens=945, total_tokens=10970
        )
        answer = ImagesResponse.model_construct(
            created=1,
            data=[Image(b64_json=base64.b64encode(WEBP).decode("ascii"))],
            usage=usage,
            output_format="webp",
        )
        response = await generate_image(self.provider(answer=answer), "a lighthouse")
        assert response.images[0].media_type == "image/webp"
        assert response.usage.image_count == 1
        assert response.cost == pytest.approx(0.01)


# ── Gemini ───────────────────────────────────────────────────────────────────


class TestGemini:
    @staticmethod
    def provider(model: str = "gemini-3.1-flash-image") -> GeminiProvider:
        return GeminiProvider(model, "test-key")

    def test_a_generation_asks_for_image_output(self) -> None:
        prepared = prepare_image(
            self.provider(), "a lighthouse", image(PNG), aspect_ratio="16:9", resolution="512"
        )
        config = prepared.params["config"]
        assert config.response_modalities == ["IMAGE"]
        assert config.image_config == types.ImageConfig(aspect_ratio="16:9", image_size="512")
        parts = prepared.params["contents"][0].parts
        assert parts[0].text == "a lighthouse"
        assert parts[1].inline_data.data == PNG

    @pytest.mark.parametrize(
        ("model", "options", "message"),
        [
            ("gemini-3.1-flash-image", {"n": 2}, "one image per request"),
            ("gemini-3.1-flash-image", {"quality": "high"}, "takes no quality"),
            ("gemini-3.1-flash-image", {"output_format": "png"}, "takes no output_format"),
            ("gemini-3.1-flash-lite-image", {"aspect_ratio": "4:1"}, "aspect_ratio in"),
            ("gemini-3-pro-image", {"resolution": "512"}, "resolution in"),
            ("gemini-3.8-flash", {}, "not an image model"),
        ],
    )
    def test_what_the_model_does_not_take_is_refused(self, model, options, message) -> None:
        with pytest.raises(RequestError, match=message):
            prepare_image(self.provider(model), "a lighthouse", **options)

    def test_an_image_model_takes_no_tools(self) -> None:
        tool = {"name": "f", "description": "", "parameters": {"type": "object"}}
        with pytest.raises(RequestError, match="takes no tools"):
            prepare(self.provider(), [user("draw")], tools=[tool])

    def test_the_final_image_comes_back_and_the_thought_images_do_not(self) -> None:
        answer = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(data=WEBP, mime_type="image/webp"),
                                thought=True,
                            ),
                            types.Part(text="Here it is."),
                            types.Part(
                                inline_data=types.Blob(data=PNG, mime_type="image/png"),
                                thought_signature=b"sig",
                            ),
                        ],
                    ),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=12,
                candidates_token_count=1130,
                thoughts_token_count=50,
                candidates_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.IMAGE, token_count=1120),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=10),
                ],
            ),
        )
        response = assembled(self.provider(), answer)
        assert response.text == "Here it is."
        assert response.images == (GeneratedImage(data=PNG, media_type="image/png"),)
        assert response.usage == Usage(input_tokens=12, output_tokens=60, image_output_tokens=1120)
        assert response.cost == pytest.approx((12 * 0.5 + 60 * 3.0 + 1120 * 60.0) / 1e6)

    def test_a_data_url_image_goes_inline_not_as_a_file_uri(self) -> None:
        url = f"data:image/png;base64,{PNG_B64}"
        prepared = prepare(
            self.provider("gemini-3.8-flash"), [user(["what is this?", image(url)])]
        )
        part = prepared.params["contents"][0].parts[1]
        assert part.file_data is None
        assert part.inline_data.data == PNG


# ── xAI ──────────────────────────────────────────────────────────────────────


def _xai_image(data: bytes = PNG, *, moderated: bool = True, cost: float | None = 0.04) -> Any:
    return SimpleNamespace(
        base64=f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}",
        respect_moderation=moderated,
        model="grok-imagine-image-2.0",
        cost_usd=cost,
        usage=usage_pb2.SamplingUsage(prompt_tokens=12),
    )


class TestXAI:
    @staticmethod
    def provider(model: str = "grok-imagine-image-2.0", *answers: Any) -> XAIProvider:
        provider = XAIProvider(model, "test-key")
        sample = AsyncMock(return_value=answers[0] if answers else _xai_image())
        batch = AsyncMock(return_value=list(answers))
        provider._client = SimpleNamespace(
            image=SimpleNamespace(sample=sample, sample_batch=batch)
        )
        return provider

    async def test_the_request_takes_the_sdk_values(self) -> None:
        call = prepare_image(
            self.provider(),
            "a lighthouse",
            image(PNG),
            aspect_ratio="16:9",
            resolution="2K",
            quality="low",
        )
        assert call.n == 1
        assert call.params == {
            "prompt": "a lighthouse",
            "model": "grok-imagine-image-2.0",
            "image_format": "base64",
            "aspect_ratio": "16:9",
            "resolution": "2k",
            "quality": "low",
            "image_url": f"data:image/png;base64,{PNG_B64}",
        }

    async def test_several_input_images_go_as_a_list(self) -> None:
        call = prepare_image(self.provider(), "merge", image(PNG), image("https://x.ai/a.png"))
        assert call.params["image_urls"] == [
            f"data:image/png;base64,{PNG_B64}",
            "https://x.ai/a.png",
        ]

    @pytest.mark.parametrize(
        ("model", "options", "message"),
        [
            ("grok-imagine-image-2.0", {"resolution": "4K"}, "'1K' or '2K'"),
            ("grok-imagine-image-2.0", {"quality": "high"}, "quality in"),
            ("grok-imagine-image-2.0", {"aspect_ratio": "21:9"}, "aspect_ratio in"),
            ("grok-imagine-image-2.0", {"output_format": "png"}, "no output_format"),
            ("grok-4.7", {}, "not an image model"),
        ],
    )
    async def test_what_the_model_does_not_take_is_refused(self, model, options, message) -> None:
        with pytest.raises(RequestError, match=message):
            prepare_image(self.provider(model), "a lighthouse", **options)

    async def test_an_image_model_takes_no_chat(self) -> None:
        with pytest.raises(RequestError, match="is an image model"):
            prepare(self.provider(), [user("hi")])

    async def test_the_reported_cost_wins(self) -> None:
        provider = self.provider()
        response = await generate_image(provider, "a lighthouse")
        assert response.images == (GeneratedImage(data=PNG, media_type="image/png"),)
        assert response.cost == pytest.approx(0.04)
        assert response.usage == Usage(input_tokens=12, image_count=1)
        provider._client.image.sample.assert_awaited_once()

    async def test_more_than_one_image_is_a_batch(self) -> None:
        provider = self.provider("grok-imagine-image-2.0", _xai_image(), _xai_image())
        response = await generate_image(provider, "a lighthouse", n=2)
        assert len(response.images) == 2
        assert response.usage.image_count == 2
        assert provider._client.image.sample_batch.call_args.kwargs["n"] == 2

    async def test_a_withheld_image_is_an_error(self) -> None:
        provider = self.provider("grok-imagine-image-2.0", _xai_image(moderated=False))
        with pytest.raises(ResponseError, match="moderation"):
            await generate_image(provider, "a lighthouse")


def test_every_image_model_has_a_price() -> None:
    for model in (
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-flare",
        "gpt-image-2",
        "gpt-image-1.5",
        "chatgpt-image-latest",
        "gpt-image-1",
        "gpt-image-1-mini",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image",
        "grok-imagine-image-2.0",
        "grok-imagine-image",
        "grok-imagine-image-quality",
        "muse-image-1.0",
    ):
        assert pricing.get(model) is not None, model
