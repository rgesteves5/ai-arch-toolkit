"""Image generation through each adapter and its real SDK, against a loopback server (I03)."""

from __future__ import annotations

import base64
import json

import pytest
from google import genai
from xai_sdk.proto import image_pb2, usage_pb2

from ai_arch_toolkit.core import image
from ai_arch_toolkit.core._providers._gemini import GeminiProvider, _http_options
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from tests.integration import fakegrpc, fakeserver
from tests.provider_calls import generate_image

pytestmark = pytest.mark.integration

PNG = b"\x89PNG\r\n\x1a\n" + b"\x01" * 48
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x02" * 24


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


OPENAI_IMAGES = {
    "created": 1,
    "data": [{"b64_json": _b64(PNG)}],
    "output_format": "png",
    "size": "1360x768",
    "usage": {
        "input_tokens": 21,
        "input_tokens_details": {"text_tokens": 21, "image_tokens": 0},
        "output_tokens": 120,
        "output_tokens_details": {"text_tokens": 0, "image_tokens": 120},
        "total_tokens": 141,
    },
}


def _local(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


async def test_openai_generates_through_the_images_api() -> None:
    server, port, stats = await fakeserver.start("status", body=OPENAI_IMAGES)
    async with server:
        provider = OpenAIProvider("gpt-image-2.5-flare", "local-test", base_url=_local(port))
        response = await generate_image(provider, "a lighthouse", aspect_ratio="16:9")
        await provider.close()
    sent = json.loads(stats.bodies[0])
    assert sent == {"model": "gpt-image-2.5-flare", "prompt": "a lighthouse", "size": "1360x768"}
    assert response.images[0].data == PNG and response.images[0].media_type == "image/png"
    assert response.usage.image_output_tokens == 120
    assert response.cost == pytest.approx((21 * 5.0 + 120 * 30.0) / 1e6)


async def test_openai_edits_with_the_image_file_in_a_multipart_body() -> None:
    server, port, stats = await fakeserver.start("status", body=OPENAI_IMAGES)
    async with server:
        provider = OpenAIProvider("gpt-image-2.5-flare", "local-test", base_url=_local(port))
        response = await generate_image(provider, "make the sky purple", image(PNG))
        await provider.close()
    body = stats.bodies[0]
    assert b'name="prompt"' in body and b"make the sky purple" in body
    assert b'name="image' in body and PNG in body
    assert len(response.images) == 1


async def test_meta_generates_with_a_flat_price_per_image() -> None:
    answer = {
        "created": 1,
        "data": [{"b64_json": _b64(WEBP)}],
        "output_format": "webp",
        "usage": {"input_tokens": 10025, "output_tokens": 945, "total_tokens": 10970},
    }
    server, port, stats = await fakeserver.start("status", body=answer)
    async with server:
        provider = MetaProvider("muse-image-1.0", "local-test", base_url=_local(port))
        response = await generate_image(provider, "a lighthouse", aspect_ratio="1:1")
        await provider.close()
    assert json.loads(stats.bodies[0])["size"] == "1024x1024"
    assert response.images[0].media_type == "image/webp"
    assert response.cost == pytest.approx(0.01)


async def test_gemini_asks_for_image_output_and_reads_the_inline_image() -> None:
    answer = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "inlineData": {"mimeType": "image/png", "data": _b64(PNG)},
                            "thought": True,
                        },
                        {"inlineData": {"mimeType": "image/png", "data": _b64(PNG)}},
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 1120,
            "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}],
        },
    }
    server, port, stats = await fakeserver.start("status", body=answer)
    async with server:
        provider = GeminiProvider("gemini-3.1-flash-image", "local-test", timeout=2.0)
        await provider.close()  # its client for Google never connected
        options = _http_options(2.0).model_copy(update={"base_url": f"http://127.0.0.1:{port}"})
        provider._install_client(lambda: genai.Client(api_key="local-test", http_options=options))
        response = await generate_image(provider, "a lighthouse", aspect_ratio="16:9")
        await provider.close()
    sent = json.loads(stats.bodies[0])
    assert sent["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert sent["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"}
    assert len(response.images) == 1  # the thought image is not the answer
    assert response.usage.image_output_tokens == 1120
    assert response.cost == pytest.approx((9 * 0.5 + 1120 * 60.0) / 1e6)


async def test_xai_samples_and_takes_the_reported_cost() -> None:
    script = fakegrpc.Script(
        images=image_pb2.ImageResponse(
            images=[
                image_pb2.GeneratedImage(
                    base64=f"data:image/png;base64,{_b64(PNG)}", respect_moderation=True
                )
            ],
            model="grok-imagine-image-2.0",
            usage=usage_pb2.SamplingUsage(prompt_tokens=12, cost_in_usd_ticks=400_000_000),
        )
    )
    async with fakegrpc.serving(script) as (_, port):
        provider = await fakegrpc.provider("grok-imagine-image-2.0", port)
        response = await generate_image(
            provider, "a lighthouse", image(PNG), aspect_ratio="16:9", quality="low"
        )
        await provider.close()
    sent = script.image_requests[0]
    assert sent.prompt == "a lighthouse" and sent.model == "grok-imagine-image-2.0"
    assert sent.image.image_url == f"data:image/png;base64,{_b64(PNG)}"
    assert response.images[0].data == PNG
    assert response.cost == pytest.approx(0.04)
