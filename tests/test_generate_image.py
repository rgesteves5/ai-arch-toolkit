"""``LLM.generate_image``: the image generation call, metered on the ``complete`` path (I02)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ai_arch_toolkit.core import (
    GeneratedImage,
    ImageRequest,
    RequestError,
    Response,
    Usage,
    image,
)
from ai_arch_toolkit.core._attempts import _request_size
from ai_arch_toolkit.core._content import user
from ai_arch_toolkit.core._exceptions import TransportError
from ai_arch_toolkit.core._metering._scope import MeterScope, RunConfig
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._pricing import ModelPricing, pricing
from ai_arch_toolkit.core._retry import RetryConfig
from ai_arch_toolkit.toolkit.budget import BudgetController, BudgetExceeded, BudgetPolicy
from tests.fake_provider import FakeProvider, fake_llm

IMAGE_MODEL = "gpt-image-test"  # routed to an adapter, priced below
PER_IMAGE_MODEL = "gpt-image-per-image-test"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture(autouse=True)
def image_prices() -> Iterator[None]:
    pricing.register(
        IMAGE_MODEL,
        ModelPricing(input=5.0, output=10.0, image_input=8.0, image_output=30.0),
    )
    pricing.register(PER_IMAGE_MODEL, ModelPricing(per_image=0.01))
    yield
    pricing.unregister(IMAGE_MODEL)
    pricing.unregister(PER_IMAGE_MODEL)


def drawn(count: int = 1, **usage: int) -> Response:
    images = tuple(GeneratedImage(data=PNG, media_type="image/png") for _ in range(count))
    return Response(images=images, usage=Usage(**usage), model=IMAGE_MODEL)


# ── the call ─────────────────────────────────────────────────────────────────


async def test_the_images_come_back_with_the_request_the_adapter_saw() -> None:
    llm, provider = fake_llm(drawn(2), model=IMAGE_MODEL, images=True, temperature=0.5)
    source = image(PNG)
    out = await llm.generate_image(
        "a lighthouse", images=[source], n=2, aspect_ratio="16:9", quality="low"
    )
    assert [i.media_type for i in out.images] == ["image/png", "image/png"]
    assert out.images[0].data == PNG
    assert bool(out) and out.text == ""
    sent = provider.last
    assert sent.image == ImageRequest(n=2, aspect_ratio="16:9", quality="low")
    assert sent.messages == [user(["a lighthouse", source])]
    # An image generation carries none of the LLM's defaults or the call options.
    assert sent.kwargs == {} and sent.system is None and sent.tools is None


def test_generate_image_sync_answers_too() -> None:
    llm, _ = fake_llm(drawn(), model=IMAGE_MODEL, images=True)
    assert len(llm.generate_image_sync("a lighthouse").images) == 1


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ({"prompt": "  "}, "prompt must be a non-empty string"),
        ({"images": [PNG]}, "images must be a sequence of image"),
        ({"images": image(PNG)}, "images must be a sequence of image"),
        ({"n": 0}, "n must be a positive integer"),
        ({"n": True}, "n must be a positive integer"),
        ({"aspect_ratio": "wide"}, "aspect_ratio must be 'W:H'"),
        ({"aspect_ratio": "0:9"}, "aspect_ratio must be 'W:H'"),
        ({"resolution": "8K"}, "resolution must be one of"),
        ({"quality": ""}, "quality must be a non-empty string"),
        ({"output_format": "gif"}, "output_format must be one of"),
    ],
)
async def test_bad_options_are_refused_before_anything_is_sent(call, message) -> None:
    llm, provider = fake_llm(drawn(), model=IMAGE_MODEL, images=True)
    arguments = {"prompt": "a lighthouse", **call}
    with pytest.raises(RequestError, match=message):
        await llm.generate_image(**arguments)
    assert provider.calls == 0


async def test_an_adapter_without_image_models_refuses_before_the_meter() -> None:
    llm, provider = fake_llm(drawn(), model=IMAGE_MODEL)
    with MeterScope() as scope, pytest.raises(RequestError, match="does not generate images"):
        await llm.generate_image("a lighthouse")
    assert provider.calls == 0
    snap = scope.snapshot()
    assert snap.llm_calls == 0 and snap.out_llm_calls == 0


# ── the same path as complete ────────────────────────────────────────────────


async def test_middleware_sees_the_image_request_and_may_change_it() -> None:
    class MoreImages:
        def before(self, request: Request) -> Request:
            assert request.image is not None
            return Request(
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                model=request.model,
                image=ImageRequest(n=3),
            )

        def after(self, request: Request, response: Response) -> Response:
            return response

    llm, provider = fake_llm(drawn(3), model=IMAGE_MODEL, images=True, middleware=[MoreImages()])
    out = await llm.generate_image("a lighthouse")
    assert provider.last.image == ImageRequest(n=3)
    assert len(out.images) == 3


async def test_a_retried_image_generation_is_charged_once() -> None:
    lost = TransportError("connection refused", delivery="not_sent")
    llm, provider = fake_llm(
        lost,
        drawn(image_output_tokens=196),
        model=IMAGE_MODEL,
        images=True,
        retry=RetryConfig(max_retries=1, base_delay=0.001),
    )
    with MeterScope() as scope:
        out = await llm.generate_image("a lighthouse")
    assert provider.calls == 2 and len(out.images) == 1
    assert [a.status for a in out.attempts] == ["failed", "ok"]
    snap = scope.snapshot()
    assert snap.cost.to_float() == pytest.approx(196 * 30.0 / 1_000_000)


async def test_a_fallback_image_model_gets_the_same_image_request() -> None:
    llm, primary = fake_llm(
        TransportError("down", delivery="not_sent"), model=IMAGE_MODEL, images=True
    )
    backup, secondary = fake_llm(drawn(), model=IMAGE_MODEL, images=True, temperature=0.9)
    llm._fallbacks = [backup]
    out = await llm.generate_image("a lighthouse", n=1, quality="low")
    assert primary.calls == 1 and secondary.calls == 1
    assert secondary.last.image == ImageRequest(quality="low")
    assert secondary.last.kwargs == {}
    assert len(out.images) == 1


# ── cost and budget ──────────────────────────────────────────────────────────


async def test_image_tokens_are_charged_at_the_image_rates() -> None:
    llm, _ = fake_llm(
        drawn(input_tokens=28, image_input_tokens=1024, image_output_tokens=196),
        model=IMAGE_MODEL,
        images=True,
    )
    with MeterScope() as scope:
        out = await llm.generate_image("make the sky purple", images=[image(PNG)])
    expected = (28 * 5.0 + 1024 * 8.0 + 196 * 30.0) / 1_000_000
    assert out.cost == pytest.approx(expected)
    snap = scope.snapshot()
    assert snap.cost.to_float() == pytest.approx(expected)
    # Image tokens count against the token caps like any other.
    assert snap.input_tokens == 28 + 1024 and snap.output_tokens == 196
    assert out.tokens == 28 + 1024 + 196


async def test_a_per_image_price_charges_each_image() -> None:
    response = Response(
        images=(GeneratedImage(data=PNG, media_type="image/webp"),) * 2,
        usage=Usage(input_tokens=9990, output_tokens=916, image_count=2),
        model=PER_IMAGE_MODEL,
    )
    llm, _ = fake_llm(response, model=PER_IMAGE_MODEL, images=True)
    with MeterScope() as scope:
        out = await llm.generate_image("a lighthouse", n=2)
    assert out.cost == pytest.approx(0.02)
    assert scope.snapshot().cost.to_float() == pytest.approx(0.02)


def _strict(max_cost: float) -> MeterScope:
    policy = BudgetPolicy(max_cost=max_cost, reserve="strict")
    return MeterScope(RunConfig(controller=BudgetController(policy)))


async def test_a_strict_budget_reserves_every_image_asked_for() -> None:
    # Two images reserve 2 x 16,000 image output tokens at $30/M: $0.96, plus the prompt.
    llm, provider = fake_llm(drawn(2, image_output_tokens=392), model=IMAGE_MODEL, images=True)
    with _strict(0.90), pytest.raises(BudgetExceeded):
        await llm.generate_image("a lighthouse", n=2)
    assert provider.calls == 0
    with _strict(1.00) as scope:
        out = await llm.generate_image("a lighthouse", n=2)
    assert len(out.images) == 2
    assert scope.snapshot().cost.to_float() == pytest.approx(392 * 30.0 / 1_000_000)


async def test_a_strict_budget_reserves_the_per_image_price() -> None:
    response = Response(usage=Usage(image_count=3), model=PER_IMAGE_MODEL)
    llm, provider = fake_llm(response, model=PER_IMAGE_MODEL, images=True)
    with _strict(0.025), pytest.raises(BudgetExceeded):
        await llm.generate_image("a lighthouse", n=3)
    assert provider.calls == 0


# ── request size: images are parts, not text ────────────────────────────────


def test_an_image_counts_as_a_part_and_not_by_its_bytes() -> None:
    big = bytes(range(256)) * 4000  # about 1 MB
    with_image = Request(
        messages=[user(["describe this", image(big)])], system=None, tools=None, model="m"
    )
    text_only = Request(messages=[user(["describe this"])], system=None, tools=None, model="m")
    chars, parts = _request_size(with_image)
    assert parts == 1
    assert chars == _request_size(text_only)[0]


def test_a_dict_media_part_still_counts() -> None:
    message = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    request = Request(messages=[message], system=None, tools=None, model="m")
    assert _request_size(request)[1] == 1


# ── the types ────────────────────────────────────────────────────────────────


def test_usage_adds_every_counter() -> None:
    total = Usage(input_tokens=1, image_output_tokens=2, image_count=1) + Usage(
        output_tokens=3, image_input_tokens=4, image_count=2, cache_read_tokens=5
    )
    assert total == Usage(
        input_tokens=1,
        output_tokens=3,
        cache_read_tokens=5,
        image_input_tokens=4,
        image_output_tokens=2,
        image_count=3,
    )
    assert sum([Usage(input_tokens=1)] * 3, Usage()) == Usage(input_tokens=3)


def test_a_response_with_images_only_is_truthy_and_says_so() -> None:
    response = drawn(2)
    assert bool(response)
    assert repr(response) == "Response(text='', images=2)"
    assert repr(response.images[0]) == f"GeneratedImage(image/png, {len(PNG)} bytes)"


def test_the_image_request_reads_its_ratio() -> None:
    assert ImageRequest(aspect_ratio="16:9").ratio == pytest.approx(16 / 9)
    assert ImageRequest(aspect_ratio="19.5:9").ratio == pytest.approx(19.5 / 9)
    assert ImageRequest().ratio is None


def test_the_fake_provider_refuses_images_unless_asked() -> None:
    request = Request(messages=[], system=None, tools=None, model="m", image=ImageRequest())
    with pytest.raises(RequestError, match="does not generate images"):
        FakeProvider().prepare_image(request)
    assert FakeProvider(images=True).prepare_image(request) is request
