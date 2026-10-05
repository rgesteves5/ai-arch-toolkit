"""The most image output tokens one image costs, by model, quality and size (G-40).

The counts are the providers' published ones: OpenAI's table for the models before gpt-image-2
and its calculator for gpt-image-2 and later, Gemini's tokens per image size. Where the app
measured a count live (ai-network, 2026-10-04), the case says so.
"""

from __future__ import annotations

import pytest

from ai_arch_toolkit.core._images import ImageRequest
from ai_arch_toolkit.core._metering._operation import OperationRequest
from ai_arch_toolkit.core._metering._worst_case import IMAGE_OUTPUT_TOKEN_ALLOWANCE, worst_case
from ai_arch_toolkit.core._pricing import pricing
from ai_arch_toolkit.core._providers._gemini import GeminiProvider
from ai_arch_toolkit.core._providers._meta import MetaProvider
from ai_arch_toolkit.core._providers._openai import OpenAIProvider
from ai_arch_toolkit.core._providers._openai_images import TileTokens
from ai_arch_toolkit.core._providers._xai import XAIProvider
from ai_arch_toolkit.core._response import Usage


def openai(model: str, **options: object) -> int | None:
    return OpenAIProvider(model, "test-key").image_token_bound(ImageRequest(**options))  # type: ignore[arg-type]


@pytest.mark.parametrize("model", ["gpt-image-1", "gpt-image-1.5", "chatgpt-image-latest"])
@pytest.mark.parametrize(
    ("options", "tokens"),
    [
        ({"quality": "low", "aspect_ratio": "1:1"}, 272),
        ({"quality": "low", "aspect_ratio": "3:2"}, 400),
        ({"quality": "medium", "aspect_ratio": "2:3"}, 1584),
        ({"quality": "high", "aspect_ratio": "2:3"}, 6240),
        ({"quality": "high"}, 6240),  # the model's own size: the largest
        ({"aspect_ratio": "1:1"}, 4160),  # no quality: the dearest
        ({"quality": "auto", "aspect_ratio": "3:2"}, 6208),  # "auto" has no published count
        ({}, 6240),
    ],
)
def test_the_older_models_follow_openais_table(
    model: str, options: dict[str, object], tokens: int
) -> None:
    assert openai(model, **options) == tokens


def test_gpt_image_1_mini_follows_its_prices() -> None:
    assert openai("gpt-image-1-mini", quality="low", aspect_ratio="1:1") == 687
    assert openai("gpt-image-1-mini", quality="high", aspect_ratio="2:3") == 6562
    assert openai("gpt-image-1-mini") == 6562


@pytest.mark.parametrize(
    ("model", "options", "tokens"),
    [
        # OpenAI's calculator's default, and the live count of 2026-10-03
        ("gpt-image-2.5-flare", {"quality": "low", "aspect_ratio": "1:1"}, 196),
        ("gpt-image-2", {"quality": "low", "aspect_ratio": "1:1"}, 196),
        # measured by the app at 2:3 and 1K (832x1248)
        ("gpt-image-2.5-flare", {"quality": "medium", "aspect_ratio": "2:3"}, 292),
        ("gpt-image-2", {"quality": "medium", "aspect_ratio": "2:3"}, 1167),
        # the guide's gpt-image-2 table: $0.211 at high 1024x1024 is 7,024 tokens at $30/M
        ("gpt-image-2", {"quality": "high", "aspect_ratio": "1:1"}, 7024),
        ("gpt-image-2.5-sunburst", {"quality": "max", "aspect_ratio": "1:1"}, 7024),
        (
            "gpt-image-2.5-sunburst",
            {"quality": "xhigh", "aspect_ratio": "16:9", "resolution": "4K"},
            5930,
        ),
        # the model's own size: the largest square, 2880x2880
        ("gpt-image-2.5-flare", {"quality": "high"}, 5930),
        ("gpt-image-2", {"quality": "high"}, 23_719),
        ("gpt-image-2.5-flare", {}, 23_719),  # no quality, no size: max at 2880x2880
        ("gpt-image-9", {"quality": "low"}, 659),  # an unlisted gpt-image- id: the newest rules
    ],
)
def test_gpt_image_2_and_later_follow_openais_calculator(
    model: str, options: dict[str, object], tokens: int
) -> None:
    assert openai(model, **options) == tokens


def test_the_largest_published_image_fits_the_allowance() -> None:
    assert openai("gpt-image-2", quality="high") <= IMAGE_OUTPUT_TOKEN_ALLOWANCE  # type: ignore[operator]


@pytest.mark.parametrize(
    ("model", "resolution", "tokens"),
    [
        ("gemini-3.1-flash-image", "512", 747),
        ("gemini-3.1-flash-image", "4K", 2520),
        ("gemini-3.1-flash-image", None, 2520),
        ("gemini-3.1-flash-lite-image", None, 1120),
        ("gemini-3-pro-image", "2K", 1120),
        ("gemini-3-pro-image", None, 2000),
    ],
)
def test_gemini_counts_by_image_size(model: str, resolution: str | None, tokens: int) -> None:
    image = ImageRequest(resolution=resolution)  # type: ignore[arg-type]
    assert GeminiProvider(model, "test-key").image_token_bound(image) == tokens


async def test_a_model_billed_per_image_holds_no_image_tokens() -> None:
    assert MetaProvider("muse-image-1.0", "test-key").image_token_bound(ImageRequest()) == 0
    async with XAIProvider("grok-imagine-image-2.0", "test-key") as xai:
        assert xai.image_token_bound(ImageRequest()) == 0


async def test_a_model_that_does_not_draw_publishes_no_count() -> None:
    assert openai("gpt-6-luna") is None
    assert MetaProvider("muse-spark-1.3", "test-key").image_token_bound(ImageRequest()) is None
    async with XAIProvider("grok-4.7", "test-key") as xai:
        assert xai.image_token_bound(ImageRequest()) is None
    gemini = GeminiProvider("gemini-3-flash-preview", "test-key")
    assert gemini.image_token_bound(ImageRequest()) is None
    assert gemini.image_text_token_bound(ImageRequest()) is None


def test_a_tile_count_on_a_half_goes_to_the_even_tile() -> None:
    # 16 tiles over 1024/672 is 10.5: the calculator takes 10, so 160 tiles.
    assert TileTokens({"low": 16}).count("low", "1024x672") == -(-160 * 2_688_128 // 4_000_000)


@pytest.mark.parametrize(
    ("model", "limit"),
    [
        ("gemini-3.1-flash-image", 32_768),
        ("gemini-3-pro-image", 32_768),
        ("gemini-3.1-flash-lite-image", 4_096),
    ],
)
def test_a_gemini_image_model_bounds_its_thinking_by_its_output_limit(
    model: str, limit: int
) -> None:
    assert GeminiProvider(model, "test-key").image_text_token_bound(ImageRequest()) == limit


def test_the_worst_case_of_a_thinking_image_model_covers_its_thinking() -> None:
    # Gemini 3 Pro Image thinks unasked: 1,500 thought tokens beside a 1K image stay within.
    facts = OperationRequest(
        kind="llm",
        parent_span_id="run",
        model="gemini-3-pro-image",
        declared_max_output_tokens=32_768,
        declared_images=1,
        declared_image_tokens=1120,
        content_size_hint=40,
    )
    held = worst_case(facts, pricing)
    billed = pricing.price(
        facts, Usage(input_tokens=10, output_tokens=1500, image_output_tokens=1120, image_count=1)
    )

    assert held is not None and billed.amount is not None
    assert billed.amount <= held.cost


def test_the_worst_case_holds_the_published_count_per_image() -> None:
    facts = OperationRequest(
        kind="llm",
        parent_span_id="run",
        model="gpt-image-2.5-flare",
        declared_images=2,
        declared_image_tokens=196,
    )
    held = worst_case(facts, pricing)
    unknown = worst_case(
        OperationRequest(
            kind="llm", parent_span_id="run", model="gpt-image-2.5-flare", declared_images=2
        ),
        pricing,
    )

    assert held is not None and unknown is not None
    assert held.output_tokens == 2 * 196
    assert unknown.output_tokens == 2 * IMAGE_OUTPUT_TOKEN_ALLOWANCE
    assert held.cost < unknown.cost
