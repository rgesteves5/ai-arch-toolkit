"""The Images API of the ``openai`` SDK, shared by OpenAI's and Meta's adapters (D46, D47).

``POST /v1/images/generations`` draws from a prompt, ``POST /v1/images/edits`` from a prompt and
input images (https://developers.openai.com/api/reference/resources/images). Meta serves both for
``muse-image-1.0`` (https://dev.meta.ai/docs/image-generation). Each adapter keeps its image
models' rules as :class:`ImageModel` entries; this module turns the portable options into the
SDK's request and reads the answer.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Literal, cast

from openai.resources.images import AsyncImages
from openai.types import ImagesResponse
from openai.types.image_edit_params import ImageEditParamsNonStreaming
from openai.types.image_generate_params import ImageGenerateParamsNonStreaming

from ai_arch_toolkit.core._exceptions import RequestError
from ai_arch_toolkit.core._images import ImageRequest
from ai_arch_toolkit.core._middleware import Request
from ai_arch_toolkit.core._providers._base import image_bytes, image_media_type, image_prompt
from ai_arch_toolkit.core._response import GeneratedImage, Response, Usage

type SizeRule = Literal["fixed", "free", "ratio"]

# The three sizes of the GPT Image models before gpt-image-2, by aspect ratio; live,
# gpt-image-1.5 answers anything else with "Supported sizes are 1024x1024, 1024x1536, 1536x1024,
# and auto." (I01 check 5).
_FIXED_SIZES = {1.0: "1024x1024", 2 / 3: "1024x1536", 3 / 2: "1536x1024"}

# gpt-image-2 and later take any WIDTHxHEIGHT: "Width and height must both be divisible by 16 and
# the requested aspect ratio must be between 1:3 and 3:1 ... the maximum supported resolution is
# 3840x2160", with a total of 655,360 to 8,294,400 pixels
# (https://developers.openai.com/api/docs/guides/image-generation).
_STEP = 16
_MAX_EDGE = 3840
_MIN_PIXELS = 655_360
_MAX_PIXELS = 8_294_400
_MAX_RATIO = 3.0

# A resolution class is the area of a square with that edge, as Gemini sizes its "1K", "2K" and
# "4K" images (https://ai.google.dev/gemini-api/docs/generate-content/image-generation).
_EDGES = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}

type ImagesParams = ImageGenerateParamsNonStreaming | ImageEditParamsNonStreaming


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageModel:
    """What one image model takes on the Images API.

    Attributes:
        size: ``"fixed"`` for the three sizes of the older GPT Image models, ``"free"`` for any
            size within gpt-image-2's limits, ``"ratio"`` for a model that reads only the aspect
            ratio of the size (Meta).
        qualities: The quality levels it takes; empty when it takes none.
        max_images: The most images one request returns (``n``).
        max_inputs: The most input images an edit takes.
    """

    size: SizeRule
    qualities: frozenset[str] = frozenset()
    max_images: int = 10
    max_inputs: int = 16


@dataclass(frozen=True, slots=True)
class ImagesCall:
    """An Images API request: an edit when it carries input images, else a generation."""

    params: ImagesParams
    edit: bool


def _rounded(value: float) -> int:
    return max(_STEP, round(value / _STEP) * _STEP)


def _free_size(ratio: float, resolution: str) -> str:
    """A WIDTHxHEIGHT of ``ratio`` with the area of the resolution class, within the limits."""
    if not 1 / _MAX_RATIO <= ratio <= _MAX_RATIO:
        raise RequestError("the aspect ratio must be between 1:3 and 3:1 on this model")
    pixels = min(_EDGES[resolution] ** 2, _MAX_PIXELS)
    width, height = math.sqrt(pixels * ratio), math.sqrt(pixels / ratio)
    scale = min(1.0, _MAX_EDGE / max(width, height))
    w, h = _rounded(width * scale), _rounded(height * scale)
    while w * h > _MAX_PIXELS:  # rounding up can step past the cap
        w, h = (w - _STEP, h) if w >= h else (w, h - _STEP)
    if w * h < _MIN_PIXELS:
        raise RequestError(
            f"resolution {resolution!r} gives {w}x{h}, below this model's {_MIN_PIXELS:,} pixels"
        )
    return f"{w}x{h}"


def _fixed_size(ratio: float, resolution: str | None) -> str:
    if resolution not in (None, "1K"):
        raise RequestError("this model draws at 1K only: resolution must be '1K' or left out")
    for known, size in _FIXED_SIZES.items():
        if math.isclose(ratio, known, rel_tol=1e-3):
            return size
    raise RequestError("this model takes the aspect ratios 1:1, 2:3 and 3:2 only")


def image_size(options: ImageRequest, rule: SizeRule) -> str | None:
    """The ``size`` for the request's aspect ratio and resolution, or ``None`` for the model's
    own choice (neither asked)."""
    if options.aspect_ratio is None and options.resolution is None:
        return None
    ratio = options.ratio or 1.0
    if rule == "fixed":
        return _fixed_size(ratio, options.resolution)
    if rule == "ratio":
        if options.resolution is not None:
            raise RequestError("this model takes no resolution, only an aspect ratio")
        return _free_size(ratio, "1K")
    return _free_size(ratio, options.resolution or "1K")


def images_call(model: str, request: Request, rules: ImageModel) -> ImagesCall:
    """The Images API request for an image generation, checked against the model's rules."""
    options = request.image
    assert options is not None  # prepare_image is only called for image generations
    prompt, inputs = image_prompt(request)
    if options.n > rules.max_images:
        raise RequestError(f"{model} returns at most {rules.max_images} images, not {options.n}")
    if len(inputs) > rules.max_inputs:
        raise RequestError(f"{model} edits at most {rules.max_inputs} input images")
    if options.quality is not None and options.quality not in rules.qualities:
        takes = f"quality in {sorted(rules.qualities)}" if rules.qualities else "no quality"
        raise RequestError(f"{model} takes {takes}, not {options.quality!r}")
    params: ImageGenerateParamsNonStreaming = {"model": model, "prompt": prompt}
    if options.n != 1:
        params["n"] = options.n
    if (size := image_size(options, rules.size)) is not None:
        params["size"] = size
    if options.quality is not None:
        params["quality"] = cast("Literal['low']", options.quality)
    if options.output_format is not None:
        params["output_format"] = options.output_format
    if not inputs:
        return ImagesCall(params, edit=False)
    files = [
        (f"image-{index}", image_bytes(part), part.media_type) for index, part in enumerate(inputs)
    ]
    # One image goes as one file: the SDK names a list's files "image[]", which Meta refuses
    # ("use image[N] with indices numbered consecutively from 0"; live, I03).
    edit = cast(
        "ImageEditParamsNonStreaming", {**params, "image": files[0] if len(files) == 1 else files}
    )
    return ImagesCall(edit, edit=True)


async def send_images(images: AsyncImages, call: ImagesCall) -> ImagesResponse:
    if call.edit:
        return await images.edit(**cast("ImageEditParamsNonStreaming", call.params))
    return await images.generate(**cast("ImageGenerateParamsNonStreaming", call.params))


def images_response(final: ImagesResponse, model: str) -> Response:
    """The ``Response`` for an Images API answer: its images, as bytes with their type."""
    declared = f"image/{final.output_format}" if final.output_format else None
    images = []
    for item in final.data or []:
        if not item.b64_json:
            continue
        data = base64.b64decode(item.b64_json)
        images.append(
            GeneratedImage(
                data=data,
                media_type=image_media_type(data, declared),
                revised_prompt=item.revised_prompt or "",
            )
        )
    return Response(images=tuple(images), model=model, raw=final, stop_reason="completed")


def images_usage(final: ImagesResponse) -> Usage:
    """The usage of an Images API answer.

    The text and image tokens of each side come apart when the provider details them (OpenAI,
    I01 check 1); undetailed output tokens of an image model are its image's. ``image_count``
    is there for a provider that bills per image (Meta).
    """
    count = sum(1 for item in final.data or [] if item.b64_json)
    usage = final.usage
    if usage is None:
        return Usage(image_count=count)
    # Meta's answer leaves the details out (I01 check 7), although the SDK types them as required:
    # a model built from such a body has no such attribute.
    inputs = getattr(usage, "input_tokens_details", None)
    outputs = getattr(usage, "output_tokens_details", None)
    return Usage(
        input_tokens=inputs.text_tokens if inputs else usage.input_tokens,
        image_input_tokens=inputs.image_tokens if inputs else 0,
        output_tokens=outputs.text_tokens if outputs else 0,
        image_output_tokens=outputs.image_tokens if outputs else usage.output_tokens,
        image_count=count,
    )
