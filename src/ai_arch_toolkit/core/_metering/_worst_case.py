"""The most an operation can cost, from its facts alone (D49).

It is the meter's ceiling for a failure the provider may have billed, and the default reservation
of a strict budget. Its rules are bounds, not estimates: a token takes at least
:data:`CHARS_PER_TOKEN` characters, and each image or document part, and each image an image
generation asks for, is held at a generous allowance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ai_arch_toolkit.core._metering._admission import Reservation
from ai_arch_toolkit.core._response import Usage

if TYPE_CHECKING:
    from ai_arch_toolkit.core._metering._operation import OperationRequest
    from ai_arch_toolkit.core._metering._scope import Pricer

__all__ = ["worst_case"]

CHARS_PER_TOKEN = 4  # rough English-text ratio; deliberately conservative
MEDIA_TOKEN_ALLOWANCE = 4000  # input tokens held per image or document part
# Image output tokens held per image asked for when the adapter knows no count for the model, its
# quality and its size (G-40). The dearest published image is 23,719 tokens: a 2880x2880 one on
# gpt-image-2 at "high", or on a GPT Image 2.5 model at "max".
IMAGE_OUTPUT_TOKEN_ALLOWANCE = 24_000


def worst_case(request: OperationRequest, pricer: Pricer) -> Reservation | None:
    """The operation's worst-case tokens and cost at ``pricer``'s prices.

    The input is the request's character count (``content_size_hint``, rounded up to tokens) plus
    an allowance per media part; the output is ``declared_max_output_tokens`` plus, per declared
    image, the tokens its provider publishes for it (``declared_image_tokens``) or an allowance.
    ``None`` when the pricer cannot price it (it raises, or the cost is unknown): an unpriced
    model, or a provider-hosted tool.
    """
    if request.kind != "llm":
        return _priced(request, Usage(), pricer, 0, 0)
    input_tokens = math.ceil((request.content_size_hint or 0) / CHARS_PER_TOKEN)
    input_tokens += request.non_text_parts * MEDIA_TOKEN_ALLOWANCE
    output_tokens = request.declared_max_output_tokens or 0
    per_image = request.declared_image_tokens
    image_tokens = request.declared_images * (
        per_image if per_image is not None else IMAGE_OUTPUT_TOKEN_ALLOWANCE
    )
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        image_output_tokens=image_tokens,
        image_count=request.declared_images,
    )
    return _priced(request, usage, pricer, input_tokens, output_tokens + image_tokens)


def _priced(
    request: OperationRequest,
    usage: Usage,
    pricer: Pricer,
    input_tokens: int,
    output_tokens: int,
) -> Reservation | None:
    try:
        cost = pricer.price(request, usage)
    except Exception:
        return None
    if cost.kind == "unknown" or cost.amount is None:
        return None
    return Reservation(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost.amount)
