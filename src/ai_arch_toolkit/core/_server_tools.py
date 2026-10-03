"""Server-side tool types (web search, code execution, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_arch_toolkit.core._exceptions import RequestError
from ai_arch_toolkit.core._images import ImageFormat, ImageRequest, ImageResolution


@dataclass(frozen=True, slots=True)
class ServerTool:
    """A server-side tool managed by the LLM provider.

    Unlike function tools, server tools are executed by the provider's
    infrastructure (e.g. web search, code interpreter).

    .. todo:: ``config`` is spread into the wire dict by ``prepare_tools()``
       but currently ignored by all provider implementations. Add
       provider-specific config forwarding (e.g. Anthropic ``max_uses``,
       OpenAI ``allowed_domains``) or remove config until the shapes are known.
    """

    type: str
    config: dict[str, Any] = field(default_factory=dict)


def web_search(**config: Any) -> ServerTool:
    """Create a web search server tool."""
    return ServerTool(type="web_search", config=config)


def code_execution(**config: Any) -> ServerTool:
    """Create a code execution/interpreter server tool."""
    return ServerTool(type="code_execution", config=config)


def image_generation(
    *,
    model: str,
    quality: str | None = None,
    aspect_ratio: str | None = None,
    resolution: ImageResolution | None = None,
    output_format: ImageFormat | None = None,
    partial_images: int | None = None,
) -> ServerTool:
    """A hosted image generation tool: the model draws in the middle of its turn (D46).

    OpenAI's Responses API runs it; the images come back in ``Response.images`` and, streamed, as
    ``image`` events. ``model`` is the image model that draws (OpenAI's own default is an old
    one); the other options are ``LLM.generate_image``'s (D47), and the adapter refuses what the
    image model does not take.

    Args:
        model: The image model, such as ``"gpt-image-2.5-flare"``.
        quality: The image model's quality level.
        aspect_ratio: Width to height, as ``"16:9"``.
        resolution: ``"1K"``, ``"2K"`` or ``"4K"``.
        output_format: ``"png"``, ``"jpeg"`` or ``"webp"``.
        partial_images: How many previews a stream gets while it draws, 0 to 3.
    """
    if not model:
        raise RequestError("image_generation needs the image model it draws with")
    ImageRequest(
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        quality=quality,
        output_format=output_format,
    )
    if partial_images is not None and partial_images not in range(4):
        raise RequestError(f"partial_images must be 0 to 3, got {partial_images!r}")
    options = {
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "output_format": output_format,
        "partial_images": partial_images,
    }
    config = {"model": model, **{k: v for k, v in options.items() if v is not None}}
    return ServerTool(type="image_generation", config=config)
